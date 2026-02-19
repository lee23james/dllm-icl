"""
GPT2LeverLM: GPT2-style Transformer for ICD sequence generation.
Uses precomputed Qwen-Embedding vectors (no CLIP). Adapter projects input_dim -> n_embd.
Qwen3-Embedding-4B output dim = 2560.
"""
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from .base_lever_lm import BaseLeverLM


class GPT2LeverLM(BaseLeverLM):
    """
    GPT2-based Lever LM for ICD sequence generation.
    Input: precomputed Qwen embedding (e.g. 2560-dim). Adapter projects to n_embd (e.g. 512).
    No CLIP; query/ICD features come from offline embedding only.
    """

    def __init__(
        self,
        lm_config: Dict[str, int],
        index_ds_size: int,
        input_dim: int = 2560,
        adapter: bool = True,
        norm: bool = False,
        freeze_prefix_list: Optional[List[str]] = None,
        query_encoding_flag: Optional[List[str]] = None,
        icd_encoding_flag: Optional[List[str]] = None,
        embedding_lookup: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            adapter=adapter,
            norm=norm,
            query_encoding_flag=query_encoding_flag or ["embedding"],
            icd_encoding_flag=icd_encoding_flag or ["embedding"],
        )
        n_embd = lm_config["n_embd"]
        n_head = lm_config["n_head"]
        n_layer = lm_config["n_layer"]

        vocab_size = index_ds_size + 3  # content ids + BOS + EOS + QUERY placeholder
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            eos_token_id=index_ds_size,
            bos_token_id=index_ds_size + 1,
        )
        self.lm_model = GPT2LMHeadModel(config)

        # No CLIP: single adapter from Qwen embedding dim -> n_embd
        if self._adapter:
            self.input_adapter = nn.Sequential(
                nn.Linear(input_dim, n_embd * 4),
                nn.ReLU(),
                nn.Linear(n_embd * 4, n_embd),
            )
        else:
            self.input_adapter = None

        # embedding_lookup: [vocab_size, input_dim]，用于 ICD 位置的 text embedding 查表
        # 与多模态版一致：每个位置 = 可学习向量(wte) + text embedding；纯 NLP 下 ICD 也用 text emb
        if embedding_lookup is not None:
            self.register_buffer("embedding_lookup", embedding_lookup)
        else:
            self.embedding_lookup = None
        self.freeze_prefix(freeze_prefix_list)

    def forward(
        self,
        query_embedding: torch.Tensor,
        icd_seq_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            query_embedding: [batch, input_dim] precomputed Qwen embedding per sample.
            icd_seq_idx: [batch, seq_len] token ids (BOS, QUERY_placeholder, icd_1, ..., EOS).

        Returns:
            lm_output (e.g. loss, logits) from GPT2LMHeadModel.
        """
        # 1. Base embeddings from token ids
        inputs_embeds = self.lm_model.transformer.wte(icd_seq_idx)

        # 2. Project query and add at position 1 (QUERY slot)：可学习向量 + query text emb
        if self._adapter and self.input_adapter is not None:
            query_feat = self.input_adapter(query_embedding)
        else:
            query_feat = query_embedding

        if self._norm:
            query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)

        inputs_embeds[:, 1] = inputs_embeds[:, 1] + query_feat

        # 3. Add ICD text embeddings at positions 2..L-1：可学习向量 + ICD text emb（与多模态版一致）
        # 仅对 content id (0..index_ds_size-1) 叠加；BOS/EOS/QUERY 为结构 token，不经过 adapter
        if self.embedding_lookup is not None:
            index_ds_size = self.embedding_lookup.shape[0] - 3
            icd_embeds = self.embedding_lookup[icd_seq_idx]  # [B, L, input_dim]
            if self._adapter and self.input_adapter is not None:
                icd_feats = self.input_adapter(icd_embeds)  # [B, L, n_embd]
            else:
                icd_feats = icd_embeds
            if self._norm:
                icd_feats = icd_feats / (icd_feats.norm(dim=-1, keepdim=True) + 1e-8)
            content_mask = (icd_seq_idx < index_ds_size).unsqueeze(-1).float()  # [B, L, 1]
            inputs_embeds[:, 2:] = inputs_embeds[:, 2:] + icd_feats[:, 2:] * content_mask[:, 2:]

        # 4. LM forward (allow caller to mask padding with labels=-100)
        if labels is None:
            labels = icd_seq_idx
        output = self.lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output

    @torch.inference_mode()
    def generation(
        self,
        query_embedding: torch.Tensor,
        shot_num: int,
        index_ds_size: int,
        device: torch.device,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Autoregressive ICD sequence generation (no CLIP / index_ds).
        Starts with [BOS, QUERY_placeholder] and appends shot_num ICD token ids.

        Args:
            query_embedding: [batch, input_dim] (e.g. 2560 for Qwen3-Embedding-4B).
            shot_num: number of ICD tokens to generate.
            index_ds_size: vocab size for content tokens (ids in [0, index_ds_size-1]).
            device: torch device.
            bos_token_id: defaults to index_ds_size.
            eos_token_id: defaults to index_ds_size + 1.

        Returns:
            List of sequences (each is [BOS, QUERY_ph, id_1, ..., id_shot_num]).
        """
        if bos_token_id is None:
            bos_token_id = index_ds_size
        if eos_token_id is None:
            eos_token_id = index_ds_size + 1
        query_placeholder_id = index_ds_size + 2

        batch_size = query_embedding.shape[0]
        icd_seq_idx = torch.tensor(
            [[bos_token_id, query_placeholder_id]] * batch_size,
            dtype=torch.long,
            device=device,
        )

        for _ in range(shot_num):
            out = self.forward(query_embedding, icd_seq_idx)
            logits = out["logits"][:, -1, :]
            # Mask special tokens and already used ids
            logits[:, index_ds_size:] = float("-inf")
            for b in range(batch_size):
                for j in range(icd_seq_idx.shape[1]):
                    logits[b, icd_seq_idx[b, j].item()] = float("-inf")
            next_token = logits.argmax(dim=-1, keepdim=True)
            icd_seq_idx = torch.cat([icd_seq_idx, next_token], dim=1)

        return icd_seq_idx.detach().cpu().tolist()
