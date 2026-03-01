import torch
from torch import nn
# Inheriting from the structure used in your previous models
from .base_lever_lm import BaseLeverLM


# 支持 few_shot=3（序列长 5）与 few_shot=4（序列长 6），预留余量
MAX_SEQ_LEN = 8


class InsertionSelector(BaseLeverLM):
    def __init__(self, emb_dim, n_head, dropout=0.1):
        super().__init__()

        # We need BOS and EOS as learnable parameters since we use offline embeddings
        self.bos_embed = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.eos_embed = nn.Parameter(torch.randn(1, 1, emb_dim))

        # 类型嵌入：区分「上下文」(BOS,E1..En,EOS) 与「待插入的 query」
        self.context_type_embed = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.query_type_embed = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, emb_dim) * 0.02)

        # 输入归一化：避免 embedding 尺度过大导致梯度饱和
        self.input_norm = nn.LayerNorm(emb_dim)

        # Layer 1: Self-Attention (Sequence interacts with itself)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_head,
            dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True
        )

        # Layer 2: Cross-Attention (slots as query, full sequence as key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=n_head,
            dropout=dropout, batch_first=True
        )

        # Layer 3: FFN for final gap scoring（FFN 前 LayerNorm + GELU 减轻死神经元）
        self.ffn_norm = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, 1)  # Outputs a scalar score for each slot
        )
        # 零初始化最后一层：使初始 logits 全为 0，softmax 均匀，避免「永远预测位置 0」
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_embed, icd_embeds, labels=None):
        """
        query_embed: [batch_size, emb_dim]
        icd_embeds: [batch_size, num_examples, emb_dim]
        labels: The index of the correct gap (0 to N)
        """
        bs = icd_embeds.size(0)

        # 1. 序列 = [BOS, E1, ..., En, EOS, QUERY]，query 放最后
        bos = self.bos_embed.expand(bs, -1, -1)
        eos = self.eos_embed.expand(bs, -1, -1)
        query_vec = query_embed.unsqueeze(1)
        full_seq = torch.cat([bos, icd_embeds, eos, query_vec], dim=1)
        # 2. 类型嵌入 + 位置编码
        full_seq[:, :-1, :] = full_seq[:, :-1, :] + self.context_type_embed
        full_seq[:, -1:, :] = full_seq[:, -1:, :] + self.query_type_embed
        seq_len = full_seq.size(1)
        full_seq = full_seq + self.pos_embed[:, :seq_len, :]
        full_seq = self.input_norm(full_seq)

        # 3. Self-Attention：整段序列（含 query）一起交互
        seq_features = self.self_attn(full_seq)
        num_gaps = icd_embeds.size(1) + 1

        # 4. Cross-Attention：槽位作为 query，整段序列（含 query 标记）作为 key/value，保留槽位自身信息并引用查询与上下文
        attn_output, _ = self.cross_attn(
            query=seq_features[:, :num_gaps, :],
            key=seq_features,
            value=seq_features,
        )

        # 5. 将生成的特征传递给前馈神经网络 (FFN)
        gap_features = self.ffn_norm(attn_output)
        logits = self.ffn(gap_features).squeeze(-1) 

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.criterion(logits, labels)

        return output
