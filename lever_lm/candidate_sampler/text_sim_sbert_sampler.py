import os
from typing import Any, List

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from .base_sampler import BaseSampler


class TextSimSbertSampler(BaseSampler):
    """
    基于 SBERT 的文本相似度采样器

    功能：
    - 使用 SentenceTransformer 对数据集中指定字段进行编码（文本 → 向量）
    - 对每个 anchor 样本，在全数据集中检索最相似的 top-k 样本，作为候选集
    - 结果格式与 RandSampler 一致：
        {
            "anchor_set": [idx_1, idx_2, ...],
            "candidate_set": {
                idx_1: [cand_1, cand_2, ...],
                ...
            }
        }
    """

    def __init__(
        self,
        candidate_num: int,
        sampler_name: str,
        anchor_sample_num: int,
        index_ds_len: int,
        cache_dir: str,
        dataset_name: str,
        overwrite: bool,
        sbert_model_name: str,
        feature_cache_filename: str,
        text_field_name: str,
        device: str = "cuda:0",
        candidate_set_encode_bs: int = 64,
        anchor_idx_list: List[int] | None = None,
    ) -> None:
        # 注意：other_info 用于区分不同特征缓存（比如不同模型 / 不同字段）
        super().__init__(
            candidate_num=candidate_num,
            sampler_name=sampler_name,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            overwrite=overwrite,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            other_info=feature_cache_filename,
            anchor_idx_list=anchor_idx_list,
        )
        self.sbert_model_name = sbert_model_name
        self.feature_cache_filename = feature_cache_filename
        self.feature_cache = os.path.join(self.cache_dir, self.feature_cache_filename)
        self.text_field_name = text_field_name
        self.device = device
        self.bs = candidate_set_encode_bs

    @torch.inference_mode()
    def _encode_corpus(self, train_ds) -> torch.Tensor:
        """
        使用 SBERT 对整个数据集编码，返回 shape = (N, D) 的向量。
        """
        if os.path.exists(self.feature_cache) and not self.overwrite:
            logger.info(f"feature cache {self.feature_cache} exists, loading...")
            features = torch.load(self.feature_cache, map_location=self.device)
            # 确保是 float32 tensor
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32, device=self.device)
            return features

        logger.info(
            f"feature cache {self.feature_cache} not exists or overwrite=True, encoding with SBERT ({self.sbert_model_name})..."
        )
        model = SentenceTransformer(self.sbert_model_name, device=self.device)

        texts: List[str] = []
        for i in range(len(train_ds)):
            sample = train_ds[i]
            text = sample.get(self.text_field_name, "")
            if not isinstance(text, str):
                text = str(text)
            texts.append(text)

        all_embs: List[torch.Tensor] = []
        for start in range(0, len(texts), self.bs):
            end = min(start + self.bs, len(texts))
            batch_texts = texts[start:end]
            # encode 返回的是 numpy 或 tensor，这里统一为 torch.Tensor
            emb = model.encode(
                batch_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
                normalize_embeddings=True,  # 归一化后内积等价于 cosine
            )
            all_embs.append(emb)

        features = torch.cat(all_embs, dim=0).to(torch.float32)
        logger.info(f"saving the SBERT features cache in {self.feature_cache} ...")
        torch.save(features, self.feature_cache)
        return features

    @torch.inference_mode()
    def sample(self, anchor_set, train_ds) -> Any:
        """
        对每个 anchor，在全数据集中基于 SBERT 向量检索最相似的 candidate_num 个样本。
        """
        features = self._encode_corpus(train_ds)  # shape: (N, D)

        # 取出 anchor 的向量
        anchor_indices = torch.tensor(anchor_set, dtype=torch.long, device=self.device)
        anchor_features = features[anchor_indices]  # (A, D)

        # 相似度：归一化后用内积，相当于 cosine
        # sim[i, j] = <anchor_i, sample_j>
        sim = anchor_features @ features.t()  # (A, N)

        # 对每个 anchor 取 top_k，相似度最高的可能是自身，后面过滤
        top_k = self.candidate_num + 1
        scores, idxs = torch.topk(sim, k=min(top_k, sim.size(1)), dim=-1)

        candidate_set_idx: dict[int, list[int]] = {}
        for anchor_idx_val, cand_indices_row in zip(anchor_set, idxs.tolist()):
            # 去掉自身索引（如果存在）
            filtered: list[int] = []
            for cand_idx in cand_indices_row:
                if cand_idx == anchor_idx_val:
                    continue
                filtered.append(cand_idx)
                if len(filtered) >= self.candidate_num:
                    break

            # 如果因为各种原因不够 candidate_num，就随机补齐（避免后续代码出错）
            if len(filtered) < self.candidate_num:
                all_indices = list(range(len(train_ds)))
                # 去掉 anchor 本身与已选
                all_indices = [
                    i for i in all_indices if i != anchor_idx_val and i not in filtered
                ]
                extra_needed = self.candidate_num - len(filtered)
                if extra_needed > 0 and len(all_indices) > 0:
                    import random

                    extra = random.sample(
                        all_indices, min(extra_needed, len(all_indices))
                    )
                    filtered.extend(extra)

            candidate_set_idx[anchor_idx_val] = filtered

        return candidate_set_idx

