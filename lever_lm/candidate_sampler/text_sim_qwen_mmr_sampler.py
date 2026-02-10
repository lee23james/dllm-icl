import os
from typing import Any, List, Dict

import torch
from loguru import logger
from datasets import Dataset

from .base_sampler import BaseSampler
from lever_lm.utils import encode_text_qwen, recall_and_rerank_mmr


class TextSimQwenMMRSampler(BaseSampler):
    """
    基于 Qwen-Embedding + MMR 的文本相似度采样器
    
    功能：
    - 使用 Qwen3-Embedding-4B 对数据集中指定字段进行编码（文本 → 向量）
    - 对每个 anchor 样本，先用相似度粗筛出 m 个例子
    - 然后使用 MMR（Maximal Marginal Relevance）算法，同时考虑相似度和多样性
    - 最终选出 top-k 个例子作为候选集
    
    结果格式与 RandSampler 一致：
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
        qwen_model_path: str,
        qwen_model_name: str,
        feature_cache_filename: str,
        text_field_name: str,
        dataset_type: str = "auto",  # "auto", "mmlu", "gsm8k", etc.
        device: str = "cuda:0",
        encode_batch_size: int = 64,
        coarse_k: int = 200,  # 粗筛数量
        mmr_lambda: float = 0.1,  # MMR 多样性参数
        query_task_description: str | None = None,
        anchor_idx_list: List[int] | None = None,
    ) -> None:
        # 注意：other_info 用于区分不同特征缓存（比如不同模型 / 不同字段）
        # 这里我们把 coarse_k 和 mmr_lambda 也编码进 candidate_set 的 cache 文件名中，
        # 保证不同检索超参数不会错误复用同一份候选缓存。
        other_info = f"coarse_k:{coarse_k}-lambda:{mmr_lambda}"
        super().__init__(
            candidate_num=candidate_num,
            sampler_name=sampler_name,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            overwrite=overwrite,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            other_info=other_info,
            anchor_idx_list=anchor_idx_list,
        )
        # qwen_model_path: 实际用于 AutoModel.from_pretrained 的本地/远程路径
        # qwen_model_name: 仅用于日志与 cache 文件名中的简短标识
        self.qwen_model_path = qwen_model_path
        self.qwen_model_name = qwen_model_name
        self.feature_cache_filename = feature_cache_filename
        self.feature_cache = os.path.join(self.cache_dir, self.feature_cache_filename)
        self.text_field_name = text_field_name
        self.dataset_type = dataset_type
        self.device = device
        self.encode_batch_size = encode_batch_size
        self.coarse_k = max(coarse_k, candidate_num)  # 确保 coarse_k >= candidate_num
        self.mmr_lambda = mmr_lambda
        # Query 侧的任务描述，用于构造 Qwen 官方推荐的 Instruct 格式
        # 如果未显式指定，使用一个通用的检索任务描述
        if query_task_description is None:
            self.query_task_description = (
                "Given a query, retrieve in-context examples that are most helpful for solving the query"
            )
        else:
            self.query_task_description = query_task_description

    def _format_text(self, sample: Dict) -> str:
        """
        根据数据集类型格式化文本（用于 embedding）
        
        当前设计尽量简单、与 PromptTemplate.generate_text_for_embedding 的新逻辑对齐：
        - 对于 MMLU：使用已展开的字段 question + choice_a~choice_d 构造“问题 + 选项”文本
        - 其他任务：直接使用 text_field_name 对应字段
        
        未覆盖到的情况一律报错，避免隐式兜底逻辑。
        
        Args:
            sample: 数据样本字典
        
        Returns:
            格式化后的文本字符串（通常不含答案）
        """
        # MMLU：依赖 load_mmlu_ds 展开的字段 question + choice_a~d
        if self.dataset_type == "mmlu":
            required_keys = ["question", "choice_a", "choice_b", "choice_c", "choice_d"]
            missing = [k for k in required_keys if k not in sample]
            if missing:
                raise ValueError(
                    f"MMLU sample missing required keys {missing} for embedding text formatting. "
                    f"Got keys: {list(sample.keys())}"
                )
            
            question = str(sample["question"]).strip()
            choice_a = str(sample["choice_a"])
            choice_b = str(sample["choice_b"])
            choice_c = str(sample["choice_c"])
            choice_d = str(sample["choice_d"])
            
            options_text = (
                f"A. {choice_a}\n"
                f"B. {choice_b}\n"
                f"C. {choice_c}\n"
                f"D. {choice_d}"
            )
            
            return f"{question}\n{options_text}".strip()
        
        # 其他任务：直接使用指定字段
        if self.text_field_name not in sample:
            raise ValueError(
                f"text_field_name '{self.text_field_name}' not found in sample keys: {list(sample.keys())}"
            )
        
        text = sample[self.text_field_name]
        if not isinstance(text, str):
            text = str(text)
        return text

    @torch.inference_mode()
    def _encode_corpus(self, train_ds: Dataset) -> torch.Tensor:
        """
        使用 Qwen-Embedding 对整个数据集编码，返回 shape = (N, D) 的向量。
        """
        if os.path.exists(self.feature_cache) and not self.overwrite:
            logger.info(f"Feature cache {self.feature_cache} exists, loading...")
            features = torch.load(self.feature_cache, map_location=self.device)
            # 确保是 float32 tensor
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32, device=self.device)
            logger.info(f"Loaded features with shape {features.shape}")
            return features

        logger.info(
            f"Feature cache {self.feature_cache} not exists or overwrite=True, "
            f"encoding with Qwen-Embedding model '{self.qwen_model_name}' from '{self.qwen_model_path}'..."
        )

        # 1. 数据预处理：将数据集转为文本列表
        texts: List[str] = []
        logger.info(f"Formatting {len(train_ds)} samples...")
        for i in range(len(train_ds)):
            sample = train_ds[i]
            text = self._format_text(sample)
            texts.append(text)

        # 2. 使用 Qwen-Embedding 编码
        logger.info(f"Encoding {len(texts)} texts with Qwen-Embedding...")
        features = encode_text_qwen(
            text_list=texts,
            model_name=self.qwen_model_path,
            device=self.device,
            batch_size=self.encode_batch_size,
            normalize=True,  # 归一化以便使用余弦相似度
        )

        # 3. 保存到缓存
        logger.info(f"Saving Qwen features cache to {self.feature_cache}...")
        torch.save(features, self.feature_cache)

        return features

    @torch.inference_mode()
    def sample(self, anchor_set: List[int], train_ds: Dataset) -> Any:
        """
        对每个 anchor，使用两阶段检索（粗筛 + MMR 细筛）选择候选集。
        
        Args:
            anchor_set: Anchor 索引列表
            train_ds: 训练数据集
        
        Returns:
            candidate_set_idx: dict[int, list[int]]，每个 anchor 对应的候选索引列表
        """
        # 1. 对整个数据集编码（只编码一次，作为“documents”）
        #    features_docs: shape = (N, D)
        features_docs = self._encode_corpus(train_ds)
        features_docs = features_docs.to(self.device)

        # 2. 单独对 anchor 编码（作为“queries”），使用 Qwen 官方推荐的 Instruct 格式：
        #    "Instruct: <task_description>\nQuery: <formatted_text>"
        anchor_texts: List[str] = []
        for idx in anchor_set:
            sample = train_ds[idx]
            base_text = self._format_text(sample)  # 与 documents 一致的基础文本
            query_text = f"Instruct: {self.query_task_description}\nQuery:{base_text}"
            anchor_texts.append(query_text)

        # 使用同一个 Qwen embedding 模型对 anchor_texts 进行编码
        anchor_features = encode_text_qwen(
            text_list=anchor_texts,
            model_name=self.qwen_model_path,
            device=self.device,
            batch_size=self.encode_batch_size,
            normalize=True,
        )
        anchor_features = anchor_features.to(self.device)

        # 3. 两阶段检索：粗筛 + MMR 细筛
        logger.info(
            f"Performing two-stage retrieval: coarse_k={self.coarse_k}, "
            f"final_k={self.candidate_num}, mmr_lambda={self.mmr_lambda}"
        )
        
        selected_indices_list = recall_and_rerank_mmr(
            query_vecs=anchor_features,  # Query 向量
            candidate_vecs=features_docs,  # 候选池向量（documents）
            top_k=self.candidate_num,  # 最终目标数量
            coarse_k=self.coarse_k,  # 粗筛数量
            lambda_param=self.mmr_lambda,  # MMR 多样性参数
            exclude_indices=anchor_set,  # 排除 anchor 本身
        )

        # 4. 打包返回结果
        candidate_set_idx: Dict[int, List[int]] = {}
        for anchor_idx, selected_indices in zip(anchor_set, selected_indices_list):
            if len(selected_indices) < self.candidate_num:
                logger.warning(
                    f"Anchor {anchor_idx} only got {len(selected_indices)} candidates "
                    f"(expected {self.candidate_num}), padding with random samples..."
                )
                # 如果候选不足，随机补齐
                all_indices = list(range(len(train_ds)))
                exclude_set = set(selected_indices) | {anchor_idx}
                available = [i for i in all_indices if i not in exclude_set]
                if len(available) > 0:
                    import random
                    extra_needed = self.candidate_num - len(selected_indices)
                    extra = random.sample(available, min(extra_needed, len(available)))
                    selected_indices.extend(extra)
            
            candidate_set_idx[anchor_idx] = selected_indices[:self.candidate_num]

        logger.info(
            f"Generated candidate sets for {len(candidate_set_idx)} anchors, "
            f"each with {self.candidate_num} candidates"
        )

        return candidate_set_idx
