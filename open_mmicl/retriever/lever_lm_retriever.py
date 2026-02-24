"""
LeverLM 检索器：使用训练好的 LeverLM 模型为每个 query 生成 ICD 序列

与数据集解耦：通过 query_text_extractor 可调用对象提取 query 文本，不绑定具体数据集。
"""
from typing import List, Dict, Any, Optional, Callable

import torch
from datasets import Dataset
from loguru import logger

from lever_lm.utils import encode_text_qwen


class LeverLMRetriever:
    """
    使用训练好的 LeverLM 模型生成 ICD 序列的检索器。

    与数据集解耦：query 文本通过 query_text_extractor(sample) -> str 提取，
    支持任意数据格式，无需在 retriever 内硬编码 MMLU/GSM8K 等。
    """

    def __init__(
        self,
        index_ds: Dataset,
        lever_lm,  # GPT2LeverLM 等
        qwen_model_path: str,
        query_text_extractor: Callable[[Dict[str, Any]], str],
        nshot: int = 4,
        device: str = "cuda:0",
        batch_size: int = 32,
        reverse_seq: bool = False,
    ):
        """
        Args:
            index_ds: 索引数据集（训练集），用于确定 index_ds_size
            lever_lm: LeverLM 模型实例
            qwen_model_path: Qwen-Embedding 模型路径，用于编码 query 文本
            query_text_extractor: 从样本中提取 query 文本的可调用对象，签名 sample -> str
            nshot: 每个 query 检索的 ICD 数量
            device: 计算设备
            batch_size: 批量编码时的 batch size
            reverse_seq: 是否反转生成的 ICD 序列顺序
        """
        self.index_ds = index_ds
        self.lever_lm = lever_lm
        self.qwen_model_path = qwen_model_path
        self.query_text_extractor = query_text_extractor
        self.nshot = nshot
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.reverse_seq = reverse_seq
        self.index_ds_size = len(index_ds)
        self._qwen_device = device
        logger.info(
            f"LeverLMRetriever: nshot={nshot}, index_ds_size={self.index_ds_size}, reverse_seq={reverse_seq}"
        )

    def _extract_text(self, sample: Dict[str, Any]) -> str:
        """使用配置的 extractor 提取 query 文本"""
        return self.query_text_extractor(sample)

    @torch.inference_mode()
    def _encode_query(self, text: str) -> torch.Tensor:
        """用 Qwen 编码单个 query 文本"""
        feats = encode_text_qwen(
            text_list=[text],
            model_name=self.qwen_model_path,
            device=self._qwen_device,
            batch_size=1,
            normalize=True,
        )
        return feats[0:1]

    def _postprocess_ids(self, pred_ids: List[int]) -> List[int]:
        """后处理：截断、过滤、可选反转"""
        ids = [int(x) for x in pred_ids if x < self.index_ds_size][: self.nshot]
        if self.reverse_seq:
            ids = list(reversed(ids))
        return ids

    @torch.inference_mode()
    def retrieve(
        self,
        test_sample: Dict[str, Any],
        exclude_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        为测试样本使用 LeverLM 生成 ICD 索引序列

        Args:
            test_sample: 测试样本（任意格式，由 query_text_extractor 解析）
            exclude_indices: 要排除的索引（可选）

        Returns:
            ICD 索引列表 [id_1, id_2, ..., id_nshot]
        """
        exclude_indices = exclude_indices or []
        text = self._extract_text(test_sample)
        query_emb = self._encode_query(text).to(self.device)
        pred_seqs = self.lever_lm.generation(
            query_embedding=query_emb,
            shot_num=self.nshot,
            index_ds_size=self.index_ds_size,
            device=self.device,
        )
        pred_ids = pred_seqs[0][2 : 2 + self.nshot]
        return self._postprocess_ids(pred_ids)

    @torch.inference_mode()
    def retrieve_batch(
        self,
        test_samples: List[Dict[str, Any]],
        exclude_indices_list: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        """批量检索"""
        exclude_indices_list = exclude_indices_list or [[]] * len(test_samples)
        results = []
        for i in range(0, len(test_samples), self.batch_size):
            batch = test_samples[i : i + self.batch_size]
            texts = [self._extract_text(s) for s in batch]
            feats = encode_text_qwen(
                text_list=texts,
                model_name=self.qwen_model_path,
                device=self._qwen_device,
                batch_size=len(texts),
                normalize=True,
            )
            query_embs = feats.to(self.device)
            pred_seqs = self.lever_lm.generation(
                query_embedding=query_embs,
                shot_num=self.nshot,
                index_ds_size=self.index_ds_size,
                device=self.device,
            )
            for pred_full in pred_seqs:
                pred_ids = pred_full[2 : 2 + self.nshot]
                results.append(self._postprocess_ids(pred_ids))
        return results
