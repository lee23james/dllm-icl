"""
随机检索器：从训练集中随机选择ICD
"""
import random
from typing import List, Dict, Any
from datasets import Dataset
from loguru import logger


class RandRetriever:
    """
    随机检索器：从训练集中随机选择ICD示例
    """
    
    def __init__(
        self,
        train_ds: Dataset,
        nshot: int = 4,
        seed: int = 42,
    ):
        """
        初始化随机检索器
        
        Args:
            train_ds: 训练数据集
            nshot: 要选择的ICD数量
            seed: 随机种子
        """
        self.train_ds = train_ds
        self.nshot = nshot
        self.seed = seed
        random.seed(seed)
        logger.info(f"RandRetriever initialized with nshot={nshot}, seed={seed}")
    
    def retrieve(
        self,
        test_sample: Dict[str, Any],
        exclude_indices: List[int] = None,
    ) -> List[int]:
        """
        为测试样本随机检索ICD索引
        
        Args:
            test_sample: 测试样本（当前未使用，但保留接口一致性）
            exclude_indices: 要排除的索引列表（例如，避免选择test_sample本身）
        
        Returns:
            ICD索引列表
        """
        exclude_indices = exclude_indices or []
        
        # 获取所有可用索引
        available_indices = list(range(len(self.train_ds)))
        
        # 排除指定的索引
        for idx in exclude_indices:
            if idx in available_indices:
                available_indices.remove(idx)
        
        # 随机选择nshot个索引
        if len(available_indices) < self.nshot:
            logger.warning(
                f"Available indices ({len(available_indices)}) < nshot ({self.nshot}), "
                f"returning all available indices"
            )
            return available_indices
        
        selected_indices = random.sample(available_indices, self.nshot)
        return sorted(selected_indices)
    
    def retrieve_batch(
        self,
        test_samples: List[Dict[str, Any]],
        exclude_indices_list: List[List[int]] = None,
    ) -> List[List[int]]:
        """
        批量检索ICD索引
        
        Args:
            test_samples: 测试样本列表
            exclude_indices_list: 每个测试样本要排除的索引列表
        
        Returns:
            ICD索引列表的列表
        """
        exclude_indices_list = exclude_indices_list or [[]] * len(test_samples)
        
        results = []
        for test_sample, exclude_indices in zip(test_samples, exclude_indices_list):
            icd_indices = self.retrieve(test_sample, exclude_indices)
            results.append(icd_indices)
        
        return results




