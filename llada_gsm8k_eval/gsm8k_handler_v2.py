"""
GSM8K Dataset Handler v2
使用HuggingFace datasets的标准方法管理GSM8K数据集
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
from tqdm import tqdm


class GSM8KHandler:
    """GSM8K数据集处理器 - 使用HuggingFace标准方法 - 优化版本"""
    
    def __init__(self, data_dir: str = "/data/share/datasets/gsm8k"):
        """
        初始化GSM8K处理器
        
        Args:
            data_dir: 数据集存储目录
        """
        self.data_dir = Path(data_dir)
        self.processed_data_dir = Path("./processed_gsm8k_data")
        
        # 原始数据路径
        self.raw_train_path = self.data_dir / "train-00000-of-00001.parquet"
        self.raw_test_path = self.data_dir / "test-00000-of-00001.parquet"
        
        # 处理后的数据路径
        self.processed_train_path = self.processed_data_dir / "train"
        self.processed_test_path = self.processed_data_dir / "test"
        
        # 数据集URLs
        self.urls = {
            "train": "https://github.com/openai/gsm8k/raw/main/data/train.jsonl",
            "test": "https://github.com/openai/gsm8k/raw/main/data/test.jsonl"
        }
        
        # 性能优化：内存缓存
        self._dataset_cache = {}  # 缓存已加载的数据集
        self._file_exists_cache = {}  # 缓存文件存在性检查
        self._validated_cache = set()  # 缓存已验证的数据集

    def _check_file_exists(self, file_path: Path) -> bool:
        """缓存文件存在性检查"""
        path_str = str(file_path)
        if path_str not in self._file_exists_cache:
            self._file_exists_cache[path_str] = file_path.exists()
        return self._file_exists_cache[path_str]
    
    def load_raw_dataset(self, split: str = "train") -> Optional[Dataset]:
        """
        从原始parquet文件加载数据集 - 优化版本（带缓存）
        
        Args:
            split: 数据集分割 ("train" 或 "test")
            
        Returns:
            HuggingFace Dataset对象
        """
        # 优化：检查缓存
        cache_key = f"raw_{split}"
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]
        
        if split == "train":
            parquet_path = self.raw_train_path
        else:
            parquet_path = self.raw_test_path
        
        # 优化：使用缓存的文件存在性检查
        if not self._check_file_exists(parquet_path):
            print(f"Error: {split} parquet file not found at {parquet_path}")
            return None
        
        try:
            df = pd.read_parquet(parquet_path)
            dataset = Dataset.from_pandas(df)
            
            # 缓存结果
            self._dataset_cache[cache_key] = dataset
            return dataset
            
        except Exception as e:
            print(f"Error loading raw {split} dataset: {e}")
            return None
    
    def process_dataset(self, dataset: Dataset, split: str) -> Dataset:
        """
        处理数据集 - 精简版本（仅保留评估必需字段）
        
        Args:
            dataset: 原始数据集
            split: 数据集分割
            
        Returns:
            处理后的数据集
        """
        def process_example(example):
            # 仅保留评估必需的字段，避免冗余处理
            return {
                "question": example["question"].strip() if isinstance(example["question"], str) else str(example["question"]).strip(),
                "answer": example["answer"].strip() if isinstance(example["answer"], str) else str(example["answer"]).strip()
            }
        
        # 使用map函数处理整个数据集，移除不必要的progress bar
        processed_dataset = dataset.map(process_example, num_proc=4)  # 使用多进程加速
        
        return processed_dataset
    
    def save_processed_dataset(self, dataset: Dataset, split: str) -> bool:
        """
        保存处理后的数据集到磁盘
        
        Args:
            dataset: 处理后的数据集
            split: 数据集分割
            
        Returns:
            是否保存成功
        """
        try:
            save_path = self.processed_train_path if split == "train" else self.processed_test_path
            
            # 确保目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            dataset.save_to_disk(str(save_path))
            return True
            
        except Exception as e:
            print(f"Error saving processed {split} dataset: {e}")
            return False
    
    def load_processed_dataset(self, split: str = "train") -> Optional[Dataset]:
        """
        从磁盘加载处理后的数据集 - 优化版本（带缓存）
        
        Args:
            split: 数据集分割
            
        Returns:
            处理后的数据集
        """
        # 优化：检查缓存
        cache_key = f"processed_{split}"
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]
        
        if split == "train":
            load_path = self.processed_train_path
        else:
            load_path = self.processed_test_path
        
        # 优化：使用缓存的文件存在性检查
        if not self._check_file_exists(load_path):
            print(f"Processed {split} dataset not found at {load_path}")
            return None
        
        try:
            dataset = load_from_disk(str(load_path))
            
            # 缓存结果
            self._dataset_cache[cache_key] = dataset
            return dataset
            
        except Exception as e:
            print(f"Error loading processed {split} dataset: {e}")
            return None
    
    def get_dataset(self, split: str = "train", use_processed: bool = True) -> Optional[Dataset]:
        """
        获取数据集（优先使用处理后的版本）
        
        Args:
            split: 数据集分割
            use_processed: 是否优先使用处理后的数据
            
        Returns:
            数据集
        """
        if use_processed:
            # 尝试加载处理后的数据
            dataset = self.load_processed_dataset(split)
            if dataset is not None:
                return dataset
            
        
        # 加载原始数据并处理
        raw_dataset = self.load_raw_dataset(split)
        if raw_dataset is None:
            return None
        
        # 处理数据
        processed_dataset = self.process_dataset(raw_dataset, split)
        
        # 保存处理后的数据
        self.save_processed_dataset(processed_dataset, split)
        
        return processed_dataset
    
    def prepare_for_evaluation(self, test_split: str = "test", n_shots: int = 4) -> Tuple[Dataset, Dataset]:
        """
        为评估准备数据集 - 精简版本
        
        Args:
            test_split: 测试集分割
            n_shots: few-shot示例数量
            
        Returns:
            (train_dataset, test_dataset) 元组
        """
        # 直接加载数据集，减少打印输出
        train_dataset = self.get_dataset("train")
        if train_dataset is None:
            raise RuntimeError("Failed to load train dataset")
        
        test_dataset = self.get_dataset(test_split)
        if test_dataset is None:
            raise RuntimeError(f"Failed to load {test_split} dataset")
        
        return train_dataset, test_dataset
    
    def get_sample_data(self, split: str = "train", n_samples: int = 5) -> List[Dict[str, Any]]:
        """
        获取样本数据
        
        Args:
            split: 数据集分割
            n_samples: 样本数量
            
        Returns:
            样本数据列表
        """
        dataset = self.get_dataset(split)
        if dataset is None:
            return []
        
        return dataset.select(range(min(n_samples, len(dataset))))
    
    def validate_dataset(self, split: str = "train") -> bool:
        """
        验证数据集格式 - 精简版本（最小化验证）
        
        Args:
            split: 数据集分割
            
        Returns:
            是否验证通过
        """
        # 检查缓存，避免重复验证
        if split in self._validated_cache:
            return True
        
        dataset = self.get_dataset(split)
        if dataset is None:
            return False
        
        # 最小化验证：仅检查必要字段存在，跳过数据类型检查
        if "question" not in dataset.column_names or "answer" not in dataset.column_names:
            return False
        
        # 缓存验证结果
        self._validated_cache.add(split)
        return True
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Returns:
            数据集信息字典
        """
        info = {
            "data_dir": str(self.data_dir),
            "processed_dir": str(self.processed_data_dir),
            "raw_files": {},
            "processed_files": {},
            "datasets": {}
        }
        
        # 检查原始文件
        info["raw_files"]["train"] = self.raw_train_path.exists()
        info["raw_files"]["test"] = self.raw_test_path.exists()
        
        # 检查处理后的文件
        info["processed_files"]["train"] = self.processed_train_path.exists()
        info["processed_files"]["test"] = self.processed_test_path.exists()
        
        # 检查数据集大小和字段
        for split in ["train", "test"]:
            try:
                dataset = self.get_dataset(split)
                if dataset is not None:
                    info["datasets"][split] = {
                        "size": len(dataset),
                        "columns": dataset.column_names,
                        "features": str(dataset.features)  # 转换为字符串
                    }
                else:
                    info["datasets"][split] = {"error": "Failed to load"}
            except Exception as e:
                info["datasets"][split] = {"error": str(e)}
        
        return info
    
    def download_from_huggingface(self, split: str = "train") -> Optional[Dataset]:
        """
        从HuggingFace Hub下载数据集
        
        Args:
            split: 数据集分割
            
        Returns:
            数据集
        """
        try:
            print(f"Downloading {split} dataset from HuggingFace Hub...")
            dataset = load_dataset("gsm8k", "main", split=split)
            print(f"Successfully downloaded {split} dataset with {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            print(f"Error downloading {split} dataset from HuggingFace: {e}")
            return None


def main():
    """主函数，演示使用方法"""
    # 创建处理器
    handler = GSM8KHandler("/data/share/datasets/gsm8k")
    
    # 显示数据集信息
    print("=== Dataset Information ===")
    info = handler.get_dataset_info()
    print(json.dumps(info, indent=2))
    
    # 验证数据集
    print("\n=== Validating Datasets ===")
    handler.validate_dataset("train")
    handler.validate_dataset("test")
    
    # 准备评估数据
    print("\n=== Preparing for Evaluation ===")
    train_dataset, test_dataset = handler.prepare_for_evaluation("test", n_shots=4)
    
    # 显示样本数据
    print("\n=== Sample Data ===")
    samples = handler.get_sample_data("test", n_samples=3)
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"Question: {sample['question'][:500]}")
        print(f"Answer: {sample['answer'][:500]}")
        print(f"Question Length: {sample['question_length']}")
        print(f"Answer Length: {sample['answer_length']}")
        print(f"Has Calculation: {sample['has_calculation']}")
        print(f"Final Answer: {sample['final_answer']}")


if __name__ == "__main__":
    main()
