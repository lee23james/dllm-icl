#专门为gsm8k构建一个n-shot提示
#同时要注意query的位置，尽量保持
import random
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class BasePromptConstructor(ABC):
    """基类，定义了构建提示词的抽象方法"""
    def __init__(self, n_shots: int=4, random_seed: int = 42):
        """
        初始化指令构造器
        args:
        n_shots: 用于icl的示例数量
        random_seed: 用于可重现抽样的随机种子
        """
        self.n_shots = n_shots
        self.random_seed = random_seed
        random.seed(random_seed)
    
    #基类是随机采样
    @abstractmethod
    def construct_prompt(self, train_samples: List[Dict], test_sample: Dict) -> str:
        """Construct a prompt for the given task."""
        pass
    
    def sample_examples(self, train_samples: List[Dict], n_shots: int) -> List[Dict]:
        """Sample n_shots examples from train_samples."""
        if len(train_samples) <= n_shots:
            return train_samples
        return random.sample(train_samples, n_shots)