"""
基础接口类：用于构建prompt和调用模型
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
from datasets import Dataset
from loguru import logger

#这里我认为应该要做的就是进行prompt的准备
#然后由子类变成适合输入模型的prompt
class BaseInterface(ABC):
    """
    基础接口类：定义ICL推理的通用接口
    子类需要实现具体的prompt构建和模型调用逻辑
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        task: str,
        mask_id: int = 126336,
        mask_length: int = 256,
    ):
        """
        初始化接口
        
        Args:
            model: 模型对象
            tokenizer: tokenizer对象
            task: 任务名称（如'gsm8k'）
            mask_id: mask token的ID
            mask_length: mask token的长度（用于生成）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.mask_id = mask_id
        self.mask_length = mask_length
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda')
    
    @abstractmethod
    def build_prompt(
        self,
        ice_samples: List[Dict[str, Any]],
        test_sample: Dict[str, Any],
        query_position: int = 0,
    ) -> str:
        """
        构建ICL prompt（抽象方法，子类需要实现）
        
        Args:
            ice_samples: ICD示例列表（每个示例是一个字典）
            test_sample: 测试样本（query）
            query_position: query在序列中的位置（0表示最后，len(ice_samples)表示最前）
        
        Returns:
            prompt字符串
        """
        raise NotImplementedError
    
    def tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """
        将prompt tokenize为tensor
        
        Args:
            prompt: prompt字符串
        
        Returns:
            tokenized tensor, shape (1, L)
        """
        # 直接解码，不进行检查
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        
        return input_ids
    
    @abstractmethod
    def generate(self, prompt: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        生成文本（子类需要实现）
        
        Args:
            prompt: tokenized prompt tensor, shape (1, L)
            **kwargs: 其他生成参数
        
        Returns:
            生成的序列 tensor, shape (1, L')
        """
        raise NotImplementedError
    
    def decode_output(self, output: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        解码输出
        
        Args:
            output: 生成的tensor
            skip_special_tokens: 是否跳过特殊token
        
        Returns:
            解码后的文本
        """
        if output.dim() == 1:
            output = output.unsqueeze(0)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
        return decoded
    
    # def extract_answer(self, generated_text: str) -> str:
    #     """
    #     从生成的文本中提取答案（子类可以重写）
        
    #     Args:
    #         generated_text: 生成的文本
        
    #     Returns:
    #         提取的答案
    #     """
    #     # 默认实现：尝试从<answer>标签中提取
    #     import re
    #     match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
    #     if match:
    #         return match.group(1).strip()
    #     return generated_text.strip()

