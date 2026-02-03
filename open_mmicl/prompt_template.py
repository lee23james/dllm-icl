"""
Prompt模板类：用于根据模板字符串生成prompt
"""
from typing import Dict, Any, Optional
import re
from loguru import logger


class PromptTemplate:
    """
    Prompt模板类：根据模板字符串生成prompt
    
    模板格式：
    - <Q>: 会被question替换
    - <A>: 会被answer替换（对于query，会被mask替换）
    """
    
    def __init__(
        self,
        prompt_template: str,
        mask_token_str: str = "<|mdm_mask|>",
        mask_length: int = 256,
    ):
        """
        初始化PromptTemplate
        
        Args:
            prompt_template: 模板字符串，如 "question: <Q>\n<answer>\n<A>\n</answer>"
            mask_token_str: mask token字符串
            mask_length: mask token的长度
        """
        self.prompt_template = prompt_template
        self.mask_token_str = mask_token_str
        self.mask_length = mask_length
        self.mask_tokens = mask_token_str * mask_length
    
    def generate_ice_item(self, sample: Dict[str, Any]) -> str:
        """
        生成单个ICD示例的prompt
        
        Args:
            sample: 样本字典，包含question和answer字段
        
        Returns:
            格式化后的ICD文本
        """
        # 提取question和answer
        if "q_a" in sample:
            # 如果已经有q_a字段，直接使用
            return sample["q_a"]
        elif "question" in sample and "answer" in sample:
            # 使用模板格式化
            question = sample["question"]
            answer = sample["answer"]
            # 替换模板中的<Q>和<A>
            prompt = self.prompt_template.replace("<Q>", question).replace("<A>", answer)
            return prompt
        else:
            raise ValueError(f"Unknown ICD format: {sample.keys()}")
    
    def generate_query_item(self, sample: Dict[str, Any], use_mask: bool = True) -> str:
        """
        生成query的prompt
        
        Args:
            sample: 样本字典，包含question字段（如果use_mask=False，还需要answer字段）
            use_mask: 是否使用mask替换answer部分
                - True: 推理阶段，answer部分用mask替换
                - False: 打分阶段，使用完整的answer（用于蒙特卡洛估计）
        
        Returns:
            格式化后的query文本
        """
        if use_mask:
            # 推理阶段：answer部分用mask替换
            if "q_a" in sample:
                # 如果只有q_a，提取question部分
                qa_text = sample["q_a"]
                if "<answer>" in qa_text:
                    # 提取question部分，然后加上<answer>\n和mask
                    question_part = qa_text.split("<answer>")[0].strip()
                    # 构建query prompt
                    prompt = self.prompt_template.replace("<Q>", question_part.replace("question:", "").strip())
                    prompt = prompt.replace("<A>", self.mask_tokens)
                    return prompt
                else:
                    return qa_text
            elif "question" in sample:
                question = sample["question"]
                # 替换模板中的<Q>和<A>（<A>用mask替换）
                prompt = self.prompt_template.replace("<Q>", question).replace("<A>", self.mask_tokens)
                return prompt
            else:
                raise ValueError(f"Unknown query format: {sample.keys()}")
        else:
            # 打分阶段：使用完整的answer（用于蒙特卡洛估计）
            if "q_a" in sample:
                # 如果已经有q_a字段，直接使用（包含完整answer）
                return sample["q_a"]
            elif "question" in sample and "answer" in sample:
                # 使用模板格式化，包含完整answer
                question = sample["question"]
                answer = sample["answer"]
                prompt = self.prompt_template.replace("<Q>", question).replace("<A>", answer)
                return prompt
            else:
                raise ValueError(f"Unknown query format for scoring: {sample.keys()}")
