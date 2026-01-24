"""
LLaDA接口实现：负责GSM8K等任务的prompt构建
"""
from typing import Dict, List, Any
import torch
from loguru import logger

from .base_interface import BaseInterface


class LLaDAInterface(BaseInterface):
    """
    LLaDA接口实现
    负责ICD的拼接和mask操作
    如果isquery=1，需要对query部分进行mask操作
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        task: str,
        mask_id: int = 126336,
        mask_length: int = 256,
        split_token: str = "\n\n",
    ):
        """
        初始化LLaDA接口
        
        Args:
            model: LLaDA模型
            tokenizer: tokenizer
            task: 任务名称（如'gsm8k'）
            mask_id: mask token的ID
            mask_length: mask token的长度
            split_token: ICD之间的分隔符
        """
        super().__init__(model, tokenizer, task, mask_id, mask_length)
        self.split_token = split_token
        # 获取mask token字符串（用于构建prompt）
        try:
            self.mask_token_str = self.tokenizer.decode([self.mask_id])
        except:
            # 如果解码失败，使用默认值
            self.mask_token_str = "<|mdm_mask|>"
            logger.warning(f"Failed to decode mask_id {self.mask_id}, using default: {self.mask_token_str}")
    
    def _format_icd(self, ice: Dict[str, Any]) -> str:
        """
        格式化单个ICD示例
        
        Args:
            ice: ICD示例字典
        
        Returns:
            格式化后的ICD文本
        """
        # 确保不是query（isquery=0）
        if ice.get("isquery", 0) == 1:
            logger.warning("Found query in ice_samples, skipping...")
            return None
        
        # 获取ICD的文本（根据任务不同，字段可能不同）
        if "q_a" in ice:
            # 获取ice的qa版本
            icd_text = ice["q_a"]
        elif "question" in ice and "answer" in ice:
            # 适配GSM8K格式：question + answer,这个后期再改,,目前我想先按这个格式进行生成
            # 目前其他的问题我都可以按照这种格式生成问题和答案对
            icd_text = f"question: {ice['question']}\n<answer>\n{ice['answer']}\n</answer>"
        else:
            raise ValueError(f"Unknown data format for ICD: {ice.keys()}")
        
        return icd_text
    
    #这里取出test对样本并把query部分进行遮盖
    def _format_query(self, test_sample: Dict[str, Any]) -> str:
        """
        格式化query（需要mask操作）
        
        Args:
            test_sample: 测试样本（query）
        
        Returns:
            格式化后的query文本（包含mask token）
        """
        # 确保test_sample是query（isquery=1）
        if test_sample.get("isquery", 0) != 1:
            logger.warning("test_sample is not marked as query (isquery=1), but proceeding...")
        
        if "question" not in test_sample:
            raise ValueError(f"Unknown data format for query: {test_sample.keys()}")
        
        # 构建mask tokens（重复mask_length次）
        mask_tokens = self.mask_token_str * self.mask_length
        
        # 构建query文本，answer部分用mask token替换
        # GSM8K格式：question + target + mask_tokens
        #这里采用gsm8k格式,而且我觉得这个格式也可以适用于其他字符串中
        query_text = f"question: {test_sample['question']}\n<answer>\n{mask_tokens}"
        
        return query_text
    
    def build_prompt(
        self,
        ice_samples: List[Dict[str, Any]],
        test_sample: Dict[str, Any],
        query_position: int = 0,
    ) -> str:
        """
        构建ICL prompt
        负责ICD的拼接和mask操作
        
        Args:
            ice_samples: ICD示例列表（每个示例是一个字典）
            test_sample: 测试样本（query，isquery=1）
            query_position: query在序列中的位置
                - 0: query在最后
                - len(ice_samples): query在最前
                - 其他: query在中间指定位置
        
        Returns:
            prompt字符串
        """
        # 1. 构建ICD部分
        icd_parts = []
        for ice in ice_samples:
            icd_text = self._format_icd(ice)
            if icd_text is not None:
                icd_parts.append(icd_text)
        
        # 2. 构建query部分（需要mask操作）
        query_text = self._format_query(test_sample)
        
        # 3. 根据query_position插入query
        #按照习惯插入固定的query形式
        if query_position == 0:
            #和固定形式的反着来
            # query在最前
            prompt_parts = [query_text]+icd_parts 
        elif query_position == len(ice_samples):
            # query在最后
            prompt_parts =  icd_parts+ [query_text]
        else:
            # query在中间
            prompt_parts = (
                icd_parts[:query_position] + 
                [query_text] + 
                icd_parts[query_position:]
            )
        
        # 4. 使用split_token连接
        prompt = self.split_token.join(prompt_parts)
        
        return prompt
    
    def generate(self, prompt: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        使用LLaDA生成文本
        
        Args:
            prompt: tokenized prompt tensor, shape (1, L)
            **kwargs: 其他生成参数（会覆盖默认参数）
        
        Returns:
            生成的序列 tensor
        """
        # 合并生成参数
        gen_kwargs = {
            "steps": kwargs.get("steps", 256),
            "gen_length": kwargs.get("gen_length", 256),
            "block_length": kwargs.get("block_length", 256),
            "temperature": kwargs.get("temperature", 0.0),
            "cfg_scale": kwargs.get("cfg_scale", 0.0),
            "remasking": kwargs.get("remasking", "low_confidence"),
        }
        gen_kwargs.update(kwargs)
        
        # 找到mask token的位置
        mask_positions = (prompt == self.mask_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            raise ValueError("No mask tokens found in prompt")
        
        first_mask_pos = mask_positions[1][0].item()
        gen_start = first_mask_pos
        
        # 导入LLaDA生成函数
        from src.generate import generate
        
        # 调用生成函数
        output = generate(
            self.model,
            prompt,
            gen_start,
            steps=gen_kwargs["steps"],
            gen_length=gen_kwargs["gen_length"],
            block_length=gen_kwargs["block_length"],
            temperature=gen_kwargs["temperature"],
            cfg_scale=gen_kwargs["cfg_scale"],
            remasking=gen_kwargs["remasking"],
            mask_id=self.mask_id,
        )
        
        return output

