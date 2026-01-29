"""
LLaDA接口实现：负责GSM8K等任务的prompt构建
"""
from typing import Dict, List, Any, Optional
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
        #这里需要统一字符串格式,变成可以统一识别的格式
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
            test_sample: 测试样本（query，isquery=1）,表示这是一个query
            query_position: query在序列中的位置（越大越靠前）
                - 0: query在最后面
                - len(ice_samples): query在最前面
                - 其他: 按反向顺序插入（数值越小越靠后）
        
        Returns:
            prompt字符串
        """
        # 1. 构建ICD部分,注意这里是根据position的位置来进行的
        icd_parts = []
        for ice in ice_samples:
            icd_text = self._format_icd(ice)
            if icd_text is not None:
                icd_parts.append(icd_text)
        
        # 2. 构建query部分（需要mask操作）
        query_text = self._format_query(test_sample)
        
        # 3. 根据query_position插入query（数值越大越靠前，越小越靠后）
        icd_count = len(icd_parts)
        max_position = icd_count
        # 将query_position限制在可用范围内，避免越界
        normalized_position = max(0, min(query_position, max_position))
        # 反向插入：0 -> 末尾，1 -> 倒数第二，...，len -> 开头
        insertion_idx = icd_count - normalized_position

        prompt_parts = icd_parts[:]
        prompt_parts.insert(insertion_idx, query_text)
        
        # 4. 使用split_token连接
        prompt = self.split_token.join(prompt_parts)
        
        return prompt
    
    #这里需要进行改动,因为现在我需要根据query_position的位置来生成答案,同时这里包装的是一个生成的过程
    def generate(self, prompt: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        使用LLaDA生成文本
        
        Args:
            prompt: tokenized prompt tensor, shape (1, L)
            **kwargs: 其他生成参数（会覆盖默认参数）
        
        Returns:
            生成的序列 tensor
        """
        # 合并生成参数,这里从参数列表中获取需要的参数
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
        
        # 导入LLaDA生成函数,利用之前的推理函数进行生成,这里会产出模型生成的回答
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
    
    # =========================
    # 置信度 / log-likelihood（LLaDA 特定实现）
    # =========================
    @torch.no_grad()
    def compute_confidence(
        self,
        prompt: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        method: str = "margin",
    ) -> torch.Tensor:
        """
        LLaDA 的置信度计算：对 prompt 做一次 forward，然后在指定 mask 位置上计算：
        - margin: top1_prob - top2_prob
        - top1: top1_prob
        最后对 mask 位置取平均，返回标量 tensor。
        """
        import torch.nn.functional as F

        prompt = prompt.to(self.device)
        logits = self.model(prompt).logits  # (1, L, vocab)
        probs = F.softmax(logits, dim=-1)

        if method == "margin":
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1 = sorted_probs[:, :, 0]
            top2 = sorted_probs[:, :, 1]
            conf = top1 - top2
        elif method == "top1":
            conf, _ = torch.max(probs, dim=-1)
        else:
            raise ValueError(f"Unknown confidence method: {method}")

        if mask_positions is not None:
            mask_positions = mask_positions.to(self.device)
            if mask_positions.dim() == 1:
                mask_positions = mask_positions.unsqueeze(0)
            conf = torch.where(mask_positions, conf, torch.tensor(0.0, device=self.device))
            denom = mask_positions.sum().float()
            return conf.sum() / denom if denom > 0 else torch.tensor(0.0, device=self.device)

        return conf.mean()

    @torch.no_grad()
    def compute_log_likelihood(
        self,
        prompt_left: torch.Tensor,
        answer: torch.Tensor,
        prompt_right: Optional[torch.Tensor] = None,
        mc_num: int = 128,
        batch_size: int = 1,
        cfg_scale: float = 0.0,
    ) -> float:
        """
        LLaDA 的 log-likelihood：直接调用仓库内 `src.eval_likelihood.get_log_likelihood`（MC 估计）。
        """
        from src.eval_likelihood import get_log_likelihood

        score = get_log_likelihood(
            model=self.model,
            prompt_left=prompt_left,
            answer=answer,
            prompt_right=prompt_right,
            mc_num=mc_num,
            batch_size=batch_size,
            cfg_scale=cfg_scale,
            mask_id=self.mask_id,
        )
        return float(score)

    #每个模型的计算蒙特卡洛的方式都是不同的
    def compute_score_with_mc(
        self,
        prompt_left: torch.Tensor,           # 1D: 左侧 prompt tokens
        answer: torch.Tensor,                # 1D: 答案 tokens
        prompt_right: Optional[torch.Tensor] = None,  # 1D: 右侧 prompt tokens（可为 None）
        mc_num: int = 128,
        batch_size: int = 1,
        cfg_scale: float = 0.0,
    ) -> float:
        """
        使用蒙特卡洛方法计算单个 prompt 的分数（log likelihood）
        
        直接参考并调用 `src/eval_likelihood.py` 中的 `get_log_likelihood`：
        - prompt_left:  左侧上下文（ICD + question 等），shape: (L_left,)
        - answer:       答案 token 序列，shape: (L_answer,)
        - prompt_right: 右侧上下文（可选），shape: (L_right,)
        
        Args:
            prompt_left: 左侧 prompt token 序列（1D tensor）
            answer: 答案 token 序列（1D tensor）
            prompt_right: 右侧 prompt token 序列（1D tensor 或 None）
            mc_num: Monte Carlo 采样次数
            batch_size: mini-batch 大小（这里一般用 1）
            cfg_scale: CFG scale
        
        Returns:
            单个 prompt 的 log likelihood 分数（float）
        """
        #这里直接用接口进行处理,根据左右的部分进行处理
        return float(
            self.compute_log_likelihood(
                prompt_left=prompt_left,
                answer=answer,
                prompt_right=prompt_right,
                mc_num=mc_num,
                batch_size=batch_size,
                cfg_scale=cfg_scale,
            )
        )
    
    #这里基本不用管,因为暂时用不到
    def compute_score_with_margin(
        self,
        ice_samples: List[Dict[str, Any]],  # ICD序列
        test_sample: Dict[str, Any],        # 测试样本（query，答案可以是mask的）
        query_position: int = 0,            # query在序列中的位置
    ) -> float:
        """
        使用margin方法计算单个prompt的分数（置信度差值）
        
        参考 src/generate.py 的 margin_function 实现
        计算 top1_prob - top2_prob 的平均值
        
        Args:
            ice_samples: ICD序列（列表，每个元素是字典）
            test_sample: 测试样本（query），答案可以是mask的
            query_position: query在序列中的位置
        
        Returns:
            score: 分数（margin置信度，标量）
        
        Note:
            这是未来可能需要的接口，目前不使用
        """
        # 1. 构建prompt
        prompt = self.build_prompt(
            ice_samples=ice_samples,
            test_sample=test_sample,
            query_position=query_position,
        )
        
        # 2. Tokenize prompt
        prompt_tensor = self.tokenize_prompt(prompt)
        
        # 3. 找到mask位置
        mask_positions = (prompt_tensor == self.mask_id)
        
        # 4. 计算置信度（使用margin方法）
        score = self.compute_confidence(
            prompt_tensor,
            mask_positions=mask_positions,
            method="margin",
        )
        
        return score.item()
    
    def compute_score_with_top1(
        self,
        ice_samples: List[Dict[str, Any]],  # ICD序列
        test_sample: Dict[str, Any],        # 测试样本（query，答案可以是mask的）
        query_position: int = 0,            # query在序列中的位置
    ) -> float:
        """
        使用top1概率方法计算单个prompt的分数
        
        Args:
            ice_samples: ICD序列（列表，每个元素是字典）
            test_sample: 测试样本（query），答案可以是mask的
            query_position: query在序列中的位置
        
        Returns:
            score: 分数（top1概率，标量）
        
        Note:
            这是未来可能需要的接口，目前不使用
        """
        # 1. 构建prompt
        prompt = self.build_prompt(
            ice_samples=ice_samples,
            test_sample=test_sample,
            query_position=query_position,
        )
        
        # 2. Tokenize prompt
        prompt_tensor = self.tokenize_prompt(prompt)
        
        # 3. 找到mask位置
        mask_positions = (prompt_tensor == self.mask_id)
        
        # 4. 计算置信度（使用top1方法）
        score = self.compute_confidence(
            prompt_tensor,
            mask_positions=mask_positions,
            method="top1",
        )
        
        return score.item()

