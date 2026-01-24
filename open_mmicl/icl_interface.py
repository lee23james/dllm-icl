# LLaDA ICL推理接口
# 作用:
# 一个 interface（模型 + prompt 模板）
# 一个 retriever（如何选 ICD）
# 训练集（用于选 ICD）
# 测试集（需要预测）
# 就帮你完成：
# 构造 prompt（包含mask token）
# 调LLaDA模型生成
# 收集结果
# 核心类：LLaDAICLInferencer
# pyright: reportMissingImports=false
from typing import Optional, List, Dict, Any
import os
import sys

import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
import numpy as np

# 添加项目路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块（路径在运行时动态添加，类型检查器无法识别）
# 这里默认使用项目根目录下的utils和src，封装DLLM（目前主要是LLaDA）的推理逻辑
from utils.eval_utils import query_extract  # type: ignore
from src.generate import (  # type: ignore
    generate,
    generate_with_fast_dllm,
    generate_with_conf_sampler,
    generate_with_pc_sampler,
)


class DLLMICLInferencer:
    """
    DLLM 的 ICL 推理接口类（当前实现主要封装 LLaDA）
    封装了 prompt 构建和 DLLM 生成逻辑，复用现有的 LLaDA 代码
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_ds: Dataset,
        test_ds: Dataset,
        task: str,
        generation_kwargs: Dict[str, Any],
        query_position: int = 0,
        nshot: Optional[int] = None,
        mode: str = "original",  # "original", "fast_dllm", "conf_sampler", "pc_sampler"
        situation: str = "base",  # "base" or "instruct"
        mask_id: int = 126336,
        other_save_field: Optional[List[str]] = None,
        output_json_filepath: Optional[str] = "./llada_icl_inference_output",
        output_json_filename: Optional[str] = "predictions",
    ) -> None:
        """
        初始化 DLLM ICL 推理器（当前主要支持 LLaDA）
        
        Args:
            model: LLaDA模型
            tokenizer: tokenizer
            train_ds: 训练数据集（用于ICD）
            test_ds: 测试数据集
            task: 任务名称（如'gsm8k', 'mbpp', 'sudoku'等）
            generation_kwargs: 生成参数，包含：
                - steps: 采样步数
                - gen_length: 生成长度
                - block_length: 块长度
                - temperature: 温度
                - lambd: (可选) conf_sampler参数
                - alpha: (可选) conf_sampler参数
                - baseline_name: (可选) conf_sampler参数
                - num: (可选) conf_sampler参数
                - thread: (可选) fast_dllm阈值
            query_position: query在ICD序列中的位置（0表示最后，nshot表示最前）
            nshot: few-shot示例数量
            mode: 生成模式
            situation: "base"或"instruct"
            mask_id: mask token的ID
            other_save_field: 需要保存的其他字段
            output_json_filepath: 输出文件路径
            output_json_filename: 输出文件名
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.task = task
        self.generation_kwargs = generation_kwargs
        self.query_position = query_position
        self.nshot = nshot
        self.mode = mode
        self.situation = situation
        self.mask_id = mask_id
        self.other_save_field = other_save_field or []
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename
        
        # 从generation_kwargs中提取参数
        self.steps = generation_kwargs.get("steps", 128)
        self.gen_length = generation_kwargs.get("gen_length", 128)
        self.block_length = generation_kwargs.get("block_length", 128)
        self.temperature = generation_kwargs.get("temperature", 0.0)
        self.lambd = generation_kwargs.get("lambd", 1)
        self.alpha = generation_kwargs.get("alpha", 1)
        self.baseline_name = generation_kwargs.get("baseline_name", "P_baseline.json")
        self.num = generation_kwargs.get("num", None)
        self.thread = generation_kwargs.get("thread", None)
        
        # 确保模型在正确的设备上
        self.device = next(model.parameters()).device
    #咱们构建不用太复杂了,因为本身只是一个想法的验证环节,只需要配置一个差不多的,如果idea成功,再想办法把配置做复杂详细点,耦合就耦合吧
    def _build_prompt(
        self,
        test_sample: Dict[str, Any],#这里有可能是一组也可能是一个
        ice_idx_list: List[int],
    ) -> torch.Tensor:
        """
        构建包含ICD和query的prompt（带mask token）
        
        Args:
            test_sample: 测试样本
            ice_idx_list: ICD索引列表
        
        Returns:
            prompt tensor: shape (1, L)
        """
        # 获取ICD样本
        #获取训练集上的对应icd样本
        ice_samples = [self.train_ds[idx] for idx in ice_idx_list]
        #下面这个需要去购买
        #=========================================
        # 构建prompt（复用现有代码）
        # query_extract会根据task、query_position、nshot构建prompt
        #这里需要改,因为生成prompt自带模版,看来真的需要重新去创造prompt的逻辑
        #这个并不是query,她是自带模版的prompt
        #这里我想在utils上进行改进,因为应该在后面加上的是mask token
        query = query_extract(
            test_sample,
            self.task,
            self.query_position,
            self.gen_length,
            self.nshot,
            #这里默认是False
            iscot=False
        )
        #=========================================
        # 处理base/instruct模式
        if self.situation == 'base':
            user_input = query
        elif self.situation == 'instruct':
            messages = [{"role": "user", "content": query}]
            user_input = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            raise ValueError(f"Unknown situation: {self.situation}")
        
        # 转换为token ids
        #转化为prompt_id准备进行推理
        prompt_ids = self.tokenizer(user_input)['input_ids']
        prompt = torch.tensor(prompt_ids).to(self.device).unsqueeze(0)
        
        return prompt
    
    #这里先进行耦合,特定于llada的推理
    #这个后期要放到固定模型中
    def _generate_with_llada(
        self,
        prompt: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用LLaDA生成（复用src/generate.py中的函数）
        
        Args:
            prompt: prompt tensor, shape (1, L)
        
        Returns:
            生成的序列 tensor, shape (1, L')
        """
        # 找到mask token的位置
        mask_positions = (prompt == self.mask_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            raise ValueError("No mask tokens found in prompt")
        
        first_mask_pos = mask_positions[1][0].item()
        last_mask_pos = mask_positions[1][-1].item()
        gen_start = first_mask_pos
        #根据特定的参数代入llada模型进行计算
        # 根据mode选择不同的生成函数
        if self.mode == 'original':
            output = generate(
                self.model,
                prompt,
                gen_start,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=0.0,
                remasking='low_confidence',
                mask_id=self.mask_id,
            )
        elif self.mode == 'fast_dllm':
            if self.thread is None:
                raise ValueError("fast_dllm mode requires 'thread' parameter in generation_kwargs")
            output = generate_with_fast_dllm(
                self.model,
                prompt,
                gen_start,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=0.0,
                remasking='low_confidence',
                threshold=self.thread,
                mask_id=self.mask_id,
            )[0]  # fast_dllm返回tuple，取第一个元素
        elif self.mode == 'conf_sampler':
            output = generate_with_conf_sampler(
                self.model,
                prompt,
                gen_start,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                lambd=self.lambd,
                alpha=self.alpha,
                baseline_name=self.baseline_name,
                temperature=self.temperature,
                cfg_scale=0.0,
                remasking='low_confidence',
                num=self.num,
                mask_id=self.mask_id,
            )
        elif self.mode == 'pc_sampler':
            output = generate_with_pc_sampler(
                self.model,
                prompt,
                gen_start,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                lambd=self.lambd,
                alpha=self.alpha,
                baseline_name=self.baseline_name,
                temperature=self.temperature,
                cfg_scale=0.0,
                remasking='low_confidence',
                mask_id=self.mask_id,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return output
    
    @torch.inference_mode()
    def inference(
        self,
        ice_idx_list: List[List[int]],
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行ICL推理
        
        Args:
            ice_idx_list: 每个测试样本对应的ICD索引列表
                ice_idx_list[i] = [idx1, idx2, ...] 表示第i个测试样本的ICD索引
        
        Returns:
            结果字典
        """
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filepath:
            os.makedirs(output_json_filepath, exist_ok=True)
        
        if output_json_filename is None:
            output_json_filename = self.output_json_filename
        
        # 初始化结果存储
        results = {
            "predictions": [],
            "outputs": [],
            "prompts": [],
            "ice_idx": ice_idx_list,
        }
        
        # 添加其他字段
        for field in self.other_save_field:
            if field in self.test_ds.column_names:
                results[field] = []
        
        logger.info(f"Starting LLaDA ICL inference for {len(self.test_ds)} samples...")
        
        # 逐个处理测试样本
        for idx, test_sample in enumerate(tqdm(self.test_ds, desc="Inference", ncols=100)):
            try:
                # 构建prompt
                prompt = self._build_prompt(test_sample, ice_idx_list[idx])
                
                # 生成
                output = self._generate_with_llada(prompt)
                
                # 解码结果
                # 找到mask位置
                mask_positions = (prompt == self.mask_id).nonzero(as_tuple=True)
                first_mask_pos = mask_positions[1][0].item()
                last_mask_pos = mask_positions[1][-1].item()
                
                # 提取生成的答案部分
                generated = self.tokenizer.batch_decode(
                    output[:, first_mask_pos:last_mask_pos+1],
                    skip_special_tokens=True,
                )[0]
                
                # 完整输出
                complete_output = self.tokenizer.batch_decode(
                    output[0],
                    skip_special_tokens=False,
                )
                
                # 原始prompt
                origin_prompt = self.tokenizer.batch_decode(
                    prompt[0],
                    skip_special_tokens=True,
                )
                
                # 保存结果
                results["predictions"].append(generated)
                results["outputs"].append(complete_output)
                results["prompts"].append(origin_prompt)
                
                # 保存其他字段
                for field in self.other_save_field:
                    if field in test_sample:
                        if field not in results:
                            results[field] = []
                        results[field].append(test_sample[field])
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                # 添加空结果以保持索引一致
                results["predictions"].append("")
                results["outputs"].append("")
                results["prompts"].append("")
                for field in self.other_save_field:
                    if field in results:
                        results[field].append(None)
        
        # 保存结果到JSON
        if output_json_filepath:
            import json
            output_path = os.path.join(
                output_json_filepath,
                f"{output_json_filename}.json"
            )
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        return results
    
    #这里先提前把inference接口做进这里,之后再找别的方法
    @torch.inference_mode()
    def gen_inference(
        self,
        ice_idx_list: List[List[int]],
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        生成推理（与inference相同，保持接口一致性）
        """
        return self.inference(ice_idx_list, output_json_filepath, output_json_filename)
    
    @torch.inference_mode()
    def ppl_inference(
        self,
        ice_idx_list: List[List[int]],
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        困惑度推理（用于分类任务）
        
        注意：LLaDA是生成模型，困惑度推理需要特殊处理
        这里提供一个基础实现，可能需要根据具体任务调整
        """
        logger.warning("PPL inference for LLaDA is not fully implemented yet.")
        logger.info("Falling back to generation inference...")
        return self.inference(ice_idx_list, output_json_filepath, output_json_filename)


class LLaDAICLInferencer(DLLMICLInferencer):
    """
    兼容名：
    - 原 open_mmicl 中使用的是 LLaDAICLInferencer
    - 现在统一封装为 DLLMICLInferencer，这个类只是一个别名，方便平滑迁移
    """
    pass
