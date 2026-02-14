#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：使用生成的ICD序列进行推理并计算准确率

根据参数搜索对应的JSON文件，使用JSON中的ICD序列进行推理，计算准确率。

使用方法:
    python test_icd_sequences.py --task gsm8k --model llada --sampler random [其他参数...]
"""

import os
import sys
import json
import re
import argparse
import glob
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# 添加项目路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoTokenizer
from model.model_llada import LLaDAModelLM
from utils.eval_utils import gsm8k_check, eval_gsm8k
from lever_lm.load_ds_utils import load_gsm8k_ds, load_mmlu_ds
from open_mmicl.prompt_template import PromptTemplate
from open_mmicl.metrics import GSM8KMetrics, MMLUMetrics
import hydra
from omegaconf import DictConfig


def find_json_by_params(
    search_dir: str,
    task: str,
    model: str,
    sampler: str,
    scorer: str,
    construct_order: str,
    beam_size: int,
    few_shot: int,
    candidate_num: int,
    sample_num: int,
    mc_num: Optional[int] = None,
    coarse_k: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
) -> Optional[str]:
    """
    根据参数搜索对应的JSON文件
    
    Args:
        search_dir: 搜索目录（通常是sub_proc_data或generated_data）
        task: 任务名称（如gsm8k）
        model: 模型名称（如llada）
        sampler: 采样器名称（如random_sampler或random）
        scorer: 评分函数（如infoscore）
        construct_order: 构造顺序（如no_order）
        beam_size: beam大小
        few_shot: few-shot数量
        candidate_num: 候选数量
        sample_num: 样本数量
        
    Returns:
        JSON文件路径，如果找不到返回None
    """
    # 构建搜索模式
    # 文件名格式（新版）:
    #   task-task-model-sampler-scorer:xxx-construct_order:xxx-beam_size:x-few_shot:x-candidate_num:x-sample_num:x-mc_num:y.json
    # - 如果指定 mc_num：精确匹配该 mc_num
    # - 如果未指定：在 sample_num 后面使用通配符 *.json，兼容旧版（无 mc_num 字段）
    # 注意：实际文件名使用的是 sampler_name（如 text_sim_qwen_mmr），可能没有 _sampler 后缀
    
    # 先尝试使用原始 sampler 名称（因为 generate_data_main.py 使用的是 cfg.sampler.sampler_name）
    sampler_patterns = [sampler]
    # 如果原始名称中没有 _sampler，也尝试添加后缀（兼容旧格式）
    if "_sampler" not in sampler:
        sampler_patterns.append(f"{sampler}_sampler")
    
    # 尝试每个 sampler_pattern
    for sampler_pattern in sampler_patterns:
        pattern = (
            f"{task}-{task}-{model}-{sampler_pattern}-scorer:{scorer}-"
            f"construct_order:{construct_order}-"
            f"beam_size:{beam_size}-few_shot:{few_shot}-"
            f"candidate_num:{candidate_num}-sample_num:{sample_num}"
        )
        
        # 添加 mc_num（如果指定）
        if mc_num is not None:
            pattern = f"{pattern}-mc_num:{mc_num}"
        
        # 添加 coarse_k 和 lambda（如果指定，用于 MMLU）
        if coarse_k is not None and mmr_lambda is not None:
            pattern = f"{pattern}-coarse_k:{coarse_k}-lambda:{mmr_lambda}"
        
        # 构建完整搜索路径
        # 如果 coarse_k 和 lambda 未指定，使用通配符匹配（兼容包含这些参数的文件名）
        if coarse_k is None or mmr_lambda is None:
            # 允许 sample_num 后面追加任意后缀（例如 -mc_num:128-coarse_k:200-lambda:0.1）
            pattern = f"{pattern}*.json"
        else:
            pattern = f"{pattern}.json"
        
        # 在当前搜索目录中查找
        search_path = os.path.join(search_dir, pattern)
        matches = glob.glob(search_path)
        if matches:
            return matches[0]
        
        # 如果找不到，尝试在generated_data目录中搜索（如果当前在sub_proc_data中）
        if "sub_proc_data" in search_dir:
            parent_dir = os.path.dirname(search_dir)  # 获取 .../generated_data
            generated_data_dir = parent_dir
            if os.path.exists(generated_data_dir):
                search_path = os.path.join(generated_data_dir, pattern)
                matches = glob.glob(search_path)
                if matches:
                    return matches[0]
        
        # 如果还是找不到，尝试在generated_data目录中搜索（如果当前不在sub_proc_data中）
        if "sub_proc_data" not in search_dir and "generated_data" in search_dir:
            # 已经在generated_data中，不需要再搜索
            pass
        elif "generated_data" not in search_dir:
            # 如果不在generated_data中，尝试添加generated_data路径
            if os.path.exists(os.path.join(search_dir, "generated_data")):
                generated_data_dir = os.path.join(search_dir, "generated_data")
                search_path = os.path.join(generated_data_dir, pattern)
                matches = glob.glob(search_path)
                if matches:
                    return matches[0]
    
    return None


def load_icd_data(json_file: str) -> Dict:
    """
    加载ICD序列数据
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        ICD数据字典 {anchor_id: {id_list: [...], score_list: [...]}}
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_best_icd_sequence_with_query_position(
    icd_data: Dict,
    anchor_id: str,
    icd_rank: int = 0,
) -> Tuple[List[int], int]:
    """
    获取第 icd_rank 名的 ICD 序列（按 score 从高到低排序）和 query 在序列中的位置
    
    Args:
        icd_data: ICD数据
        anchor_id: anchor ID
        
    Returns:
        (icd_sequence, query_position)
        icd_sequence: 完整序列（包含anchor）
        query_position: anchor在序列中的位置（0-based）
    """
    if anchor_id not in icd_data:
        raise ValueError(f"Anchor {anchor_id} not found in ICD data")
    
    anchor_data = icd_data[anchor_id]
    id_list = anchor_data['id_list']
    score_list = anchor_data['score_list']

    if icd_rank < 0:
        raise ValueError(f"icd_rank must be >= 0, got {icd_rank}")
    if len(id_list) == 0:
        raise ValueError(f"Anchor {anchor_id} has empty id_list")
    if len(score_list) != len(id_list):
        raise ValueError(
            f"Anchor {anchor_id} has mismatched lengths: "
            f"len(id_list)={len(id_list)} vs len(score_list)={len(score_list)}"
        )

    # 按 score 从高到低排序，选第 icd_rank 名（0 表示最高分）
    ranked = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)
    if icd_rank >= len(ranked):
        raise ValueError(
            f"icd_rank={icd_rank} out of range for anchor {anchor_id}: "
            f"only {len(ranked)} candidates"
        )
    chosen_idx = ranked[icd_rank]
    best_sequence = id_list[chosen_idx]
    
    # 找到anchor在序列中的位置
    anchor_id_int = int(anchor_id)
    if anchor_id_int not in best_sequence:
        raise ValueError(f"Anchor {anchor_id} not found in best sequence {best_sequence}")
    
    query_position = best_sequence.index(anchor_id_int)
    
    return best_sequence, query_position


def build_prompt_from_template(
    icd_samples: List[Dict],
    query_sample: Dict,
    query_position: int,
    prompt_template: Optional[PromptTemplate] = None,
    split_token: str = "\n\n",
) -> str:
    """
    使用 PromptTemplate 构建 prompt（支持 GSM8K 和 MMLU）
    
    Args:
        icd_samples: ICD样本列表（训练样本）
        query_sample: query样本（anchor样本）
        query_position: query在序列中的位置（0表示最前，nshot表示最后）
        prompt_template: PromptTemplate 实例（必须提供）
        split_token: 分隔符（默认 "\n\n"）
        
    Returns:
        构建好的prompt字符串
    """
    if prompt_template is None:
        raise ValueError("prompt_template must be provided. Please ensure task config has template and column_token_map.")
    
    # 使用 PromptTemplate 构建 prompt
    # 1. 生成 ICD prompts
    icd_prompts = []
    for sample in icd_samples:
        icd_prompt = prompt_template.generate_ice_item(sample)
        icd_prompts.append(icd_prompt)
    
    # 2. 生成 query prompt（使用 mask）
    query_prompt = prompt_template.generate_query_item(query_sample, use_mask=True)
    
    # 3. 根据 query_position 组合
    if query_position <= 0:
        # query 在最前
        all_prompts = [query_prompt] + icd_prompts
    elif query_position >= len(icd_prompts):
        # query 在最后
        all_prompts = icd_prompts + [query_prompt]
    else:
        # query 在中间
        insert_pos = query_position
        all_prompts = icd_prompts[:insert_pos] + [query_prompt] + icd_prompts[insert_pos:]
    
    # 4. 组合所有 prompts
    combined_prompt = split_token.join(all_prompts)
    return combined_prompt


def test_icd_sequences(
    # 搜索JSON的参数
    task: str = "gsm8k",
    model: str = "llada",
    sampler: str = "random",
    scorer: str = "infoscore",
    construct_order: str = "no_order",
    beam_size: int = 3,
    few_shot: int = 4,
    candidate_num: int = 10,
    sample_num: int = 10,
    mc_num: Optional[int] = None,
    coarse_k: Optional[int] = None,  # MMLU 参数
    mmr_lambda: Optional[float] = None,  # MMLU 参数
    
    # 评测参数（可从config读取默认值）
    config_path: str = "./configs",
    config_name: str = "generate_data.yaml",
    model_path: Optional[str] = None,
    device: str = "cuda:0",
    mask_length: Optional[int] = None,
    mask_id: Optional[int] = None,
    block_length: Optional[int] = None,
    gen_length: Optional[int] = None,
    steps: Optional[int] = None,
    temperature: Optional[float] = None,
    mode: str = "original",
    icd_rank: int = 0,
) -> Dict:
    """
    测试ICD序列的准确率
    
    Args:
        task: 任务名称
        model: 模型名称
        sampler: 采样器名称
        scorer: 评分函数
        construct_order: 构造顺序
        beam_size: beam大小
        few_shot: few-shot数量
        candidate_num: 候选数量
        sample_num: 样本数量
        mc_num: 生成时使用的 Monte Carlo 采样次数（用于精确匹配特定结果文件，可选）
        config_path: 配置文件路径
        config_name: 配置文件名称
        model_path: 模型路径（如果为None，从config读取）
        device: 设备
        mask_length: mask长度（如果为None，从config读取）
        mask_id: mask token ID（如果为None，从config读取）
        block_length: 块长度（如果为None，从config读取）
        gen_length: 生成长度（如果为None，从config读取）
        steps: 采样步数（如果为None，从config读取）
        temperature: 温度（如果为None，从config读取）
        mode: 生成模式
        
    Returns:
        测试结果字典
    """
    print("="*80)
    print("ICD序列测试脚本")
    print("="*80)
    
    # 1. 加载配置
    print(f"\n⚙️  加载配置...")
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)
    
    # 从config读取默认值
    if model_path is None:
        model_path = cfg.infer_model.get("model_path")
        if model_path is None:
            raise ValueError("model_path must be provided via --model_path or in configs/infer_model/llada.yaml")
    
    if mask_id is None:
        mask_id = cfg.infer_model.get("mask_id")
        if mask_id is None:
            raise ValueError("mask_id must be provided via --mask_id or in configs/infer_model/llada.yaml")
    
    # 从 task.gen_args 读取参数（如果命令行未指定）
    task_gen_args = cfg.task.get("gen_args", None)
    if task_gen_args is None:
        raise ValueError(f"task.gen_args not found in configs/task/{cfg.task.task_name}.yaml")
    
    # mask_length: 命令行 > task.gen_args
    if mask_length is None:
        if "mask_length" not in task_gen_args:
            raise ValueError(f"mask_length not found in task.gen_args. Please set it in configs/task/{cfg.task.task_name}.yaml or via --mask_length")
        mask_length = int(task_gen_args.mask_length)
    
    # block_length: 命令行 > task.gen_args
    if block_length is None:
        if "block_length" not in task_gen_args:
            raise ValueError(f"block_length not found in task.gen_args. Please set it in configs/task/{cfg.task.task_name}.yaml or via --block_length")
        block_length = int(task_gen_args.block_length)
    
    # gen_length: 命令行 > task.gen_args
    if gen_length is None:
        if "gen_length" not in task_gen_args:
            raise ValueError(f"gen_length not found in task.gen_args. Please set it in configs/task/{cfg.task.task_name}.yaml or via --gen_length")
        gen_length = int(task_gen_args.gen_length)
    
    # steps: 命令行 > task.gen_args
    if steps is None:
        if "steps" not in task_gen_args:
            raise ValueError(f"steps not found in task.gen_args. Please set it in configs/task/{cfg.task.task_name}.yaml or via --steps")
        steps = int(task_gen_args.steps)
    
    # temperature: 命令行 > task.gen_args (可选，有默认值)
    if temperature is None:
        if "temperature" in task_gen_args:
            temperature = float(task_gen_args.temperature)
        else:
            temperature = 0.0  # 默认值
    
    print(f"   模型路径: {model_path}")
    print(f"   mask_length: {mask_length}")
    print(f"   mask_id: {mask_id}")
    # 根据任务类型显示模板格式
    if task == "gsm8k":
        print(f"   使用格式: question: <Q>\\n<answer>\\n<A>\\n</answer> (GSM8K格式)")
    elif task == "mmlu":
        template_str = cfg.task.get("template", "")
        print(f"   使用格式: {template_str} (MMLU格式，使用 PromptTemplate)")
    else:
        print(f"   使用格式: 根据 task.column_token_map 动态生成")
    print(f"   ICD Rank: {icd_rank} (0表示最高分，2表示第三名)")
    
    # 2. 搜索JSON文件
    print(f"\n📂 搜索JSON文件...")
    # 优先在generated_data目录中搜索
    base_dir = os.path.join(cfg.get("output_dir", "./generated_icd_data"), "generated_data")
    search_dir = base_dir  # 首先在generated_data中搜索
    
    json_file = find_json_by_params(
        search_dir=search_dir,
        task=task,
        model=model,
        sampler=sampler,
        scorer=scorer,
        construct_order=construct_order,
        beam_size=beam_size,
        few_shot=few_shot,
        candidate_num=candidate_num,
        sample_num=sample_num,
        mc_num=mc_num,
        coarse_k=coarse_k,
        mmr_lambda=mmr_lambda,
    )
    
    if json_file is None:
        raise FileNotFoundError(
            f"找不到匹配的JSON文件。搜索目录: {search_dir}\n"
            f"参数: task={task}, model={model}, sampler={sampler}, "
            f"scorer={scorer}, construct_order={construct_order}, "
            f"beam_size={beam_size}, few_shot={few_shot}, "
            f"candidate_num={candidate_num}, sample_num={sample_num}"
        )
    
    print(f"   ✅ 找到JSON文件: {json_file}")
    
    # 3. 加载ICD数据
    print(f"\n📋 加载ICD数据...")
    icd_data = load_icd_data(json_file)
    anchor_ids = list(icd_data.keys())
    print(f"   找到 {len(anchor_ids)} 个anchor样本")
    
    # 4. 加载数据集
    print(f"\n📊 加载数据集...")
    if task == "gsm8k":
        train_ds = load_gsm8k_ds(
            version=cfg.dataset.version,
            data_path=cfg.dataset.train_path,
            split="train"
        )
    elif task == "mmlu":
        train_ds = load_mmlu_ds(
            version=cfg.dataset.version,
            data_path=cfg.dataset.train_path,
            split="train"
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    print(f"   训练集大小: {len(train_ds)}")
    
    # 5. 加载模型
    print(f"\n🤖 加载模型...")
    print(f"   模型路径: {model_path}")
    print(f"   设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 使用本地 LLaDAModelLM（含 all_tied_weights_keys 兼容），与 generate_data_main 一致，避免 transformers 版本差异报错
    model = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    print("   ✅ 模型加载完成")
    
    # 5.5. 初始化 PromptTemplate（如果任务需要）
    prompt_template_obj = None
    if task in ["gsm8k", "mmlu"]:
        # 从配置中读取 prompt 相关参数
        prompt_template_str = cfg.task.get("template", None)
        column_token_map = cfg.task.get("column_token_map", None)
        if column_token_map is not None:
            column_token_map = dict(column_token_map)
        mask_column_token_map = cfg.task.get("mask_column_token_map", None)
        if isinstance(mask_column_token_map, dict):
            mask_column_token_map = dict(mask_column_token_map)
        split_token = cfg.task.get("split_token", "\n\n")
        
        if prompt_template_str and column_token_map:
            prompt_template_obj = PromptTemplate(
                prompt_template=prompt_template_str,
                mask_token_str="<|mdm_mask|>",
                mask_length=mask_length,
                column_token_map=column_token_map,
                mask_column_token_map=mask_column_token_map,
            )
            print(f"   ✅ PromptTemplate 初始化完成（使用模板和 column_token_map）")
        else:
            print(f"   ⚠️  Warning: 未提供 template 或 column_token_map，将使用旧逻辑")
    
    # 6. 准备测试样本
    print(f"\n🎯 准备测试样本...")
    test_samples = []
    for anchor_id in anchor_ids:
        anchor_idx = int(anchor_id)
        if anchor_idx < len(train_ds):
            anchor_sample = train_ds[anchor_idx].copy()  # 复制以避免修改原始数据
            
            # 提取 answer（用于评估）
            if task == "gsm8k":
                # GSM8K: answer 字段包含完整的解答过程（包括####格式）
                answer = anchor_sample.get('answer', '')
            elif task == "mmlu":
                # MMLU: answer 字段已经是 "A"/"B"/"C"/"D"
                answer = anchor_sample.get('answer', '')
            else:
                answer = anchor_sample.get('answer', '')
            
            test_samples.append({
                'idx': anchor_idx,
                'answer': answer,  # 用于评估的答案
                'anchor_id': anchor_id,
                'sample': anchor_sample  # 保存完整样本用于构建prompt
            })
        else:
            print(f"   ⚠️  Warning: anchor_id {anchor_id} 超出训练集范围")
    
    print(f"   准备测试 {len(test_samples)} 个样本")
    
    # 7. 对每个anchor进行推理
    print(f"\n🚀 开始推理...")
    results = []
    split_token = cfg.task.get("split_token", "\n\n") if task in ["gsm8k", "mmlu"] else "\n\n"
    
    for test_sample in tqdm(test_samples, desc="推理进度"):
        anchor_id = test_sample['anchor_id']
        
        try:
            # 获取最佳ICD序列和query位置
            icd_sequence, query_pos_in_sequence = get_best_icd_sequence_with_query_position(
                icd_data, anchor_id, icd_rank=icd_rank
            )
            
            # 加载ICD样本（排除anchor本身）
            anchor_id_int = int(anchor_id)
            icd_indices = [idx for idx in icd_sequence if idx != anchor_id_int]
            icd_samples = [train_ds[idx] for idx in icd_indices if idx < len(train_ds)]
            
            # 计算query在few-shot中的位置
            # query_pos_in_sequence是anchor在完整序列中的位置
            # 我们需要计算它在few-shot示例中的位置（排除anchor后）
            query_position = sum(1 for idx in icd_sequence[:query_pos_in_sequence] if idx != anchor_id_int)
            
            # 构建prompt（使用 PromptTemplate 或旧逻辑）
            prompt_text = build_prompt_from_template(
                icd_samples=icd_samples,
                query_sample=test_sample['sample'],
                query_position=query_position,
                prompt_template=prompt_template_obj,
                split_token=split_token,
            )
            
            # 打印prompt信息
            print(f"\n{'='*80}")
            print(f"Anchor ID: {anchor_id}")
            print(f"Query Position: {query_position}")
            print(f"ICD Rank: {icd_rank}")
            print(f"ICD Sequence: {icd_sequence}")
            print(f"ICD Indices (excluding anchor): {icd_indices}")
            print(f"{'='*80}")
            print("PROMPT:")
            print(f"{'='*80}")
            print(prompt_text)
            print(f"{'='*80}\n")
            
            # Tokenize prompt
            prompt_tokens = tokenizer(prompt_text, return_tensors='pt')['input_ids'].to(device)
            
            # 找到mask token位置
            mask_positions = (prompt_tokens == mask_id).nonzero(as_tuple=True)
            if len(mask_positions[0]) == 0:
                raise ValueError("No mask tokens found in prompt")
            
            first_mask_pos = mask_positions[1][0].item()
            last_mask_pos = mask_positions[1][-1].item()
            
            # 调用src/generate.py中的generate函数
            from src.generate import generate as generate_core
            generated_tokens = generate_core(
                model=model,
                prompt=prompt_tokens,
                gen_start=first_mask_pos,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=0.0,
                remasking='low_confidence',
                mask_id=mask_id
            )
            
            # 解码生成的文本（只解码mask部分）
            generated_text = tokenizer.batch_decode(
                generated_tokens[:, first_mask_pos:last_mask_pos+1],
                skip_special_tokens=False
            )[0]
            answer = generated_text
            
            print(f"Generated Answer: {answer}\n")
            results.append(answer)
            
        except Exception as e:
            print(f"\n   ❌ 处理anchor {anchor_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results.append("")  # 添加空答案
    
    # 8. 评估准确率
    print(f"\n📈 计算准确率...")
    
    if task == "gsm8k":
        # GSM8K: 使用 eval_gsm8k
        eval_dataset = [{'answer': s['answer']} for s in test_samples]
        
        class Args:
            def __init__(self):
                self.task = task
                self.model_name = model_path
                self.device = device
                self.gen_length = gen_length
                self.steps = steps
                self.block_length = block_length
                self.temperature = temperature
                self.mode = mode
                self.nshot = few_shot
                self.query_position = 0
                self.icd_rank = icd_rank
                self.sampler = sampler
                self.scorer = scorer
                self.construct_order = construct_order
                self.beam_size = beam_size
                self.candidate_num = candidate_num
                self.sample_num = sample_num
        
        args = Args()
        result_path = os.path.join(os.path.dirname(json_file), "test_results")
        
        accuracy = eval_gsm8k(
            results=results,
            dataset=eval_dataset,
            result_path=result_path,
            args=args,
            position=0,
            iswrite=True
        )
    elif task == "mmlu":
        # MMLU: 使用 MMLUMetrics
        metrics = MMLUMetrics()
        ground_truths = [s['answer'] for s in test_samples]
        
        # 批量评估
        batch_result = metrics.evaluate_batch(results, ground_truths)
        accuracy = batch_result['accuracy']
        
        result_path = os.path.join(os.path.dirname(json_file), "test_results")
        os.makedirs(result_path, exist_ok=True)
        
        # 保存结果
        result_file = os.path.join(result_path, f"{task}_test_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': accuracy,
                'correct_count': batch_result['correct_count'],
                'total_count': batch_result['total_count'],
                'results': batch_result['results'],
                'generated_texts': results,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"   结果保存到: {result_file}")
    else:
        raise ValueError(f"Unsupported task for evaluation: {task}")
    
    # 9. 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    print(f"JSON文件: {json_file}")
    print(f"任务: {task}")
    print(f"测试样本数: {len(test_samples)}")
    print(f"准确率: {accuracy:.4f}")
    print(f"结果保存到: {result_path}")
    print("="*80)
    
    return {
        'json_file': json_file,
        'task': task,
        'num_samples': len(test_samples),
        'accuracy': accuracy,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="测试ICD序列的准确率")
    
    # 搜索JSON的参数
    parser.add_argument('--task', type=str, default='gsm8k', help='任务名称')
    parser.add_argument('--model', type=str, default='llada', help='模型名称')
    parser.add_argument('--sampler', type=str, default='random', help='采样器名称')
    parser.add_argument('--scorer', type=str, default='infoscore', help='评分函数')
    parser.add_argument('--construct_order', type=str, default='no_order', help='构造顺序')
    parser.add_argument('--beam_size', type=int, default=3, help='beam大小')
    parser.add_argument('--few_shot', type=int, default=4, help='few-shot数量')
    parser.add_argument('--candidate_num', type=int, default=10, help='候选数量')
    parser.add_argument('--sample_num', type=int, default=10, help='样本数量')
    parser.add_argument('--icd_rank', type=int, default=0, help='选择第几名score的ICD序列（0表示最高分）')
    parser.add_argument('--mc_num', type=int, default=None, help='生成时使用的 Monte Carlo 采样次数（用于精确匹配特定结果文件，可选）')
    parser.add_argument('--coarse_k', type=int, default=None, help='MMLU 粗筛数量（用于精确匹配特定结果文件，可选）')
    parser.add_argument('--mmr_lambda', type=float, default=None, help='MMLU MMR lambda 参数（用于精确匹配特定结果文件，可选）')
    
    # 评测参数（可选，会从config读取默认值）
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--mask_length', type=int, default=None, help='mask长度')
    parser.add_argument('--mask_id', type=int, default=None, help='mask token ID')
    parser.add_argument('--block_length', type=int, default=None, help='块长度')
    parser.add_argument('--gen_length', type=int, default=None, help='生成长度')
    parser.add_argument('--steps', type=int, default=None, help='采样步数')
    parser.add_argument('--temperature', type=float, default=None, help='温度')
    parser.add_argument('--mode', type=str, default='original', help='生成模式')
    
    args = parser.parse_args()
    
    # 运行测试
    test_icd_sequences(
        task=args.task,
        model=args.model,
        sampler=args.sampler,
        scorer=args.scorer,
        construct_order=args.construct_order,
        beam_size=args.beam_size,
        few_shot=args.few_shot,
        candidate_num=args.candidate_num,
        sample_num=args.sample_num,
        mc_num=args.mc_num,
        coarse_k=args.coarse_k,
        mmr_lambda=args.mmr_lambda,
        model_path=args.model_path,
        device=args.device,
        mask_length=args.mask_length,
        mask_id=args.mask_id,
        block_length=args.block_length,
        gen_length=args.gen_length,
        steps=args.steps,
        temperature=args.temperature,
        mode=args.mode,
        icd_rank=args.icd_rank
    )


if __name__ == "__main__":
    main()
