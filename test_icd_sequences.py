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

from transformers import AutoTokenizer, AutoModel
from utils.eval_utils import gsm8k_check, eval_gsm8k
from lever_lm.load_ds_utils import load_gsm8k_ds
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
    sample_num: int
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
    # 文件名格式: task-task-model-sampler-scorer:xxx-construct_order:xxx-beam_size:x-few_shot:x-candidate_num:x-sample_num:x.json
    # 注意：sampler可能是random_sampler或random
    sampler_pattern = sampler if "_sampler" in sampler else f"{sampler}_sampler"
    
    pattern = (
        f"{task}-{task}-{model}-{sampler_pattern}-scorer:{scorer}-"
        f"construct_order:{construct_order}-"
        f"beam_size:{beam_size}-few_shot:{few_shot}-"
        f"candidate_num:{candidate_num}-sample_num:{sample_num}.json"
    )
    
    search_path = os.path.join(search_dir, pattern)
    matches = glob.glob(search_path)
    
    if matches:
        return matches[0]
    
    # 如果找不到，尝试在generated_data目录中搜索（如果当前在sub_proc_data中）
    if "sub_proc_data" in search_dir:
        # 如果search_dir是 .../generated_data/sub_proc_data，则generated_data_dir应该是 .../generated_data
        parent_dir = os.path.dirname(search_dir)  # 获取 .../generated_data
        generated_data_dir = parent_dir  # generated_data目录就是父目录
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
    mask_token_str: str
) -> str:
    """
    按照utils.py中的format_icd和format_query_prefix格式构建prompt
    
    Args:
        icd_samples: ICD样本列表（训练样本）
        query_sample: query样本（anchor样本）
        query_position: query在序列中的位置
        mask_token_str: mask token字符串（如 "<|mdm_mask|>" * mask_length）
        
    Returns:
        构建好的prompt字符串
    """
    # 构建ICD示例的prompt（按照utils.py中的format_icd格式）
    icd_prompts = []
    for sample in icd_samples:
        # 按照utils.py中的format_icd逻辑格式化
        if 'q_a' in sample:
            # 如果已经有q_a字段，直接使用
            icd_prompt = sample['q_a']
        elif 'question' in sample and 'answer' in sample:
            # 按照format_icd的格式：question: {question}\n<answer>\n{answer}\n</answer>
            question = sample['question']
            answer = sample['answer']
            icd_prompt = f"question: {question}\n<answer>\n{answer}\n</answer>"
        else:
            raise ValueError(f"Unknown ICD format: {sample.keys()}")
        
        icd_prompts.append(icd_prompt)
    
    # 构建query prompt（按照utils.py中的format_query_prefix格式，answer部分用mask替换）
    if 'question' in query_sample:
        # 按照format_query_prefix的格式：question: {question}\n<answer>\n
        query_question = query_sample['question']
        query_prompt = f"question: {query_question}\n<answer>\n{mask_token_str}"
    elif 'q_a' in query_sample:
        # 如果只有q_a，提取question部分
        q_a = query_sample['q_a']
        if '<answer>' in q_a:
            # 提取question部分，然后加上<answer>\n和mask
            question_part = q_a.split('<answer>')[0].strip()
            query_prompt = f"{question_part}\n<answer>\n{mask_token_str}"
        else:
            query_prompt = q_a
    else:
        raise ValueError(f"Unknown query format: {query_sample.keys()}")
    
    # 根据 query_position 插入 query（直接语义：从左到右计数）
    # 这里的 query_position 是 anchor 在 best_sequence 中的索引（去掉 anchor 之后，等价于
    # “前面有多少个 ICD 示例”），因此：
    #   - query_position = 0  表示 query 在最前
    #   - query_position = len(icd_prompts) 表示 query 在最后
    if query_position <= 0:
        # query 在最前
        all_prompts = [query_prompt] + icd_prompts
    elif query_position >= len(icd_prompts):
        # query 在最后
        all_prompts = icd_prompts + [query_prompt]
    else:
        # query 在中间，保持 JSON 序列的从左到右顺序
        insert_pos = query_position
        all_prompts = icd_prompts[:insert_pos] + [query_prompt] + icd_prompts[insert_pos:]
    
    # 组合prompt（参考gsm8k_prompt的前置说明）
    front_prompt = '''Please solve the new question step by step just like the following examples. For this question:
1. Break down the problem into logical steps
2. Show all intermediate calculations
3. End with a final sentence that states the result
4. Wrap the final numeric answer in <answer> tags in the format: <answer> ### X </answer>'''
    
    combined_prompt = front_prompt + "\n\n" + "\n\n".join(all_prompts)
    
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
        model_path = cfg.infer_model.get("model_path", "/home/share/model_weight/llada/LLaDA-8B-Base/")
    if mask_length is None:
        mask_length = cfg.infer_model.get("mask_length", 256)
    if mask_id is None:
        mask_id = cfg.infer_model.get("mask_id", 126336)
    if block_length is None:
        block_length = cfg.infer_model.generation_kwargs.get("block_length", 128)
    if gen_length is None:
        gen_length = cfg.infer_model.generation_kwargs.get("gen_length", 128)
    if steps is None:
        steps = cfg.infer_model.generation_kwargs.get("steps", 128)
    if temperature is None:
        temperature = cfg.infer_model.generation_kwargs.get("temperature", 0.0)
    
    print(f"   模型路径: {model_path}")
    print(f"   mask_length: {mask_length}")
    print(f"   mask_id: {mask_id}")
    print(f"   使用格式: question: <Q>\\n<answer>\\n<A>\\n</answer> (按照utils.py格式)")
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
        sample_num=sample_num
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
    train_ds = load_gsm8k_ds(
        version=cfg.dataset.version,
        data_path=cfg.dataset.train_path,
        split="train"
    )
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
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()
    print("   ✅ 模型加载完成")
    
    # 6. 准备测试样本
    print(f"\n🎯 准备测试样本...")
    test_samples = []
    for anchor_id in anchor_ids:
        anchor_idx = int(anchor_id)
        if anchor_idx < len(train_ds):
            anchor_sample = train_ds[anchor_idx]
            # 提取question和answer
            # 注意：answer字段应该包含完整的解答过程（包括####格式），用于评估
            if 'q_a' in anchor_sample:
                q_a = anchor_sample['q_a']
                q_match = re.search(r'question:(.*?)(?:\n<answer>|\nAnswer:)', q_a, re.DOTALL)
                if q_match:
                    question = q_match.group(1).strip()
                else:
                    question = anchor_sample.get('question', '')
            else:
                question = anchor_sample.get('question', '')
            
            # 使用完整的answer字段（包含####格式），而不是从<answer>标签中提取
            # 因为gsm8k_check需要从ground_truth中提取####格式的数字
            answer = anchor_sample.get('answer', '')
            
            test_samples.append({
                'idx': anchor_idx,
                'question': question,
                'answer': answer,  # 完整的answer字段，包含####格式
                'anchor_id': anchor_id,
                'sample': anchor_sample  # 保存完整样本用于构建prompt
            })
        else:
            print(f"   ⚠️  Warning: anchor_id {anchor_id} 超出训练集范围")
    
    print(f"   准备测试 {len(test_samples)} 个样本")
    
    # 7. 对每个anchor进行推理
    print(f"\n🚀 开始推理...")
    results = []
    mask_token_str = "<|mdm_mask|>" * mask_length
    
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
            
            # 构建prompt（按照utils.py的格式）
            prompt_text = build_prompt_from_template(
                icd_samples=icd_samples,
                query_sample=test_sample['sample'],
                query_position=query_position,
                mask_token_str=mask_token_str
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
    # 准备评估用的数据集格式
    eval_dataset = [{'answer': s['answer']} for s in test_samples]
    
    # 创建args对象用于eval_gsm8k
    class Args:
        def __init__(self):
            # 基本评测配置
            self.task = task
            self.model_name = model_path
            self.device = device
            self.gen_length = gen_length
            self.steps = steps
            self.block_length = block_length
            self.temperature = temperature
            self.mode = mode
            self.nshot = few_shot
            self.query_position = 0  # 这里使用0，因为我们已经根据序列中的位置构建了prompt
            self.icd_rank = icd_rank

            # 来自 run_test_icd_sequences.sh / CLI 的 JSON 相关配置
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
