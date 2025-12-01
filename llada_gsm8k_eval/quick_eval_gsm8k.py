#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K快速评估脚本
使用4-shot评估小样本，测试模型性能
"""

import os
import sys
import json
import random
from turtle import position
import numpy as np
import time
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dataset.sudoku.sudoku_handler import SudokuHandler
# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有模块
from gsm8k_handler_v2 import GSM8KHandler
from llada_loader import load_model
# from llada_inference import create_llada_inference
from llada_inference import create_llada_inference
from prompt_constructor_gsm8k import GSM8KPromptConstructor
from utils import extract_gsm8k_answer,extract_sudoku_answer


class QuickGSM8KEvaluator:
    """GSM8K快速评估器 - 4-shot小样本评估"""
    
    def __init__(
        self,
        model_path: str = "/data/share/model_weight/llada/LLaDA-8B-Base/",
        data_dir: str = "/data/share/datasets/gsm8k/",
        device: str = "cuda:0",  # 指定单个GPU
        n_shots: int = 4,  # 4-shot评估
        mask_length: int = 256,
        sampling_steps: int = 256,
        block_length: int = 256,
        temperature: float = 0.0,
        random_seed: int = 1234,
        batch_size: int = 8,
        use_multi_gpu: bool = False,  # 禁用多GPU
        available_gpus: List[int] = [],  # 清空GPU列表
        query_position: int = 2,  # 查询问题在prompt中的位置
        output_attentions: bool = False,
        ACAR_analyse: bool = False
    ):
        """
        初始化快速评估器
        
        Args:
            model_path: LLaDA模型路径
            data_dir: GSM8K数据集目录
            device: 设备类型
            n_shots: few-shot示例数量
            mask_length: mask token长度
            sampling_steps: 采样步数
            block_length: 块长度
            temperature: 采样温度
            random_seed: 随机种子
            batch_size: 批处理大小
            use_multi_gpu: 是否使用多GPU
            available_gpus: 可用GPU列表
            query_position: 查询问题在prompt中的位置
            output_attentions: 是否开启注意力捕获
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = device
        self.n_shots = n_shots
        self.mask_length = mask_length
        self.sampling_steps = sampling_steps
        self.block_length = block_length
        self.temperature = temperature
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.use_multi_gpu = False  # 强制禁用多GPU
        self.available_gpus = []
        self.gpu_inferences = {}
        self.query_position = query_position
        self.output_attentions = output_attentions
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # 初始化组件
        self.data_handler = GSM8KHandler(data_dir)
        self.prompt_constructor = GSM8KPromptConstructor(n_shots=n_shots, random_seed=random_seed, query_position=query_position)
        self.inference = None
        
        print(f"QuickGSM8KEvaluator initialized with {n_shots}-shot evaluation")
    
    def load_model(self):
        """加载模型 - 单GPU优化版本"""
        print(f"Loading LLaDA model on {self.device}...")
        
        # 使用现有的模型加载器
        model, tokenizer = load_model(
            model_path=self.model_path,
            device=self.device,
            use_accelerate=False,
            mask_id=126336,
            max_length=4096,
            torch_dtype=torch.bfloat16
        )
        
        # 创建推理器
        self.inference = create_llada_inference(
            model_path=self.model_path,
            device=self.device,
            tokenizer=tokenizer,
            model=model,
            mask_id=126336,
            max_length=4096,
            torch_dtype="bfloat16"
        )
        
        print(f"Model loaded successfully on {self.device}")
    
    def load_data(self):
        """加载数据集"""
        print("Loading datasets...")
        
        # 加载训练集和测试集
        train_dataset = self.data_handler.get_dataset("train")
        test_dataset = self.data_handler.get_dataset("test")
        
        if train_dataset is None or test_dataset is None:
            raise RuntimeError("Failed to load datasets")
        
        print(f"Train dataset: {len(train_dataset)} examples")
        print(f"Test dataset: {len(test_dataset)} examples")
        
        return train_dataset, test_dataset
    
    def sample_few_shot_examples(self, train_dataset):
        """从训练集中随机采样few-shot示例"""
        print(f"Sampling {self.n_shots} few-shot examples from train dataset...")
        
        # 从训练集中随机采样few-shot示例
        train_indices = random.sample(range(len(train_dataset)), self.n_shots)
        train_samples = [train_dataset[i] for i in train_indices]
        
        print(f"Sampled train examples: {train_indices}")
        
        # 显示采样的示例
        for i, sample in enumerate(train_samples):
            print(f"Example {i+1}: {sample['question'][:100]}...")
        
        return train_samples
    
    def sample_test_cases(self, test_dataset, n_samples=10):
        """从测试集中随机采样测试用例"""
        print(f"Sampling {n_samples} test cases from test dataset...")
        
        # 从测试集中随机采样测试用例
        test_indices = random.sample(range(len(test_dataset)), n_samples)
        test_indices.sort()  # 排序索引
        test_samples = [test_dataset[i] for i in test_indices]
        
        print(f"Sampled test indices: {test_indices}")
        
        return test_samples
    
    def evaluate_single(self, train_samples: List[Dict], test_sample: Dict, sample_idx: int) -> Dict[str, Any]:
        """评估单个样本"""
        # 构建prompt
        prompt = self.prompt_constructor.construct_prompt(
            train_samples, 
            test_sample, 
            mask_length=self.mask_length
        )
        
        # 生成答案
        generated_text = self.inference.generate_text(
            prompt=prompt,
            answer_length=self.mask_length,
            sampling_steps=self.sampling_steps,
            block_length=self.block_length,
            temperature=self.temperature,
            stop_tokens=["Question:", "Answer:", "\nQuestion:", "\nAnswer:"],
            output_attentions=self.output_attentions, # 开启注意力捕获
            ACAR_analyse=self.ACAR_analyse,
            sample_idx=sample_idx,
            query_position=self.query_position
        )
        
        # 移除冗余后处理 - stop_tokens已在llada_inference.py中处理
        
        # 提取答案
        predicted_answer = extract_gsm8k_answer(generated_text)
        true_answer = extract_gsm8k_answer(test_sample['answer'])
        
        # 判断正确性
        is_correct = predicted_answer == true_answer
        
        # 简洁输出
        print(f"测试问题: {test_sample['question'][:100]}...")
        print(f"生成内容: {generated_text[:256]}")
        print(f"真实答案: {true_answer}")
        print(f"预测答案: {predicted_answer}")
        print(f"结果: {'✓' if is_correct else '✗'}")
        print("-" * 80)
        
        return {
            "question": test_sample['question'],
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "generated_text": generated_text,
            "is_correct": is_correct,
            "prompt": prompt
        }
    
    def save_checkpoint(self, results: List[Dict], current_index: int, output_dir: Path):
        """保存断点数据"""
        checkpoint_file = output_dir / f"checkpoint_{int(time.time())}.json"
        checkpoint_data = {
            "current_index": current_index,
            "total_processed": len(results),
            "results": results,
            "timestamp": time.time()
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        return checkpoint_file

    def load_checkpoint(self, checkpoint_file: str) -> tuple:
        """加载断点数据"""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data["current_index"], checkpoint_data["results"]
        except Exception as e:
            print(f"无法加载断点文件 {checkpoint_file}: {e}")
            return 0, []

    def evaluate_batch(self, train_samples: List[Dict], test_samples: List[Dict], test_indices: List[int], use_progress_bar: bool = False, save_interval: int = 100) -> List[Dict[str, Any]]:
        """单GPU批量评估 - 带进度条和断点保存"""
        total_samples = len(test_samples)
        print(f"Evaluating {total_samples} samples on {self.device}...")
        
        # 创建结果目录
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 检查是否有断点文件需要恢复
        start_index = 0
        results = []
        checkpoint_files = list(results_dir.glob("checkpoint_*.json"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            print(f"发现断点文件: {latest_checkpoint}")
            print("自动从最新断点继续评估...")
            start_index, results = self.load_checkpoint(latest_checkpoint)
            print(f"从第 {start_index + 1} 个样本继续评估 (已完成 {len(results)} 个样本)")
        
        # 设置进度条
        if use_progress_bar:
            progress_bar = tqdm(
                total=total_samples, 
                initial=start_index,
                desc="评估进度", 
                unit="samples",
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] 准确率: {postfix}'
            )
        
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        
        for i in range(start_index, total_samples):
            test_sample = test_samples[i]
            sample_idx = test_indices[i]  # 获取原始索引
            try:
                # 评估单个样本
                result = self.evaluate_single(train_samples, test_sample, sample_idx)
                results.append(result)
                
                # 更新正确数量
                if result['is_correct']:
                    correct_count += 1
                
                # 更新进度条或显示进度
                if use_progress_bar:
                    current_accuracy = correct_count / len(results) if results else 0
                    progress_bar.set_postfix_str(f"{current_accuracy:.3f}")
                    progress_bar.update(1)
                else:
                    # 传统的进度显示
                    if (i + 1) % 50 == 0 or i == 0:
                        current_accuracy = correct_count / len(results) if results else 0
                        print(f"Processing sample {i+1}/{total_samples}... 当前准确率: {current_accuracy:.3f}")
                
                # 定期保存断点
                if (i + 1) % save_interval == 0:
                    checkpoint_file = self.save_checkpoint(results, i, results_dir)
                    if use_progress_bar:
                        progress_bar.write(f"已保存断点: {checkpoint_file.name} (样本 {i+1}/{total_samples})")
                    else:
                        print(f"已处理 {i+1}/{total_samples}, 保存断点: {checkpoint_file.name}")
                
            except Exception as e:
                error_result = {
                    "question": test_sample['question'],
                    "true_answer": extract_gsm8k_answer(test_sample['answer']),
                    "predicted_answer": None,
                    "generated_text": "",
                    "is_correct": False,
                    "error": str(e)
                }
                results.append(error_result)
                
                if use_progress_bar:
                    progress_bar.write(f"样本 {i+1} 错误: {e}")
                    current_accuracy = correct_count / len(results) if results else 0
                    progress_bar.set_postfix_str(f"{current_accuracy:.3f}")
                    progress_bar.update(1)
                else:
                    print(f"样本 {i+1} 错误: {e}")
        
        if use_progress_bar:
            progress_bar.close()
        
        # 保存最终断点
        final_checkpoint = self.save_checkpoint(results, total_samples - 1, results_dir)
        print(f"评估完成，最终结果已保存: {final_checkpoint.name}")
        
        return results
    
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算评估指标 - 简化版本"""
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['is_correct'])
        wrong_samples = total_samples - correct_samples
        accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        return {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "wrong_samples": wrong_samples,
            "accuracy": accuracy
        }
    
    def run_evaluation(self, n_test_samples=None) -> Dict[str, Any]:
        """运行快速评估"""
        print("=" * 80)
        print("GSM8K Full Evaluation - 4-shot")
        print("=" * 80)
        
        start_time = time.time()
        
        # 加载模型
        self.load_model()
        
        # 加载数据
        train_dataset, test_dataset = self.load_data()
        
        # 采样few-shot示例
        train_samples = self.sample_few_shot_examples(train_dataset)
        
        # 采样测试用例
        if n_test_samples is None:
            # 使用全部测试集
            test_samples = list(test_dataset)
            test_indices = list(range(len(test_dataset)))  # 添加索引
            print(f"Using entire test dataset: {len(test_samples)} samples")
        else:
            test_samples = self.sample_test_cases(test_dataset, n_test_samples)
        
        print(f"\n--- 开始评估 {len(test_samples)} 个测试样本 ---")
        
        # 单GPU评估
        evaluation_start = time.time()
        # 如果是全数据集评估，使用tqdm进度条
        use_progress_bar = (n_test_samples is None)
        results = self.evaluate_batch(train_samples, test_samples, test_indices, use_progress_bar=use_progress_bar)
        evaluation_end = time.time()
        
        # 计算指标
        metrics = self.calculate_metrics(results)
        
        # 显示结果
        print(f"\n{'='*80}")
        print("评估结果")
        print(f"{'='*80}")
        print(f"总样本数: {metrics['total_samples']}")
        print(f"正确样本: {metrics['correct_samples']}")
        print(f"错误样本: {metrics['wrong_samples']}")
        print(f"准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        # 显示设备信息
        print(f"\n--- 设备信息 ---")
        print(f"使用设备: {self.device}")
        
        # 时间统计
        total_time = evaluation_end - evaluation_start
        if metrics['total_samples'] > 0:
            avg_time_per_sample = total_time / metrics['total_samples']
            print(f"\n--- 时间统计 ---")
            print(f"总评估时间: {total_time:.2f}秒")
            print(f"平均每样本: {avg_time_per_sample:.2f}秒")
            print(f"样本/小时: {3600/avg_time_per_sample:.1f}")
        else:
            print(f"\n--- 时间统计 ---")
            print(f"总评估时间: {total_time:.2f}秒")
            print("没有成功处理的样本")
        
        # 保存结果
        evaluation_result = {
            "config": {
                "model_path": self.model_path,
                "data_dir": self.data_dir,
                "n_shots": self.n_shots,
                "mask_length": self.mask_length,
                "sampling_steps": self.sampling_steps,
                "block_length": self.block_length,
                "temperature": self.temperature,
                "random_seed": self.random_seed,
                "batch_size": self.batch_size,
                "use_multi_gpu": self.use_multi_gpu,
                "available_gpus": self.available_gpus
            },
            "metrics": metrics,
            "timing": {
                "total_evaluation_time": total_time,
                "avg_time_per_sample": avg_time_per_sample if metrics['total_samples'] > 0 else 0,
                "samples_per_hour": 3600/avg_time_per_sample if metrics['total_samples'] > 0 else 0
            },
            "results": results
        }
        
        # 保存到results文件夹
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        output_file = results_dir / f"quick_eval_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        return evaluation_result

#针对的是数独
def format_sudoku_for_template(puzzle_str: str) -> str:
    """将数独字符串格式化为模板格式"""
    if len(puzzle_str) != 16:
        raise ValueError(f"Invalid puzzle length: {len(puzzle_str)}")
    
    # 将16位字符串转换为4x4网格
    grid = []
    for i in range(0, 16, 4):
        row = puzzle_str[i:i+4]
        grid.append(row)
    
    # 格式化为模板样式
    formatted_grid = '\n'.join(grid)
    return formatted_grid

#既然已经是评测环节了，能不能直接把前256个作为测试样本，然后从256个后面的样本中随机取5个作为5-shot即可
# def create_sudoku_samples_from_csv(n_samples:int =4)->list:
#     """从csv数据中创建数独样本"""
#     sudoku_handler = SudokuHandler()
#     raw_data = sudoku_handler.load_raw_dataset()

#     #随机挑选样本
#     selected_indices = random.sample(range(len(raw_data)), n_samples)
#     samples = []

#     for idx in selected_indices:
#         item = raw_data[idx]
#         puzzle_formatted = format_sudoku_for_template(item['question'])
#         samples.append({
#             "question":f"Puzzle:\n{puzzle_formatted}",
#             "answer":f"\n{item['answer']}\n</answer>"
#         })
    
#     return samples

def run_sudoku_evaluation(
    model_path="/home/share/model_weight/llada/LLaDA-8B-Base",
    device="cuda:0",
    n_shots=5,
    n_test_samples=10,
    mask_length=256,
    sampling_steps=256,
    block_length=256,
    temperature=0.0,
    random_seed=1234,
    query_position=2
):
    import random, numpy as np, time
    from llada_loader import load_model
    from llada_inference import create_llada_inference
    from prompt_constructor_gsm8k import GSM8KPromptConstructor

    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print("=" * 80)
    print("Sudoku Evaluation - few-shot using CSV")
    print("=" * 80)

    #加载模型和推理器
    print(f"Loading LLaDA model on {device}...")
    model, tokenizer = load_model(
        model_path=model_path,
        device=device,
        use_accelerate=False,
        mask_id=126336,
        max_length=4096,
        torch_dtype=torch.bfloat16
    )
    inference = create_llada_inference(
        model_path=model_path,
        device=device,
        tokenizer=tokenizer,
        model=model,
        mask_id=126336,
        max_length=4096,
        torch_dtype="bfloat16"
    )
    import os
    handler = SudokuHandler(csv_path=os.path.join(os.path.dirname(__file__), "dataset", "sudoku", "4x4_sudoku_unique_solution.csv"))
    #train_samples = create_sudoku_samples_from_csv(n_samples=n_shots)

    raw = handler.load_raw_dataset()
    # 简化为：从前256个样本中随机选择n_test_samples个作为测试集
    front_256 = raw[:256]
    test_idxs = random.sample(range(256), n_test_samples)
    test_samples = []
    for idx in test_idxs:
        item = front_256[idx]
        # 格式化数独谜题并添加到测试样本
        grid = format_sudoku_for_template(item['question'])
        test_samples.append({
            "question": f"Puzzle:\n{grid}",
            "answer": f"\n{item['answer']}\n</answer>",
            "orig_idx": idx  # 记录原始索引便于追踪
        })

    #找到后256个样本作为训练集
    tail=raw[256:] if len(raw)>256 else []
    if len(tail) ==0:
        pool = raw
    else:
        pool = tail
    train_idxs=random.sample(range(len(pool)), n_shots)
    train_samples=[]
    for i in train_idxs:
        item = pool[i]
        grid = format_sudoku_for_template(item['question'])
        train_samples.append({
            "question": f"Puzzle:\n{grid}",
            "answer": f"\n{item['answer']}\n</answer>"    # few-shot 里直接放真实解
        })


    #使用现有构造器来拼接few-shot+query
    constructor = GSM8KPromptConstructor(n_shots=n_shots, query_position=query_position)
    #逐样本推理
    correct =0
    results =[]
    t0=time.time()
    for i, test in enumerate(test_samples):
        prompt = constructor.construct_prompt(train_samples, test, mask_length=mask_length)
        generated = inference.generate_text(
            prompt=prompt,
            answer_length=mask_length,
            sampling_steps=sampling_steps,
            block_length=block_length,
            temperature=temperature,
            stop_tokens=["Question:", "Answer:", "\nQuestion:", "\nAnswer:"],
            output_attentions=False
        )

        pred = extract_sudoku_answer(generated)
        is_correct = (pred == test["answer"])
        if is_correct:
            correct +=1

        # 可选打印
        print("-" * 80)
        print(f"[{i+1}/{len(test_samples)}] idx={test['orig_idx']}")
        print("Question:")
        print(test["question"])
        print("GT:", test["answer"])
        print("Pred:", pred)
        print("是否正确？",is_correct)

        results.append({
            "orig_idx": test["orig_idx"],
            "question": test["question"],
            "gt_answer": test["answer"],
            "generated": generated,
            "pred_16": pred,
            "is_correct": is_correct,
            "prompt": prompt
        })

    
    elapsed = time.time() - t0
    acc = correct / len(test_samples) if test_samples else 0.0
    avg_time = elapsed / len(test_samples) if test_samples else 0.0
    
    # 保存到 sudoku_result 文件夹
    from pathlib import Path
    results_dir = Path(f"sudoku_result_{query_position}")
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f"sudoku_eval_{int(time.time())}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "model_path": model_path,
                "device": device,
                "n_shots": len(train_samples),
                "n_test_samples": len(test_samples),
                "mask_length": mask_length,
                "sampling_steps": sampling_steps,
                "block_length": block_length,
                "temperature": temperature,
                "random_seed": random_seed,
                "query_position": query_position,
                "elapsed": elapsed,
                "acc":acc,
                "avg_time":avg_time
            },
            "metrics": {
                "accuracy": acc,
                "total_time_sec": elapsed,
                "avg_time_per_sample_sec": avg_time
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")

    print("=" * 80)
    print(f"#samples={len(test_samples)}  Acc={acc:.3f}  Total={elapsed:.1f}s  Avg/sample={avg_time:.2f}s")

    return {
        "config": {
            "model_path": model_path,
            "device": device,
            "n_shots": len(train_samples),
            "n_test_samples": len(test_samples),
            "mask_length": mask_length,
            "sampling_steps": sampling_steps,
            "block_length": block_length,
            "temperature": temperature,
            "random_seed": random_seed,
            "query_position": query_position
        },
        "metrics": {
            "accuracy": acc,
            "total_time_sec": elapsed,
            "avg_time_per_sample_sec": avg_time
        },
        "results": results
    }



def main(output_dir="results",task="gsm8k"):
    """主函数"""
    if task == "sudoku":
        # 数独评测分支：遍历位置 0..5
        all_results = []
        for qp in range(6):
            res = run_sudoku_evaluation(
                model_path="/home/share/model_weight/llada/LLaDA-8B-Base",
                device="cuda:1",
                n_shots=5,               # 训练样本数（从第256条之后随机取）
                n_test_samples=256,       # 测试样本数（从前256里随机取）
                mask_length=128,
                sampling_steps=128,
                block_length=128,
                temperature=0.0,
                random_seed=1234,
                query_position=qp
            )
            all_results.append(res)
        return all_results

    # 创建评估器 - 4-shot单GPU优化评估
    evaluator = QuickGSM8KEvaluator(
        model_path="/home/share/model_weight/llada/LLaDA-8B-Base",
        data_dir="/home/share/datasets/gsm8k/",
        device="cuda:1",  # 使用单个GPU
        n_shots=4,  # 4-shot评估
        mask_length=256,
        sampling_steps=256,
        block_length=256,
        temperature=0.0,
        random_seed=1234,
        batch_size=8,
        use_multi_gpu=False,  # 禁用多GPU
        available_gpus=[],  # 空GPU列表
        query_position=4,  # 查询位置：1表示倒数第二位（原有行为保持不变）,0表示最后一位
        output_attentions=True,
        ACAR_analyse=True
    )
    
    # 运行评估 - 全部测试集
    results = evaluator.run_evaluation(n_test_samples=None)  # None表示运行全部测试集
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GSM8K/Sudoku Evaluation")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--task",type=str,default="gsm8k",choices=["gsm8k", "sudoku"], help="Task to run")
    args = parser.parse_args()
    main(output_dir=args.output_dir, task=args.task)                           