"""
GSM8K Evaluation Script for LLaDA
利用现有模块进行简化的GSM8K评估
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有模块
from gsm8k_handler_v2 import GSM8KHandler
from llada_loader import load_model
from llada_inference import create_llada_inference
from prompt_constructor_gsm8k import GSM8KPromptConstructor
from utils import extract_gsm8k_answer


class GSM8KEvaluator:
    """GSM8K评估器 - 利用现有模块"""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "/data/share/datasets/gsm8k",
        device: str = "cuda:0",  # 指定单GPU
        n_shots: int = 4,
        mask_length: int = 256,
        sampling_steps: int = 256,
        block_length: int = 256,
        temperature: float = 0.0,
        random_seed: int = 42
    ):
        """
        初始化GSM8K评估器
        
        Args:
            model_path: LLaDA模型路径
            data_dir: GSM8K数据集目录
            device: 设备类型
            n_shots: few-shot样本数
            mask_length: mask token长度
            sampling_steps: 采样步数
            block_length: 块长度
            temperature: 采样温度
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
        
        # 设置随机种子，确保结果可重现
        import random
        import numpy as np
        import torch
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # 初始化组件
        self.data_handler = GSM8KHandler(data_dir)
        self.prompt_constructor = GSM8KPromptConstructor(n_shots=n_shots, random_seed=random_seed)
        self.inference = None
        self.train_samples = None  # 缓存训练样本，避免重复加载
        
        print("GSM8KEvaluator initialized")
    
    def load_model(self):
        """加载模型"""
        print("Loading LLaDA model...")
        
        # 使用现有的模型加载器
        model, tokenizer = load_model(
            model_path=self.model_path,
            device=self.device,
            use_accelerate=False,
            mask_id=126336,
            max_length=4096,
            torch_dtype="bfloat16"
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
        
        print("Model loaded successfully")
    
    def load_train_samples(self):
        """预加载训练样本 - 避免重复加载"""
        if self.train_samples is None:
            print("Loading training samples...")
            train_dataset = self.data_handler.get_dataset("train")
            self.train_samples = train_dataset.select(range(self.n_shots))
            print(f"Loaded {len(self.train_samples)} training samples")
    
    def evaluate_single(self, test_sample: Dict, train_samples: List[Dict] = None) -> Dict[str, Any]:
        """
        评估单个样本
        
        Args:
            test_sample: 测试样本
            train_samples: 训练样本（用于few-shot）
            
        Returns:
            评估结果字典
        """
        if train_samples is None:
            # 使用缓存的训练样本
            self.load_train_samples()
            train_samples = self.train_samples
        
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
            stop_tokens=["Question:", "Answer:"]
        )
        
        # 提取答案
        predicted_answer = extract_gsm8k_answer(generated_text)
        true_answer = extract_gsm8k_answer(test_sample['answer'])
        
        # 判断正确性
        is_correct = predicted_answer == true_answer
        
        return {
            "question": test_sample['question'],
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "generated_text": generated_text,
            "is_correct": is_correct,
            "prompt": prompt
        }
    
    def evaluate_batch(self, test_samples: List[Dict], max_samples: int = None) -> List[Dict[str, Any]]:
        """
        批量评估
        
        Args:
            test_samples: 测试样本列表
            max_samples: 最大评估样本数
            
        Returns:
            评估结果列表
        """
        if max_samples is not None:
            test_samples = test_samples[:max_samples]
        
        print(f"Evaluating {len(test_samples)} samples...")
        
        # 预加载训练样本（避免重复加载）
        self.load_train_samples()
        
        results = []
        
        for i, test_sample in enumerate(test_samples):
            # 显示简化的进度信息
            if (i + 1) % 50 == 0 or i == 0:
                print(f"Processing sample {i+1}/{len(test_samples)}...")
            
            try:
                result = self.evaluate_single(test_sample, self.train_samples)
                results.append(result)
                
                # 只在错误时显示详细信息
                if not result['is_correct']:
                    print(f"  Sample {i+1} - Wrong: predicted={result['predicted_answer']}, true={result['true_answer']}")
                
            except Exception as e:
                print(f"  Sample {i+1} error: {e}")
                results.append({
                    "question": test_sample['question'],
                    "true_answer": extract_gsm8k_answer(test_sample['answer']),
                    "predicted_answer": None,
                    "generated_text": "",
                    "is_correct": False,
                    "error": str(e)
                })
        
        return results
    
    def evaluate_dataset(
        self, 
        split: str = "test", 
        max_samples: Optional[int] = None,
        save_results: bool = True,
        output_dir: str = "./results"
    ) -> Dict[str, Any]:
        """
        评估数据集
        
        Args:
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大评估样本数
            save_results: 是否保存结果
            output_dir: 结果保存目录
            
        Returns:
            评估结果字典
        """
        print(f"Evaluating GSM8K {split} dataset...")
        
        # 加载数据集
        test_dataset = self.data_handler.get_dataset(split)
        if test_dataset is None:
            raise RuntimeError(f"Failed to load {split} dataset")
        
        # 限制样本数量
        if max_samples is not None:
            test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
        
        print(f"Evaluating {len(test_dataset)} samples...")
        
        # 批量评估（记录时间）
        import time
        evaluation_start = time.time()
        results = self.evaluate_batch(test_dataset)
        evaluation_end = time.time()
        
        # 计算指标
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['is_correct'])
        accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        # 计算时间指标
        total_time = evaluation_end - evaluation_start
        avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0
        samples_per_hour = 3600 / avg_time_per_sample if avg_time_per_sample > 0 else 0
        
        # 汇总结果
        evaluation_result = {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "wrong_samples": total_samples - correct_samples,
            "accuracy": accuracy,
            "timing": {
                "total_evaluation_time": total_time,
                "avg_time_per_sample": avg_time_per_sample,
                "samples_per_hour": samples_per_hour
            },
            "results": results,
            "config": {
                "model_path": self.model_path,
                "data_dir": self.data_dir,
                "device": self.device,
                "n_shots": self.n_shots,
                "mask_length": self.mask_length,
                "sampling_steps": self.sampling_steps,
                "block_length": self.block_length,
                "temperature": self.temperature,
                "random_seed": self.random_seed
            }
        }
        
        print(f"\nEvaluation completed!")
        print(f"Total samples: {total_samples}")
        print(f"Correct samples: {correct_samples}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time per sample: {avg_time_per_sample:.2f}s")
        print(f"Samples per hour: {samples_per_hour:.1f}")
        
        # 保存结果
        if save_results:
            self._save_results(evaluation_result, output_dir, split)
        
        return evaluation_result
    
    def _save_results(self, results: Dict[str, Any], output_dir: str, split: str):
        """保存评估结果到results文件夹"""
        import time
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 添加时间戳避免文件冲突
        timestamp = int(time.time())
        
        # 保存详细结果
        results_file = output_path / f"eval_gsm8k_{split}_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存预测结果
        predictions_file = output_path / f"eval_gsm8k_{split}_predictions_{timestamp}.jsonl"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            for pred in results["results"]:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        
        print(f"Results saved to {output_path}")
        print(f"- {results_file.name}")
        print(f"- {predictions_file.name}")
    
    def evaluate_sample(self, question: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        评估单个样本
        
        Args:
            question: 问题字符串
            ground_truth: 真实答案（可选）
            
        Returns:
            评估结果
        """
        print(f"Evaluating sample: {question[:100]}...")
        
        # 创建测试样本
        test_sample = {"question": question, "answer": ground_truth or ""}
        
        # 评估
        result = self.evaluate_single(test_sample)
        
        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate LLaDA on GSM8K dataset")
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to LLaDA model")
    
    # 可选参数
    parser.add_argument("--data_dir", type=str, default="/data/share/datasets/gsm8k", help="Path to GSM8K dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--n_shots", type=int, default=4, help="Number of few-shot examples")
    parser.add_argument("--mask_length", type=int, default=256, help="Mask token length")
    parser.add_argument("--sampling_steps", type=int, default=256, help="Number of sampling steps")
    parser.add_argument("--block_length", type=int, default=256, help="Block length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed for reproducible results")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = GSM8KEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        n_shots=args.n_shots,
        mask_length=args.mask_length,
        sampling_steps=args.sampling_steps,
        block_length=args.block_length,
        temperature=args.temperature,
        random_seed=args.random_seed
    )
    
    # 加载模型
    evaluator.load_model()
    
    # 评估数据集
    results = evaluator.evaluate_dataset(
        split=args.split,
        max_samples=args.max_samples,
        save_results=True,
        output_dir=args.output_dir
    )
    
    # 打印最终结果
    print(f"\nFinal Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct_samples']}/{results['total_samples']}")
    print(f"Speed: {results['timing']['samples_per_hour']:.1f} samples/hour")


if __name__ == "__main__":
    main()