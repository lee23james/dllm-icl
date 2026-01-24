"""
阶段2：ICL推理主脚本
使用 retriever 选择 ICD，然后进行推理和评估
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import hydra
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm
from datasets import Dataset

# 添加项目路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from open_mmicl.retriever import RandRetriever
from open_mmicl.interface import LLaDAInterface
from open_mmicl.metrics import GSM8KMetrics
from open_mmicl.icl_interface import DLLMICLInferencer
from lever_lm.load_ds_utils import load_gsm8k_ds
#这里是第二阶段,需要进行推理(其实也需要batch和分类)
#这里我并没有进行详细的检查

@hydra.main(version_base=None, config_path="configs", config_name="icl_inference")
def main(cfg: DictConfig):
    """
    主函数：ICL推理
    """
    logger.info("="*80)
    logger.info("Stage 2: ICL Inference")
    logger.info("="*80)
    
    # 1. 加载数据集
    logger.info("Loading datasets...")
    if cfg.task.task_name == "gsm8k":
        # 分别加载 train 和 test 数据集
        train_ds = load_gsm8k_ds(
            version=cfg.dataset.version,
            data_path=cfg.dataset.train_path,
        )
        test_ds = load_gsm8k_ds(
            version=cfg.dataset.version,
            data_path=cfg.dataset.test_path,
        )
    else:
        raise ValueError(f"Unsupported task: {cfg.task.task_name}")
    
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Test dataset size: {len(test_ds)}")
    
    # 2. 初始化模型和tokenizer
    logger.info("Initializing model and tokenizer...")
    # TODO: 根据实际模型路径初始化
    model = None
    tokenizer = None
    
    if model is None or tokenizer is None:
        logger.error("Model and tokenizer not initialized. Please implement model loading.")
        return
    
    # 3. 初始化retriever
    logger.info("Initializing retriever...")
    retriever = RandRetriever(
        train_ds=train_ds,
        nshot=cfg.retriever.nshot,
        seed=cfg.retriever.get("seed", 42),
    )
    
    # 4. 初始化interface
    logger.info("Initializing interface...")
    interface = LLaDAInterface(
        model=model,
        tokenizer=tokenizer,
        task=cfg.task.task_name,
        mask_id=cfg.infer_model.mask_id,
        mask_length=cfg.infer_model.mask_length,
        split_token=cfg.task.get("split_token", "\n\n"),
    )
    
    # 5. 初始化评估器
    logger.info("Initializing metrics...")
    metrics = GSM8KMetrics()
    
    # 6. 执行推理
    logger.info("Starting inference...")
    results = []
    
    # 限制测试样本数量（用于初步试验）
    max_samples = cfg.get("max_samples", None)
    if max_samples:
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))
        logger.info(f"Limited to {len(test_ds)} test samples")
    
    for idx, test_sample in enumerate(tqdm(test_ds, desc="Inference", ncols=100)):
        try:
            # 检索ICD
            icd_indices = retriever.retrieve(
                test_sample=test_sample,
                exclude_indices=[test_sample.get("idx")] if "idx" in test_sample else [],
            )
            
            # 获取ICD数据
            icd_samples = [train_ds[i] for i in icd_indices]
            
            # 构建prompt
            query_position = cfg.get("query_position", 0)  # 0表示最后
            prompt = interface.build_prompt(
                ice_samples=icd_samples,
                test_sample=test_sample,
                query_position=query_position,
            )
            
            # Tokenize
            prompt_tensor = interface.tokenize_prompt(prompt)
            
            # 生成（传入生成参数）
            output_tensor = interface.generate(
                prompt_tensor,
                **cfg.infer_model.generation_kwargs
            )
            
            # 解码
            generated_text = interface.decode_output(output_tensor)
            
            # 提取答案
            predicted_answer = interface.extract_answer(generated_text)
            
            # 获取真实答案
            ground_truth = test_sample.get("answer", "")
            
            # 评估
            eval_result = metrics.evaluate_single(generated_text, ground_truth)
            
            # 保存结果
            result = {
                "test_idx": idx,
                "test_sample_idx": test_sample.get("idx", idx),
                "icd_indices": icd_indices,
                "query_position": query_position,
                "predicted": predicted_answer,
                "ground_truth": ground_truth,
                "is_correct": eval_result["is_correct"],
                "generated_text": generated_text,
                "prompt": prompt,
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing test sample {idx}: {e}")
            continue
    
    # 7. 计算总体指标
    logger.info("Calculating metrics...")
    all_generated = [r["generated_text"] for r in results]
    all_ground_truths = [r["ground_truth"] for r in results]
    
    batch_metrics = metrics.evaluate_batch(all_generated, all_ground_truths)
    
    logger.info(f"Accuracy: {batch_metrics['accuracy']:.4f}")
    logger.info(f"Correct: {batch_metrics['correct_count']}/{batch_metrics['total_count']}")
    
    # 8. 保存结果
    output_dir = Path(cfg.get("output_dir", "./icl_inference_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{cfg.task.task_name}_inference_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": batch_metrics,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()





