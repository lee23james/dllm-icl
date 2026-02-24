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
from open_mmicl.retriever.lever_lm_retriever import LeverLMRetriever
from lever_lm.utils import format_text_for_embedding
from open_mmicl.interface import LLaDAInterface
from open_mmicl.metrics import GSM8KMetrics, MMLUMetrics
from open_mmicl.icl_interface import DLLMICLInferencer
from lever_lm.load_ds_utils import load_gsm8k_ds, load_mmlu_ds
from utils import load_ds


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
            split="train",
        )
        test_ds = load_gsm8k_ds(
            version=cfg.dataset.version,
            data_path=cfg.dataset.test_path,
            split="validation",  # 注意：test数据可能存储在validation split中
        )
    elif cfg.task.task_name == "mmlu":
        # MMLU 任务：使用 load_ds 统一接口
        train_ds = load_ds(cfg, split="train")
        test_ds = load_ds(cfg, split="validation")  # MMLU 的 test 数据在 validation split
    else:
        raise ValueError(f"Unsupported task: {cfg.task.task_name}")
    
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Test dataset size: {len(test_ds)}")
    
    # 2. 初始化模型和tokenizer
    logger.info("Initializing model and tokenizer...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        # 获取模型路径
        model_path = cfg.infer_model.get("model_path", None)
        if model_path is None:
            model_path = cfg.infer_model.get("model_name", None)
        
        if model_path is None:
            raise ValueError(
                "Model path not found in config. Please set infer_model.model_path or infer_model.model_name"
            )
        
        logger.info(f"Loading model from {model_path}...")
        logger.info("Using AutoModel to load from model directory (will use modeling_llada.py from model directory)")
        
        # 准备模型参数
        model_kwargs = {
            "trust_remote_code": cfg.infer_model.get("trust_remote_code", True),
            "torch_dtype": getattr(torch, cfg.infer_model.get("torch_dtype", "bfloat16")),
            "local_files_only": cfg.infer_model.get("local_files_only", True),
        }
        
        # 使用 AutoModel 从模型目录加载，这样会使用模型目录中的 modeling_llada.py
        # 而不是项目中的 model/modeling_llada.py
        model = AutoModel.from_pretrained(
            model_path,
            **model_kwargs,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=model_kwargs["trust_remote_code"],
            local_files_only=model_kwargs["local_files_only"],
        )
        
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Please check your model_path configuration in configs/infer_model/llada.yaml")
        return
    
    # 3. 初始化 retriever
    logger.info("Initializing retriever...")
    retriever_type = cfg.retriever.get("type", "rand")
    if retriever_type == "lever_lm":
        from utils import get_lever_lm_path, init_lever_lm
        lm_cfg = cfg.get("lever_lm", {})
        ckpt_path = lm_cfg.get("ckpt_path")
        if not ckpt_path or not os.path.exists(ckpt_path):
            ckpt_dir = lm_cfg.get("ckpt_dir", "generated_icd_data/model_cpk/mmlu/debug")
            default_key = lm_cfg.get("default_cpk_key", "min_vl")
            ckpt_path = get_lever_lm_path(ckpt_dir, default_key)
            logger.info(f"LeverLM checkpoint: {ckpt_path}")
        emb_path = lm_cfg.get("embedding_path")
        if not emb_path or not os.path.exists(emb_path):
            raise ValueError(
                f"lever_lm.embedding_path 必须存在: {emb_path}\n"
                "请设置 lever_lm.embedding_path=generated_icd_data/cache/mmlu/xxx-TextFeatures.pt"
            )
        lever_lm, _ = init_lever_lm(
            cfg=cfg,
            ckpt_path=ckpt_path,
            train_ds=train_ds,
            embedding_path=emb_path,
            device=lm_cfg.get("device", "cuda:0"),
        )
        # 与训练时 TextSimQwenMMRSampler._format_text 完全一致，确保 query embedding 与 TextFeatures.pt 格式匹配
        dataset_type = cfg.dataset.get("name", cfg.task.task_name)
        text_field_name = "question"  # 与 configs/sampler/text_sim_qwen_mmr.yaml 一致
        query_text_extractor = lambda s: format_text_for_embedding(s, dataset_type, text_field_name)
        retriever = LeverLMRetriever(
            index_ds=train_ds,
            lever_lm=lever_lm,
            qwen_model_path=lm_cfg.get("qwen_model_path", "Qwen/Qwen3-Embedding-4B"),
            query_text_extractor=query_text_extractor,
            nshot=cfg.retriever.nshot,
            device=lm_cfg.get("device", "cuda:0"),
            batch_size=lm_cfg.get("batch_size", 32),
            reverse_seq=lm_cfg.get("reverse_seq", False),
        )
    else:
        retriever = RandRetriever(
            train_ds=train_ds,
            nshot=cfg.retriever.nshot,
            seed=cfg.retriever.get("seed", 42),
        )
    
    # 4. 初始化interface
    logger.info("Initializing interface...")
    # 获取prompt模板和column_token_map（从task配置中读取）
    prompt_template = cfg.task.get("template", None)
    column_token_map = cfg.task.get("column_token_map", None)
    if column_token_map is not None:
        # 将DictConfig转换为普通字典
        column_token_map = dict(column_token_map)
    
    # 获取mask_column_token_map（从task配置中读取）
    mask_column_token_map = cfg.task.get("mask_column_token_map", None)
    if mask_column_token_map is not None:
        # 将DictConfig转换为普通字典（如果是字典）或保持字符串格式
        if isinstance(mask_column_token_map, dict):
            mask_column_token_map = dict(mask_column_token_map)
        # 如果是字符串，保持原样（PromptTemplate会处理）

    # 优先使用 task.gen_args 中的 mask_length，其次回退到 infer_model.mask_length
    mask_length = cfg.infer_model.get("mask_length", 256)
    task_gen_args = cfg.task.get("gen_args", None)
    if task_gen_args is not None and "mask_length" in task_gen_args:
        try:
            mask_length = int(task_gen_args.mask_length)
            logger.info(f"Using mask_length from task.gen_args: {mask_length}")
        except Exception as e:
            logger.warning(f"Failed to read mask_length from task.gen_args, fallback to infer_model.mask_length. Error: {e}")
    
    interface = LLaDAInterface(
        model=model,
        tokenizer=tokenizer,
        task=cfg.task.task_name,
        mask_id=cfg.infer_model.mask_id,
        mask_length=mask_length,
        split_token=cfg.task.get("split_token", "\n\n"),
        prompt_template=prompt_template,
        column_token_map=column_token_map,
        mask_column_token_map=mask_column_token_map,
    )
    
    # 5. 初始化评估器
    logger.info("Initializing metrics...")
    if cfg.task.task_name == "gsm8k":
        metrics = GSM8KMetrics()
    elif cfg.task.task_name == "mmlu":
        metrics = MMLUMetrics()
    else:
        raise ValueError(f"Unsupported task for metrics: {cfg.task.task_name}")
    
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
            gen_kwargs = cfg.infer_model.get("generation_kwargs", {})
            if not gen_kwargs and cfg.task.get("gen_args"):
                gen_kwargs = dict(cfg.task.gen_args)
            output_tensor = interface.generate(prompt_tensor, **gen_kwargs)
            
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





