#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段2：ICL推理主脚本
使用 retriever 选择 ICD，然后进行推理和评估
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
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

from open_mmicl.retriever import RandRetriever, QwenTopkRetriever
from open_mmicl.retriever.lever_lm_retriever import LeverLMRetriever
from open_mmicl.interface import LLaDAInterface
from open_mmicl.metrics import GSM8KMetrics, MMLUMetrics, SubjMetrics, TrecMetrics
from open_mmicl.prompt_template import PromptTemplate
from lever_lm.load_ds_utils import load_gsm8k_ds, load_mmlu_ds
from lever_lm.models import InsertionSelector
from lever_lm.utils import encode_text_qwen
from utils import load_ds
from llada_gsm8k_eval.llada_inference import create_llada_inference
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
    
    # 1. 加载数据集（与具体数据集解耦：由 task_name + dataset 配置决定）
    logger.info("Loading datasets...")
    if cfg.task.task_name == "gsm8k":
        train_ds = load_gsm8k_ds(
            version=cfg.dataset.version,
            data_path=cfg.dataset.train_path,
            split="train",
        )
        test_ds = load_gsm8k_ds(
            version=cfg.dataset.version,
            data_path=cfg.dataset.test_path,
            split="validation",
        )
    elif cfg.task.task_name == "mmlu":
        train_ds = load_ds(cfg, split="train")
        test_ds = load_ds(cfg, split="validation")
    else:
        # 通用分支：任意 task 使用 load_ds(cfg)，由 dataset 配置提供 train_path / test_path 等
        train_ds = load_ds(cfg, split="train")
        test_ds = load_ds(cfg, split="validation")
    
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Test dataset size: {len(test_ds)}")

    # 可选：根据自定义的 idx 子集，重写测试集（常用于从 train 中抽取未见过的 200 条进行评估）
    test_subset_ids_path = cfg.get("test_subset_ids_path", None)
    if test_subset_ids_path:
        try:
            with open(test_subset_ids_path, "r", encoding="utf-8") as f:
                subset_cfg = json.load(f)
            subset_ids = subset_cfg.get("query_ids", [])
            subset_id_set = set(int(i) for i in subset_ids)

            def _keep_example(ex):
                return int(ex.get("idx", -1)) in subset_id_set

            test_ds = train_ds.filter(_keep_example)
            logger.info(
                f"Using custom test subset from {test_subset_ids_path}, size={len(test_ds)}"
            )
        except Exception as e:
            logger.error(f"Failed to apply test_subset_ids_path={test_subset_ids_path}: {e}")

    # 2. 读取 prompt 配置（供 retriever 和 interface 复用）
    prompt_template = cfg.task.get("template", None)
    column_token_map = cfg.task.get("column_token_map", None)
    if column_token_map is not None:
        column_token_map = dict(column_token_map)

    mask_column_token_map = cfg.task.get("mask_column_token_map", None)
    if mask_column_token_map is not None and isinstance(mask_column_token_map, dict):
        mask_column_token_map = dict(mask_column_token_map)

    # 2. 读取生成相关配置：这里只支持通过 task.gen_args 配置，缺少就直接报错
    task_gen_args = cfg.task.get("gen_args", None)
    if task_gen_args is None:
        raise ValueError("configs/task/<task>.yaml 中必须提供 task.gen_args，缺少则无法进行 ICL 推理。")

    try:
        mask_length = int(task_gen_args.mask_length)
        steps = int(task_gen_args.steps)
        block_length = int(task_gen_args.block_length)
    except Exception as e:
        raise ValueError(f"读取 task.gen_args 中的 mask_length/steps/block_length 失败: {e}")

    # 生成其它超参：只从 infer_model.generation_kwargs 里读，读不到就用安全默认值
    gen_cfg = cfg.infer_model.get("generation_kwargs", {}) or {}
    temperature = float(gen_cfg.get("temperature", 0.0))
    cfg_scale = float(gen_cfg.get("cfg_scale", 0.0))
    remask_strategy = gen_cfg.get("remasking", "low_confidence")

    prompt_template_obj = None
    if prompt_template is not None:
        prompt_template_obj = PromptTemplate(
            prompt_template=prompt_template,
            mask_length=mask_length,
            column_token_map=column_token_map,
            mask_column_token_map=mask_column_token_map,
        )
    
    # 3. 初始化模型和tokenizer
    logger.info("Initializing model and tokenizer...")
    try:
        import torch
        from transformers import AutoTokenizer
        from model.model_llada import LLaDAModelLM
        
        # 获取模型路径
        model_path = cfg.infer_model.get("model_path", None)
        if model_path is None:
            model_path = cfg.infer_model.get("model_name", None)
        
        if model_path is None:
            raise ValueError(
                "Model path not found in config. Please set infer_model.model_path or infer_model.model_name"
            )
        
        logger.info(f"Loading model from {model_path}...")
        logger.info("Using local LLaDAModelLM (model/model_llada.py) with weights from model_path")
        
        # 准备模型参数
        model_kwargs = {
            "trust_remote_code": cfg.infer_model.get("trust_remote_code", True),
            "torch_dtype": getattr(torch, cfg.infer_model.get("torch_dtype", "bfloat16")),
            "local_files_only": cfg.infer_model.get("local_files_only", True),
        }
        
        # 使用本地 LLaDAModelLM 架构加载权重，避免走模型目录中的 remote modeling_llada.py
        model = LLaDAModelLM.from_pretrained(
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
        
        logger.info(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Please check your model_path configuration in configs/infer_model/llada.yaml")
        return

    # 基于已加载的 model/tokenizer 创建统一的 LLaDAInference 封装，后面统一用它来做生成
    llada_infer = create_llada_inference(
        model_path=model_path,
        device=str(device),
        tokenizer=tokenizer,
        model=model,
        mask_id=cfg.infer_model.mask_id,
        max_length=cfg.infer_model.get("max_length", 4096),
        torch_dtype=getattr(torch, cfg.infer_model.get("torch_dtype", "bfloat16")),
    )
    
    # 4. 初始化 retriever
    logger.info("Initializing retriever...")
    retriever_type = cfg.retriever.get("type", "rand")
    lever_lm_ckpt_tag = "na"  # 用于结果文件名，仅 lever_lm 时有意义
    resolved_lever_lm_ckpt_path: Optional[str] = None
    if retriever_type == "lever_lm":
        # utils.py 与 utils/ 包同名，显式从 utils.py 加载函数避免导入冲突
        from importlib.machinery import SourceFileLoader
        from importlib.util import module_from_spec, spec_from_loader

        utils_py_path = os.path.join(project_root, "utils.py")
        utils_spec = spec_from_loader("utils_file", SourceFileLoader("utils_file", utils_py_path))
        if utils_spec is None or utils_spec.loader is None:
            raise ImportError(f"Failed to load utils.py from {utils_py_path}")
        utils_module = module_from_spec(utils_spec)
        utils_spec.loader.exec_module(utils_module)
        get_lever_lm_path = utils_module.get_lever_lm_path
        init_lever_lm = utils_module.init_lever_lm
        lm_cfg = cfg.get("lever_lm", {})
        ckpt_path = lm_cfg.get("ckpt_path")
        if not ckpt_path or not os.path.exists(ckpt_path):
            ckpt_dir = lm_cfg.get("ckpt_dir", "generated_icd_data/model_cpk/lever_lm")
            default_key = lm_cfg.get("default_cpk_key", "min_vl")
            ckpt_path = get_lever_lm_path(ckpt_dir, default_key)
            logger.info(f"LeverLM checkpoint: {ckpt_path}")
        resolved_lever_lm_ckpt_path = str(ckpt_path) if ckpt_path else None
        try:
            lever_lm_ckpt_tag = os.path.splitext(os.path.basename(str(ckpt_path).rstrip("/")))[0]
        except Exception as _e:
            logger.warning(f"Failed to parse lever_lm ckpt tag from {ckpt_path}: {_e}")
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
        # 与训练时 PromptTemplate.generate_text_for_embedding 对齐，确保 query embedding 格式匹配
        if prompt_template_obj is None:
            raise ValueError("lever_lm retriever requires task.template to build embedding text.")
        output_column = cfg.task.get("output_column", None)
        query_text_extractor = lambda s: prompt_template_obj.generate_text_for_embedding(
            s, output_column=output_column
        )
        retriever = LeverLMRetriever(
            index_ds=train_ds,
            lever_lm=lever_lm,
            qwen_model_path=lm_cfg.get("qwen_model_path", "Qwen/Qwen3-Embedding-4B"),
            query_text_extractor=query_text_extractor,
            nshot=cfg.retriever.nshot,
            device=lm_cfg.get("device", "cuda:0"),
            batch_size=lm_cfg.get("batch_size", 32),
            reverse_seq=lm_cfg.get("reverse_seq", False),
            debug=bool(lm_cfg.get("debug", False)),
            debug_prefix_chars=int(lm_cfg.get("debug_prefix_chars", 160)),
        )
    elif retriever_type == "qwen_topk":
        # 基于 Qwen-Embedding 的 TopK 相似度检索：
        # - 候选池：训练集
        # - 全局排除：test_subset_ids_path 中的 query_ids + forbidden_ids
        qwen_forbidden: set[int] = set()
        subset_cfg_path = cfg.get("test_subset_ids_path", None)
        if subset_cfg_path:
            try:
                with open(subset_cfg_path, "r", encoding="utf-8") as f:
                    subset_cfg = json.load(f)
                query_ids = subset_cfg.get("query_ids", [])
                forbidden_ids = subset_cfg.get("forbidden_ids", [])
                qwen_forbidden = {int(x) for x in list(query_ids) + list(forbidden_ids)}
                if qwen_forbidden:
                    logger.info(
                        f"QwenTopkRetriever will avoid {len(qwen_forbidden)} indices "
                        f"loaded from {subset_cfg_path} (query_ids + forbidden_ids)"
                    )
            except Exception as e:
                logger.warning(f"Failed to load query_ids/forbidden_ids from {subset_cfg_path}: {e}")

        if prompt_template_obj is None:
            raise ValueError("qwen_topk retriever requires task.template to build embedding text.")

        output_column = cfg.task.get("output_column", None)
        query_text_extractor = lambda s: prompt_template_obj.generate_text_for_embedding(
            s, output_column=output_column
        )

        # 复用 lever_lm 段中的 Qwen 配置（模型路径 / 设备 / batch size / embedding_path）
        lm_cfg = cfg.get("lever_lm", {})
        qwen_model_path = lm_cfg.get("qwen_model_path", "Qwen/Qwen3-Embedding-4B")
        feature_cache = lm_cfg.get("embedding_path", None)

        retriever = QwenTopkRetriever(
            index_ds=train_ds,
            qwen_model_path=qwen_model_path,
            query_text_extractor=query_text_extractor,
            nshot=cfg.retriever.nshot,
            device=lm_cfg.get("device", "cuda:0"),
            encode_batch_size=lm_cfg.get("batch_size", 64),
            feature_cache=feature_cache,
            overwrite_cache=bool(lm_cfg.get("overwrite_qwen_cache", False)),
            # 为了“最相似的放最后”，这里固定 reverse_seq=True
            reverse_seq=True,
            forbidden_indices=qwen_forbidden,
        )
    else:
        # rand 模式：可选地从 test_subset_ids_path 中读取 forbidden_ids，
        # 让随机 ICD 不与 Stage1 JSON 中出现过的样本重合。
        rand_forbidden: set[int] = set()
        subset_cfg_path = cfg.get("test_subset_ids_path", None)
        if subset_cfg_path:
            try:
                with open(subset_cfg_path, "r", encoding="utf-8") as f:
                    subset_cfg = json.load(f)
                # 若 build_mmlu_test_utils.py 同时导出了 forbidden_ids，则一并使用
                forbidden_ids = subset_cfg.get("forbidden_ids", [])
                rand_forbidden = {int(x) for x in forbidden_ids}
                if rand_forbidden:
                    logger.info(
                        f"RandRetriever will avoid {len(rand_forbidden)} globally forbidden indices "
                        f"loaded from {subset_cfg_path}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load forbidden_ids from {subset_cfg_path}: {e}")

        retriever = RandRetriever(
            train_ds=train_ds,
            nshot=cfg.retriever.nshot,
            seed=cfg.retriever.get("seed", 42),
            forbidden_indices=rand_forbidden,
        )

    # 4b. InsertionSelector（第二个分类器）
    # rand / qwen_topk：可单独使用（query 放最后，query_position=0）；也可启用 insertion_selector 预测位置。
    # lever_lm：必须启用 insertion_selector（第一个分类器不能单独使用），要么串联选 ICD+预测位置，要么在 rand/qwen_topk 采样的 ICD 上仅用 Selector 预测位置。
    insertion_selector_obj = None
    is_cfg = cfg.get("insertion_selector", {})
    if retriever_type == "lever_lm":
        if not is_cfg.get("enabled", False):
            raise ValueError(
                "retriever.type=lever_lm 时必须启用 insertion_selector（第一个分类器不能单独使用）。"
                "请在 config 中设置 insertion_selector.enabled=true 并配置 ckpt_path/ckpt_dir。"
            )
    if is_cfg.get("enabled", False):
        if prompt_template_obj is None:
            raise ValueError(
                "insertion_selector.enabled=true 需要 task.template 以格式化 query 和 ICD 文本（用于 encode_text_qwen）。"
            )
        ckpt_path = is_cfg.get("ckpt_path")
        if not ckpt_path or not os.path.exists(str(ckpt_path)):
            import glob
            ckpt_dir = is_cfg.get("ckpt_dir", "generated_icd_data/checkpoints/insertion_selector")
            default_key = is_cfg.get("default_cpk_key", "min_vl")
            if default_key == "min_vl":
                pattern = os.path.join(ckpt_dir, "min_vl-*.ckpt")
            elif default_key == "min_tl":
                pattern = os.path.join(ckpt_dir, "min_tl-*.ckpt")
            else:
                last_path = os.path.join(ckpt_dir, "last.ckpt")
                pattern = os.path.join(ckpt_dir, "last*.ckpt")
                if os.path.exists(last_path):
                    ckpt_path = last_path
            if ckpt_path is None or not os.path.exists(str(ckpt_path)):
                candidates = glob.glob(pattern)
                ckpt_path = max(candidates, key=os.path.getmtime) if candidates else None
            if ckpt_path:
                logger.info(f"InsertionSelector checkpoint: {ckpt_path}")
        if not ckpt_path or not os.path.exists(str(ckpt_path)):
            raise FileNotFoundError(
                f"InsertionSelector checkpoint not found. Set insertion_selector.ckpt_path or insertion_selector.ckpt_dir (default_cpk_key={is_cfg.get('default_cpk_key', 'min_vl')})."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        new_state = {k.replace("model.", ""): v for k, v in state.items() if k.startswith("model.")}
        emb_dim = int(is_cfg.get("emb_dim", 2560))
        n_head = int(is_cfg.get("n_head", 16))
        dropout = float(is_cfg.get("dropout", 0.2))
        insertion_selector_obj = InsertionSelector(emb_dim=emb_dim, n_head=n_head, dropout=dropout)
        insertion_selector_obj.load_state_dict(new_state, strict=True)
        insertion_selector_obj.to(is_cfg.get("device", "cuda:0"))
        insertion_selector_obj.eval()
        logger.info("InsertionSelector loaded (query position will be predicted per sample).")
    
    # 5. 初始化 interface（只用于构建 prompt / 解析答案，不再自己实现生成逻辑）
    logger.info("Initializing interface...")
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
    elif cfg.task.task_name == "subj":
        metrics = SubjMetrics()
    elif cfg.task.task_name == "trec":
        metrics = TrecMetrics()
    else:
        metrics = MMLUMetrics()  # 其他 task 暂用 MMLUMetrics
        logger.info(f"Using MMLUMetrics for task_name={cfg.task.task_name}")
    
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
            # Debug: only log first few samples' ICD indices
            if idx < 20:
                logger.info(f"[DEBUG] icd_indices for test sample {idx}: {icd_indices}")
            
            # 获取ICD数据
            icd_samples = [train_ds[i] for i in icd_indices]
            nshot = len(icd_samples)

            # 确定 query 位置：方案 A，不依赖预计算 test.pt
            # 若启用 InsertionSelector：用 PromptTemplate 得到 query/ICD 文本 -> encode_text_qwen -> Selector -> query_position = nshot - pred
            # 否则（rand/qwen_topk 单独使用）：query 放最后，query_position=0
            if insertion_selector_obj is not None:
                output_column = cfg.task.get("output_column", None)
                query_text = prompt_template_obj.generate_text_for_embedding(
                    test_sample, output_column=output_column
                )
                icd_texts = [prompt_template_obj.generate_ice_item(s) for s in icd_samples]
                text_list = [query_text] + icd_texts
                is_cfg = cfg.get("insertion_selector", {})
                qwen_path = is_cfg.get("qwen_model_path") or cfg.get("lever_lm", {}).get(
                    "qwen_model_path", "Qwen/Qwen3-Embedding-4B"
                )
                feats = encode_text_qwen(
                    text_list=text_list,
                    model_name=qwen_path,
                    device=is_cfg.get("device", "cuda:0"),
                    batch_size=len(text_list),
                    normalize=True,
                )
                device_sel = torch.device(is_cfg.get("device", "cuda:0"))
                query_emb = feats[0:1].to(device_sel)
                icd_embs = feats[1:].unsqueeze(0).to(device_sel)
                with torch.no_grad():
                    out = insertion_selector_obj(query_emb, icd_embs)
                pred = out["logits"].argmax(dim=-1).item()
                query_position = nshot - pred  # Selector gap 0=最前，gap nshot=最后
            else:
                query_position = cfg.get("query_position", 0)  # rand/qwen_topk 单独使用时默认 0=最后

            # 构建prompt
            prompt = interface.build_prompt(
                ice_samples=icd_samples,
                test_sample=test_sample,
                query_position=query_position,
            )

            # Debug: print the exact prompt fed into the model (first sample only)
            debug_cfg = cfg.get("debug", {})
            if debug_cfg and debug_cfg.get("print_prompt", False) and idx == 0:
                max_chars = int(debug_cfg.get("prompt_max_chars", 4000))
                logger.info("=" * 80)
                logger.info("[DEBUG] Final prompt fed to LLaDA (idx=0)")
                logger.info(f"[DEBUG] query_position={query_position}, n_shots={len(icd_samples)}")
                logger.info(f"[DEBUG] ground_truth(answer)={test_sample.get('answer', '')}")
                if len(prompt) > max_chars:
                    logger.info(prompt[:max_chars] + "\n...[TRUNCATED]...")
                else:
                    logger.info(prompt)
                logger.info("=" * 80)
            
            # 生成：直接复用 llada_gsm8k_eval/llada_inference.py 中的 LLaDAInference 逻辑
            # 只支持通过 task.gen_args 配置的 mask_length / steps / block_length，其它复杂用法一律不处理
            try:
                generated_text = llada_infer.generate_text(
                    prompt=prompt,
                    answer_length=mask_length,
                    sampling_steps=steps,
                    block_length=block_length,
                    remask_strategy=remask_strategy,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    stop_tokens=None,
                )
            except Exception as _e:
                logger.error(f"LLaDAInference.generate_text 失败 (test sample {idx}): {_e}")
                raise
            
            # 提取预测答案（记录在 JSON 里，便于人工检查）
            if cfg.task.task_name == "mmlu":
                # 直接按 MMLU 规则从答案片段里抽取 A/B/C/D
                predicted_answer = MMLUMetrics.extract_choice(generated_text)
                if not predicted_answer:
                    predicted_answer = generated_text
            else:
                # 其他任务暂时沿用通用的 extract_answer 逻辑
                predicted_answer = interface.extract_answer(generated_text)
            
            # 获取真实答案
            ground_truth = test_sample.get("answer", "")

            # 仅对 MMLU 做 0/1/2/3 -> A/B/C/D 的规范化，其他 task 保持原样
            if cfg.task.task_name == "mmlu":
                gt = str(ground_truth).strip()
                idx2label = {"0": "A", "1": "B", "2": "C", "3": "D"}
                if gt.upper() in {"A", "B", "C", "D"}:
                    ground_truth = gt.upper()
                elif gt in idx2label:
                    ground_truth = idx2label[gt]
                else:
                    ground_truth = gt

            # 评估
            eval_result = metrics.evaluate_single(generated_text, ground_truth)
            
            # 保存结果（query_position 采用 build_prompt 的约定：0=query 在最后，nshot=在最前；
            # 与 InsertionSelector 的 gap 约定相反：selector 的 pred 0=最前、nshot=最后，已用 query_position=nshot-pred 转换）
            result = {
                "test_idx": idx,
                "test_sample_idx": test_sample.get("idx", idx),
                "icd_indices": icd_indices,
                "query_position": query_position,  # 0=最后，nshot=最前
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
    
    # 8. 保存结果：output_dir/当前日期/ 下，文件名末尾带当前时间（时-分-秒）
    base_output_dir = Path(cfg.get("output_dir", "./icl_inference_results"))
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    time_suffix = now.strftime("%H-%M-%S")
    output_dir = base_output_dir / date_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 根据 retriever.type、retriever.nshot、max_samples、ckpt_path 构造更可读的文件名，末尾加时间
    retriever_type = cfg.retriever.get("type", "unknown")
    nshot = cfg.retriever.get("nshot", "na")
    max_samples = cfg.get("max_samples", None)
    max_samples_str = str(max_samples) if max_samples is not None else "all"
    lever_tag = lever_lm_ckpt_tag if retriever_type == "lever_lm" else "na"

    filename = (
        f"{cfg.task.task_name}_"
        f"retr-{retriever_type}_"
        f"nshot-{nshot}_"
        f"max-{max_samples_str}_"
        f"ckpt-{lever_tag}_inference_results_{time_suffix}.json"
    )
    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        run_config = {
            "task_name": cfg.task.task_name,
            "retriever": {
                "type": retriever_type,
                "nshot": nshot,
            },
            "test_subset_ids_path": cfg.get("test_subset_ids_path", None),
            "max_samples": max_samples,
            # 仅当 retriever.type=lever_lm 时该字段才有意义；否则为 None
            "ckpt_path": resolved_lever_lm_ckpt_path,
        }
        json.dump({
            "run_config": run_config,
            "metrics": batch_metrics,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()