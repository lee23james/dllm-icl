#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段2（变体）：基于 PPL / log-likelihood 的 ICL 推理脚本

适用任务：
- trec / subj 等分类任务：标签集合有限，可从 label_mapping 读取

与 icl_inference.py 的主要区别：
- 仍然使用 retriever 选 ICD（rand / qwen_topk / lever_lm）
- 不再进行 mask 生成，而是对每个候选标签计算 log P(label | ICD + query)
- 选择 log-likelihood 最高的标签作为预测结果
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import hydra
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm

# 添加项目路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from open_mmicl.retriever import RandRetriever, QwenTopkRetriever
from open_mmicl.retriever.lever_lm_retriever import LeverLMRetriever
from open_mmicl.interface import LLaDAInterface
from open_mmicl.prompt_template import PromptTemplate
from open_mmicl.metrics import SubjMetrics, TrecMetrics, MMLUMetrics
from lever_lm.load_ds_utils import load_gsm8k_ds, load_mmlu_ds
from lever_lm.models import InsertionSelector
from lever_lm.utils import encode_text_qwen
from utils import load_ds


def _resolve_label_mapping(cfg: DictConfig) -> Dict[str, str]:
    """优先从 task，再从 dataset 读取 label_mapping。"""
    label_mapping = cfg.task.get("label_mapping") or cfg.dataset.get("label_mapping")
    if not label_mapping:
        raise ValueError(
            f"Task {cfg.task.task_name} requires label_mapping in task or dataset config "
            "for PPL-based classification."
        )
    # 统一成 str -> str
    return {str(k): str(v) for k, v in label_mapping.items()}


def _build_prompt_texts(
    pt: PromptTemplate,
    left_icd_samples: List[Dict[str, Any]],
    right_icd_samples: List[Dict[str, Any]],
    query_sample: Dict[str, Any],
    output_column: str,
    split_token: str,
) -> Tuple[str, Optional[str]]:
    """
    构建左侧 prompt 文本（ICD + 无答案版 query），用于 log-likelihood 计算。
    逻辑与 utils.get_info_score 中 PromptTemplate 分支保持一致的思想：
    - left_text = join( icd_prompts + query_upper )
    其中 query_upper 通过 generate_text_for_embedding 去掉 output_column 字段。
    """
    sep = split_token

    # 1) 左侧 ICD prompts（完整）
    left_icd_texts: List[str] = []
    for s in left_icd_samples:
        icd_text = pt.generate_ice_item(s)
        left_icd_texts.append(icd_text.strip())

    # 2) 无答案版 query（去掉 output_column，例如 answer）
    query_upper = pt.generate_text_for_embedding(
        query_sample,
        output_column=output_column,
    ).strip()

    left_parts = left_icd_texts + [query_upper]
    left_text = sep.join(left_parts)

    # 3) 右侧 ICD prompts（完整），如果 query 不在末尾
    right_text: Optional[str] = None
    if right_icd_samples:
        right_icd_texts: List[str] = []
        for s in right_icd_samples:
            icd_text = pt.generate_ice_item(s)
            right_icd_texts.append(icd_text.strip())
        right_text = sep.join(right_icd_texts)

    return left_text, right_text


@hydra.main(version_base=None, config_path="configs", config_name="icl_inference")
def main(cfg: DictConfig):
    """
    基于 PPL / log-likelihood 的 ICL 推理（仅支持有限标签集合的分类任务，如 subj/trec）。
    """
    logger.info("=" * 80)
    logger.info("Stage 2 (PPL): ICL Inference with label-wise log-likelihood")
    logger.info("=" * 80)

    # 1. 加载数据集
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
        train_ds = load_ds(cfg, split="train")
        test_ds = load_ds(cfg, split="validation")

    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Test dataset size: {len(test_ds)}")

    # 仅支持有 label_mapping 的任务（如 subj/trec/race）
    label_mapping = _resolve_label_mapping(cfg)
    label_texts: List[str] = list(label_mapping.values())
    logger.info(f"Using {len(label_texts)} labels for PPL inference: {label_texts}")

    # 可选：根据 idx 子集重写测试集
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

    # 2. 读取 prompt 配置（与 icl_inference.py 保持一致）
    prompt_template = cfg.task.get("template", None)
    logger.debug(f"Initial prompt_template from config: {prompt_template}")

    if prompt_template is not None and isinstance(prompt_template, str) and prompt_template.startswith("${"):
        logger.warning(f"prompt_template appears to be unresolved: {prompt_template}, trying to resolve...")
        try:
            template_key = prompt_template.replace("${infer_model.prompt_templates.", "").replace("}", "")
            prompt_template = cfg.infer_model.prompt_templates.get(template_key, None)
            if prompt_template is None:
                logger.error(f"Failed to resolve prompt_template key: {template_key}")
            else:
                logger.info(f"Successfully resolved prompt_template to: {prompt_template[:100]}...")
        except Exception as e:
            logger.error(f"Failed to resolve prompt_template: {e}")
            prompt_template = None

    column_token_map = cfg.task.get("column_token_map", None)
    if column_token_map is not None:
        column_token_map = dict(column_token_map)
        logger.debug(f"column_token_map: {column_token_map}")

    mask_column_map = cfg.task.get("mask_column_token_map", None)
    if mask_column_map is not None and isinstance(mask_column_map, dict):
        mask_column_map = dict(mask_column_map)
    logger.debug(f"mask_column_token_map: {mask_column_map}")

    # 生成相关配置（主要用 mc_num）
    task_gen_args = cfg.task.get("gen_args", None)
    if task_gen_args is None:
        raise ValueError("configs/task/<task>.yaml 中必须提供 task.gen_args。")
    try:
        mc_num = int(task_gen_args.get("mc_num", 128))
    except Exception as e:
        raise ValueError(f"读取 task.gen_args.mc_num 失败: {e}")

    # 若 prompt_template 仍为空，对 subj/trec 使用默认模板
    if prompt_template is None and cfg.task.task_name in ("trec", "subj"):
        logger.warning(f"prompt_template is None for {cfg.task.task_name}, using default template")
        prompt_template = "Input: <S>\nType: <A>"
        if column_token_map is None:
            column_token_map = {"sentence": "<S>", "answer": "<A>"}
        if mask_column_map is None:
            mask_column_map = "answer"

    # 初始化 PromptTemplate，便于后续构建文本
    prompt_template_obj: Optional[PromptTemplate] = None
    if prompt_template is not None:
        try:
            prompt_template_obj = PromptTemplate(
                prompt_template=prompt_template,
                mask_length=task_gen_args.mask_length,
                column_token_map=column_token_map,
                mask_column_token_map=mask_column_map,
            )
            logger.debug(f"PromptTemplate initialized with template: {prompt_template[:100]}...")
        except Exception as e:
            logger.error(f"Failed to initialize PromptTemplate: {e}")
            prompt_template_obj = None

    # 3. 初始化模型和 tokenizer（与 icl_inference.py 相同）
    logger.info("Initializing model and tokenizer...")
    try:
        from transformers import AutoTokenizer
        from model.model_llada import LLaDAModelLM

        model_path = cfg.infer_model.get("model_path") or cfg.infer_model.get("model_name")
        if model_path is None:
            raise ValueError("Model path not found in config. Please set infer_model.model_path or infer_model.model_name")

        logger.info(f"Loading model from {model_path}...")
        logger.info("Using local LLaDAModelLM (model/model_llada.py) with weights from model_path")

        model_kwargs = {
            "trust_remote_code": cfg.infer_model.get("trust_remote_code", True),
            "torch_dtype": getattr(torch, cfg.infer_model.get("torch_dtype", "bfloat16")),
            "local_files_only": cfg.infer_model.get("local_files_only", True),
        }

        model = LLaDAModelLM.from_pretrained(model_path, **model_kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

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

    # 4. 初始化 retriever（与 icl_inference.py 中逻辑基本一致）
    logger.info("Initializing retriever...")
    retriever_type = cfg.retriever.get("type", "rand")
    lever_lm_ckpt_tag = "na"
    resolved_lever_lm_ckpt_path: Optional[str] = None

    if retriever_type == "lever_lm":
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
            reverse_seq=True,
            forbidden_indices=qwen_forbidden,
        )
    else:
        rand_forbidden: set[int] = set()
        subset_cfg_path = cfg.get("test_subset_ids_path", None)
        if subset_cfg_path:
            try:
                with open(subset_cfg_path, "r", encoding="utf-8") as f:
                    subset_cfg = json.load(f)
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

    # 4b. InsertionSelector（若启用，仅用于预测 query_position）
    insertion_selector_obj = None
    is_cfg = cfg.get("insertion_selector", {})
    if retriever_type == "lever_lm" and not is_cfg.get("enabled", False):
        raise ValueError(
            "retriever.type=lever_lm 时必须启用 insertion_selector。"
        )
    if is_cfg.get("enabled", False):
        if prompt_template_obj is None:
            raise ValueError(
                "insertion_selector.enabled=true 需要 task.template 以格式化文本。"
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
                f"InsertionSelector checkpoint not found. Set insertion_selector.ckpt_path or insertion_selector.ckpt_dir."
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

    # 5. 初始化 interface（scoring 模式：不使用 mask）
    logger.info("Initializing interface (scoring mode)...")
    interface = LLaDAInterface(
        model=model,
        tokenizer=tokenizer,
        task=cfg.task.task_name,
        mask_id=cfg.infer_model.mask_id,
        mask_length=task_gen_args.mask_length,
        split_token=cfg.task.get("split_token", "\n\n"),
        prompt_template=prompt_template,
        is_scoring_mode=True,
        column_token_map=column_token_map,
        mask_column_token_map=mask_column_map,
    )

    # 6. PPL 推理循环
    logger.info("Starting PPL-based inference...")
    results: List[Dict[str, Any]] = []

    max_samples = cfg.get("max_samples", None)
    if max_samples:
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))
        logger.info(f"Limited to {len(test_ds)} test samples")

    output_column = cfg.task.get("output_column", "answer")
    split_token = cfg.task.get("split_token", "\n\n")

    def encode(text: str) -> torch.Tensor:
        ids = tokenizer(text)["input_ids"]
        return torch.tensor(ids, device=interface.device)

    for idx, test_sample in enumerate(tqdm(test_ds, desc="Inference(PPL)", ncols=100)):
        try:
            icd_indices = retriever.retrieve(
                test_sample=test_sample,
                exclude_indices=[test_sample.get("idx")] if "idx" in test_sample else [],
            )
            if idx < 20:
                logger.info(f"[DEBUG] icd_indices for test sample {idx}: {icd_indices}")

            icd_samples = [train_ds[i] for i in icd_indices]
            nshot = len(icd_samples)

            # 预测 query_position（若使用 Selector）
            if insertion_selector_obj is not None and prompt_template_obj is not None:
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
                query_position = nshot - pred
            else:
                query_position = cfg.get("query_position", 0)

            # 为 PPL scoring 构建统一的 left/right 文本（ICD_left/right + 无答案版 query）
            test_sample_for_prompt = dict(test_sample)
            test_sample_for_prompt["isquery"] = 1
            if prompt_template_obj is None:
                raise ValueError("prompt_template_obj is required for PPL scoring.")

            # 根据 query_position 将 ICD 拆分为左/右两部分
            icd_count = len(icd_samples)
            max_position = icd_count
            norm_pos = max(0, min(int(query_position), max_position))
            insertion_idx = icd_count - norm_pos  # 与 LLaDAInterface.build_prompt 中的逻辑保持一致

            left_icd_samples = icd_samples[:insertion_idx]
            right_icd_samples = icd_samples[insertion_idx:]

            left_text, right_text = _build_prompt_texts(
                pt=prompt_template_obj,
                left_icd_samples=left_icd_samples,
                right_icd_samples=right_icd_samples,
                query_sample=test_sample_for_prompt,
                output_column=output_column,
                split_token=split_token,
            )
            prompt_left_ids = encode(left_text)
            prompt_right_ids = encode(right_text) if right_text else None

            # 对每个 label 候选计算 log-likelihood
            label_scores: Dict[str, float] = {}
            for label_text in label_texts:
                answer_ids = encode(str(label_text))
                score = interface.compute_score_with_mc(
                    prompt_left=prompt_left_ids,
                    answer=answer_ids,
                    prompt_right=prompt_right_ids,
                    mc_num=mc_num,
                    batch_size=1,
                    cfg_scale=0.0,
                )
                label_scores[label_text] = float(score)

            # 选择分数最高的 label
            if not label_scores:
                logger.error(f"No label scores computed for test sample {idx}")
                continue
            predicted_label = max(label_scores.items(), key=lambda x: x[1])[0]

            raw_ground_truth = str(test_sample.get("answer", ""))

            # 根据任务选择 metrics 逻辑（只用来统一输出结构）
            if cfg.task.task_name == "trec":
                eval_result = TrecMetrics.evaluate_single(predicted_label, raw_ground_truth)
            elif cfg.task.task_name == "subj":
                eval_result = SubjMetrics.evaluate_single(predicted_label, raw_ground_truth)
            elif cfg.task.task_name == "mmlu":
                # MMLU: ground truth 可能是 0/1/2/3 或 A/B/C/D，统一归一化后与 label 比较
                truth = MMLUMetrics._normalize_truth(raw_ground_truth)
                is_correct = (predicted_label.strip().upper() == truth)
                eval_result = {
                    "predicted": predicted_label,
                    "ground_truth": truth,
                    "is_correct": is_correct,
                    "generated_text": predicted_label,
                }
            else:
                # 默认 exact-match（大小写不敏感）
                is_correct = (
                    predicted_label.strip().lower() == raw_ground_truth.strip().lower()
                )
                eval_result = {
                    "predicted": predicted_label,
                    "ground_truth": raw_ground_truth,
                    "is_correct": is_correct,
                    "generated_text": predicted_label,
                }

            result = {
                "test_idx": idx,
                "test_sample_idx": test_sample.get("idx", idx),
                "icd_indices": icd_indices,
                "query_position": query_position,
                "label_scores": label_scores,
                "predicted": predicted_label,
                "ground_truth": eval_result["ground_truth"],
                "is_correct": eval_result["is_correct"],
                "left_prompt": left_text,
            }
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing test sample {idx}: {e}")
            continue

    # 7. 汇总指标
    logger.info("Calculating metrics (PPL-based)...")
    correct_count = sum(1 for r in results if r.get("is_correct"))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Correct: {correct_count}/{total_count}")

    # 8. 保存结果
    base_output_dir = Path(cfg.get("output_dir", "./generated_icd_data/icl_inference_results_ppl"))
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    time_suffix = now.strftime("%H-%M-%S")
    output_dir = base_output_dir / date_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
        f"ckpt-{lever_tag}_ppl_inference_results_{time_suffix}.json"
    )
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        run_config = {
            "task_name": cfg.task.task_name,
            "retriever": {
                "type": retriever_type,
                "nshot": nshot,
            },
            "test_subset_ids_path": cfg.get("test_subset_ids_path", None),
            "max_samples": max_samples,
            "ckpt_path": resolved_lever_lm_ckpt_path,
            "mc_num": mc_num,
        }
        json.dump(
            {
                "run_config": run_config,
                "metrics": {
                    "accuracy": accuracy,
                    "correct_count": correct_count,
                    "total_count": total_count,
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"PPL-based results saved to: {output_path}")
    return results


if __name__ == "__main__":
    main()

