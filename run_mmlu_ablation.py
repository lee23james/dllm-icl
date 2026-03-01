#!/usr/bin/env python3
"""
MMLU 消融：ICD 由 retriever_type（rand / lever_lm / qwen_topk）选择，InsertionSelector 预测 query 位置，llada 推理。
需先运行 precompute 得到 train.pt / test.pt（或 scripts/precompute_mmlu_test_embedding.py 得到 test.pt）。
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from tqdm import tqdm

from open_mmicl.retriever import RandRetriever, QwenTopkRetriever
from open_mmicl.retriever.lever_lm_retriever import LeverLMRetriever
from open_mmicl.interface import LLaDAInterface
from open_mmicl.metrics import MMLUMetrics
from open_mmicl.prompt_template import PromptTemplate
from lever_lm.models import InsertionSelector
from utils import load_ds


def load_selector_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Load InsertionSelector from Lightning checkpoint (strip model. prefix)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    new_state = {k.replace("model.", ""): v for k, v in state.items() if k.startswith("model.")}
    selector = InsertionSelector(emb_dim=2560, n_head=16, dropout=0.2)
    selector.load_state_dict(new_state, strict=True)
    selector.to(device)
    selector.eval()
    return selector


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="icl_inference")
def main(cfg: DictConfig):
    # Dataset: prefer mmlu_ablation if exists, else fall back to mmlu
    mmlu_ablation_path = PROJECT_ROOT / "configs" / "dataset" / "mmlu_ablation.yaml"
    if mmlu_ablation_path.exists():
        cfg.dataset = OmegaConf.load(mmlu_ablation_path)
    else:
        cfg.dataset = OmegaConf.load(PROJECT_ROOT / "configs" / "dataset" / "mmlu.yaml")
    for key in ("train_path", "test_path"):
        if key in cfg.dataset and cfg.dataset[key]:
            p = Path(cfg.dataset[key])
            if not p.is_absolute():
                cfg.dataset[key] = str(PROJECT_ROOT / p)
    cfg.retriever.nshot = 3
    cfg.retriever.seed = 42

    retriever_type = cfg.get("ablation", {}).get("retriever_type", "rand")

    logger.info("=" * 80)
    logger.info(f"MMLU Ablation: retriever_type={retriever_type} + InsertionSelector for query position")
    logger.info("=" * 80)

    # Paths (override via env or defaults)
    selector_ckpt = Path(cfg.get("ablation", {}).get("selector_checkpoint") or "checkpoints/epoch=8-step=1881.ckpt")
    train_pt = Path(cfg.get("ablation", {}).get("train_embedding_pt") or "generated_icd_data/embedding_cache/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt")
    test_pt = Path(cfg.get("ablation", {}).get("test_embedding_pt") or "generated_icd_data/embedding_cache/mmlu-test-Qwen3-Embedding-4B-TextFeatures.pt")
    if not selector_ckpt.is_absolute():
        selector_ckpt = PROJECT_ROOT / selector_ckpt
    if not train_pt.is_absolute():
        train_pt = PROJECT_ROOT / train_pt
    if not test_pt.is_absolute():
        test_pt = PROJECT_ROOT / test_pt

    if not selector_ckpt.exists():
        raise FileNotFoundError(f"Selector checkpoint not found: {selector_ckpt}")
    if not train_pt.exists():
        raise FileNotFoundError(f"Train embedding .pt not found: {train_pt}. Run precompute for train or use existing MMLU train .pt.")
    if not test_pt.exists():
        raise FileNotFoundError(f"Test embedding .pt not found: {test_pt}. Run: python scripts/precompute_mmlu_test_embedding.py")

    # 1. Load data
    logger.info("Loading datasets...")
    train_ds = load_ds(cfg, split="train")
    test_ds = load_ds(cfg, split="validation")
    logger.info(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

    # 2. Load LLaDA model and tokenizer
    logger.info("Loading LLaDA model...")
    model_path = cfg.infer_model.get("model_path") or cfg.infer_model.get("model_name")
    if not model_path:
        raise ValueError("infer_model.model_path or model_name required")
    model_kwargs = {
        "trust_remote_code": cfg.infer_model.get("trust_remote_code", True),
        "torch_dtype": getattr(torch, cfg.infer_model.get("torch_dtype", "bfloat16")),
        "local_files_only": cfg.infer_model.get("local_files_only", True),
    }
    # 使用本地 LLaDAModelLM 实现，避免与 transformers 版本的 all_tied_weights_keys 不兼容
    from transformers import AutoTokenizer
    from model.model_llada import LLaDAModelLM
    model = LLaDAModelLM.from_pretrained(model_path, **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_kwargs["trust_remote_code"],
        local_files_only=model_kwargs["local_files_only"],
    )

    # 3. Retriever (by ablation.retriever_type) and interface
    prompt_template = cfg.task.get("template", None)
    column_token_map = OmegaConf.to_container(cfg.task.column_token_map) if cfg.task.get("column_token_map") else None
    mask_column_token_map = cfg.task.get("mask_column_token_map", None)
    mask_length = 2
    if cfg.task.get("gen_args") and "mask_length" in cfg.task.gen_args:
        mask_length = int(cfg.task.gen_args.mask_length)
    nshot = 3

    if retriever_type == "lever_lm":
        from importlib.machinery import SourceFileLoader
        from importlib.util import module_from_spec, spec_from_loader
        utils_py_path = PROJECT_ROOT / "utils.py"
        utils_spec = spec_from_loader("utils_file", SourceFileLoader("utils_file", str(utils_py_path)))
        if utils_spec is None or utils_spec.loader is None:
            raise ImportError(f"Failed to load utils.py from {utils_py_path}")
        utils_module = module_from_spec(utils_spec)
        utils_spec.loader.exec_module(utils_module)
        get_lever_lm_path = utils_module.get_lever_lm_path
        init_lever_lm = utils_module.init_lever_lm
        lm_cfg = cfg.get("lever_lm", cfg.get("ablation", {}))
        ckpt_path = lm_cfg.get("ckpt_path")
        if not ckpt_path or not os.path.exists(str(ckpt_path)):
            ckpt_dir = lm_cfg.get("ckpt_dir", "generated_icd_data/model_cpk/mmlu/debug")
            default_key = lm_cfg.get("default_cpk_key", "min_vl")
            ckpt_path = get_lever_lm_path(ckpt_dir, default_key)
        emb_path = lm_cfg.get("embedding_path")
        if not emb_path or not os.path.exists(emb_path):
            raise ValueError("lever_lm.embedding_path (or ablation) must exist for retriever_type=lever_lm")
        lever_lm, _ = init_lever_lm(
            cfg=cfg,
            ckpt_path=ckpt_path,
            train_ds=train_ds,
            embedding_path=emb_path,
            device=lm_cfg.get("device", "cuda:0"),
        )
        output_column = cfg.task.get("output_column", None)
        prompt_template_obj = PromptTemplate(
            prompt_template=prompt_template,
            mask_length=mask_length,
            column_token_map=column_token_map,
            mask_column_token_map=mask_column_token_map,
        )
        query_text_extractor = lambda s: prompt_template_obj.generate_text_for_embedding(
            s, output_column=output_column
        )
        retriever = LeverLMRetriever(
            index_ds=train_ds,
            lever_lm=lever_lm,
            qwen_model_path=lm_cfg.get("qwen_model_path", "Qwen/Qwen3-Embedding-4B"),
            query_text_extractor=query_text_extractor,
            nshot=nshot,
            device=lm_cfg.get("device", "cuda:0"),
            batch_size=lm_cfg.get("batch_size", 32),
            reverse_seq=lm_cfg.get("reverse_seq", False),
        )
    elif retriever_type == "qwen_topk":
        if prompt_template is None:
            raise ValueError("qwen_topk requires task.template")
        prompt_template_obj = PromptTemplate(
            prompt_template=prompt_template,
            mask_length=mask_length,
            column_token_map=column_token_map,
            mask_column_token_map=mask_column_token_map,
        )
        output_column = cfg.task.get("output_column", None)
        query_text_extractor = lambda s: prompt_template_obj.generate_text_for_embedding(
            s, output_column=output_column
        )
        lm_cfg = cfg.get("lever_lm", cfg.get("ablation", {}))
        qwen_forbidden = set()
        subset_cfg_path = cfg.get("test_subset_ids_path", None)
        if subset_cfg_path:
            try:
                with open(subset_cfg_path, "r", encoding="utf-8") as f:
                    subset_cfg = json.load(f)
                query_ids = subset_cfg.get("query_ids", [])
                forbidden_ids = subset_cfg.get("forbidden_ids", [])
                qwen_forbidden = {int(x) for x in list(query_ids) + list(forbidden_ids)}
            except Exception:
                pass
        retriever = QwenTopkRetriever(
            index_ds=train_ds,
            qwen_model_path=lm_cfg.get("qwen_model_path", "Qwen/Qwen3-Embedding-4B"),
            query_text_extractor=query_text_extractor,
            nshot=nshot,
            device=lm_cfg.get("device", "cuda:0"),
            encode_batch_size=lm_cfg.get("batch_size", 64),
            feature_cache=lm_cfg.get("embedding_path", None),
            reverse_seq=True,
            forbidden_indices=qwen_forbidden,
        )
    else:
        retriever = RandRetriever(train_ds=train_ds, nshot=nshot, seed=cfg.retriever.seed)

    interface = LLaDAInterface(
        model=model,
        tokenizer=tokenizer,
        task="mmlu",
        mask_id=cfg.infer_model.mask_id,
        mask_length=mask_length,
        split_token=cfg.task.get("split_token", "\n\n"),
        prompt_template=prompt_template,
        column_token_map=column_token_map,
        mask_column_token_map=mask_column_token_map,
    )
    metrics = MMLUMetrics()

    # 4. Load Selector and embeddings
    logger.info("Loading Selector and embeddings...")
    selector = load_selector_from_checkpoint(selector_ckpt, device)
    train_emb = torch.load(train_pt, map_location="cpu", weights_only=True)
    test_emb = torch.load(test_pt, map_location="cpu", weights_only=True)
    use_train_as_test = False
    query_id_set = None

    # Generation kwargs (from task.gen_args or infer_model)
    _raw_gen = cfg.infer_model.get("generation_kwargs", None)
    if _raw_gen is None:
        gen_kwargs = {}
    elif OmegaConf.is_config(_raw_gen):
        gen_kwargs = OmegaConf.to_container(_raw_gen, resolve=True) or {}
    else:
        gen_kwargs = dict(_raw_gen) if _raw_gen else {}
    if not gen_kwargs and cfg.task.get("gen_args"):
        ga = cfg.task.gen_args
        gen_kwargs = {
            "steps": int(ga.get("steps", 64)),
            "gen_length": int(ga.get("gen_length", 2)),
            "block_length": int(ga.get("block_length", 2)),
            "temperature": 0.0,
            "cfg_scale": 0.0,
        }

    max_samples = cfg.get("max_samples", None)
    # 固定使用本地的 500 条样本列表：generated_icd_data/cache/test_500.json
    # 这些 id 被视为「训练集里的 idx」，因此在 train_ds 中选出对应样本作为 query 集合
    path = PROJECT_ROOT / "generated_icd_data" / "cache" / "test_500.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "query_ids" in data:
            query_id_set = set(data["query_ids"])
        else:
            query_id_set = set(data)
        n_before_train = len(train_ds)
        test_ds = train_ds.filter(lambda x: x.get("idx", x.get("__index__")) in query_id_set)
        use_train_as_test = True
        n_matched = len(test_ds)
        if n_matched != len(query_id_set):
            logger.warning(
                f"test_500.json 中共有 {len(query_id_set)} 个 id，"
                f"但在当前训练集下标范围 0..{n_before_train - 1} 中只匹配到 {n_matched} 条。"
            )
        logger.info(f"Using {len(test_ds)} training samples (by idx) as queries from {path.name}")
    elif max_samples:
        test_ds = test_ds.select(range(min(max_samples, len(test_ds))))

    results = []
    for idx, test_sample in enumerate(tqdm(test_ds, desc="Ablation inference", ncols=100)):
        try:
            if use_train_as_test and query_id_set is not None:
                # 严格不重合：ICD 中不能出现任何一个 query id
                exclude_indices = list(query_id_set)
            else:
                exclude_indices = [test_sample.get("idx")] if "idx" in test_sample else []
            icd_indices = retriever.retrieve(
                test_sample=test_sample,
                exclude_indices=exclude_indices,
            )
            icd_samples = [train_ds[i] for i in icd_indices]

            test_idx = test_sample.get("idx", idx)
            if use_train_as_test:
                query_embed = train_emb[test_idx : test_idx + 1].to(device)
            else:
                query_embed = test_emb[test_idx : test_idx + 1].to(device)
            icd_embeds = train_emb[icd_indices].unsqueeze(0).to(device)
            with torch.no_grad():
                out = selector(query_embed, icd_embeds)
            pred = out["logits"].argmax(dim=-1).item()
            query_position = nshot - pred

            prompt = interface.build_prompt(ice_samples=icd_samples, test_sample=test_sample, query_position=query_position)
            prompt_tensor = interface.tokenize_prompt(prompt)
            output_tensor = interface.generate(prompt_tensor, **gen_kwargs)
            generated_text = interface.decode_output(output_tensor)
            predicted_answer = interface.extract_answer(generated_text)
            ground_truth = MMLUMetrics._normalize_truth(test_sample.get("answer", ""))
            eval_result = metrics.evaluate_single(generated_text, ground_truth)

            results.append({
                "test_idx": idx,
                "test_sample_idx": test_idx,
                "icd_indices": icd_indices,
                "query_position": query_position,
                "selector_pred": pred,
                "predicted": predicted_answer,
                "ground_truth": ground_truth,
                "is_correct": eval_result["is_correct"],
                "generated_text": generated_text,
                "prompt": prompt,
            })
        except Exception as e:
            logger.error(f"Error at test sample {idx}: {e}")
            continue

    all_generated = [r["generated_text"] for r in results]
    all_ground_truths = [r["ground_truth"] for r in results]
    batch_metrics = metrics.evaluate_batch(all_generated, all_ground_truths)
    logger.info(f"Accuracy: {batch_metrics['accuracy']:.4f}")
    logger.info(f"Correct: {batch_metrics['correct_count']}/{batch_metrics['total_count']}")

    output_dir = Path(cfg.get("output_dir", "./icl_inference_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mmlu_ablation_{retriever_type}_icd_selector_pos.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": batch_metrics, "results": results}, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
