#!/usr/bin/env python3
"""
预计算 InsertionSelector 训练用嵌入：对题库 parquet 按行格式化为文本并用 Qwen 编码，保存为 .pt。
支持 MMLU（load_mmlu_ds + format_ceval_cmmlu_sample）。运行后训练时仅需加载 .pt，无需再加载 Qwen。
"""
import argparse
import sys
from pathlib import Path

# 项目根，保证可导入 lever_lm
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from loguru import logger

from lever_lm.insertion_selector_data import format_ceval_cmmlu_sample
from lever_lm.load_ds_utils import load_mmlu_ds
from lever_lm.utils import encode_text_qwen


def main():
    parser = argparse.ArgumentParser(description="Precompute embedding cache for InsertionSelector training.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "insertion_selector_train_mmlu.yaml"),
        help="Path to insertion_selector_train_mmlu.yaml",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Override config: parquet path (relative to project root)",
    )
    parser.add_argument(
        "--embedding_cache_path",
        type=str,
        default=None,
        help="Override config: output .pt path (relative to project root)",
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default=None,
        help="Override config: Qwen embedding model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for encoding",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encode_text_qwen",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    emb_cfg = cfg.get("embedding", {})

    train_path = args.train_path or data_cfg.get("train_path", "data/MMLU/auxiliary_train/train-00000-of-00001.parquet")
    embedding_cache_path = args.embedding_cache_path or emb_cfg.get("embedding_cache_path")
    qwen_model_path = args.qwen_model_path or emb_cfg.get("qwen_model_path")

    if not embedding_cache_path:
        raise ValueError("embedding_cache_path must be set in config or --embedding_cache_path")

    # 路径：相对则基于项目根
    train_full = Path(train_path) if Path(train_path).is_absolute() else PROJECT_ROOT / train_path
    out_full = Path(embedding_cache_path) if Path(embedding_cache_path).is_absolute() else PROJECT_ROOT / embedding_cache_path
    out_full.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading parquet: {train_full}")
    ds = load_mmlu_ds(version="local", data_path=str(train_full), split="train")
    n = len(ds)

    logger.info(f"Formatting {n} samples...")
    texts = []
    for i in range(n):
        sample = ds[i]
        text = format_ceval_cmmlu_sample(sample)
        texts.append(text)

    logger.info(f"Encoding with Qwen from {qwen_model_path}...")
    features = encode_text_qwen(
        text_list=texts,
        model_name=qwen_model_path,
        device=args.device,
        batch_size=args.batch_size,
        normalize=True,
    )

    logger.info(f"Saving embeddings shape {features.shape} to {out_full}")
    torch.save(features.cpu(), out_full)
    logger.info("Done.")


if __name__ == "__main__":
    main()
