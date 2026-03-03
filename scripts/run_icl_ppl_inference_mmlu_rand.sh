#!/usr/bin/env bash
# MMLU 多项选择题：rand retriever + PPL 推理（不启用 LeverLM / InsertionSelector）

set -euo pipefail

cd "$(dirname "$0")/.."

# 指定 GPU，未设置则用 0。例: GPU=1 bash scripts/run_icl_ppl_inference_mmlu_rand.sh
GPU_ID=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 测试样本上限，默认 200，便于快速验证
MAX_SAMPLES=${MAX_SAMPLES:-200}

# few-shot 个数（ICD 数量），默认 4（MMLU 常用 4-shot）
NSHOT=${NSHOT:-4}

python icl_ppl_inference.py \
  task=mmlu \
  dataset=mmlu \
  retriever.type=rand \
  insertion_selector.enabled=false \
  retriever.nshot="${NSHOT}" \
  max_samples="${MAX_SAMPLES}"


