#!/usr/bin/env bash
# 使用训练好的 LeverLM 在 MMLU 测试集上进行 ICL 推理

set -euo pipefail

# Force everything (LLaDA + LeverLM embedding) onto the same physical GPU.
# Usage:
#   GPU=1 bash scripts/run_icl_inference_mmlu_lever_lm.sh
# If GPU is not set, default to 0.
GPU_ID=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# NOTE:
# After setting CUDA_VISIBLE_DEVICES, the selected physical GPU becomes "cuda:0" inside this process.
LEVER_DEVICE="cuda:0"

# Optional: limit samples for quick smoke test
MAX_SAMPLES=${MAX_SAMPLES:-20}



python icl_inference.py \
  retriever.type=lever_lm \
  retriever.nshot=3 \
  lever_lm.device=${LEVER_DEVICE} \
  max_samples=${MAX_SAMPLES}