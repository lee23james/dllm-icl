#!/usr/bin/env bash
# LeverLM + InsertionSelector 两分类器在 MMLU 测试集上 ICL 推理（lever_lm 必须配合第二个分类器）

set -euo pipefail

cd "$(dirname "$0")/.."

# 指定 GPU，未设置则用 0。例: GPU=1 bash scripts/run_icl_inference_mmlu_lever_lm.sh
GPU_ID=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
LEVER_DEVICE="cuda:0"

MAX_SAMPLES=${MAX_SAMPLES:-200}

# 第一个分类器 (LeverLM) ckpt：留空则用 config 里 ckpt_dir+default_cpk_key
# 例: CKPT_PATH="generated_icd_data/model_cpk/mmlu/debug/min_vl-xxx.ckpt" bash scripts/run_icl_inference_mmlu_lever_lm.sh
CKPT_PATH="${CKPT_PATH:-}"

# 训练集 embedding（与训练 LeverLM 时一致），供 LeverLM 用
# 例: EMBEDDING_PATH="generated_icd_data/cache/mmlu/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt"
EMBEDDING_PATH="${EMBEDDING_PATH:-}"

# 第二个分类器 (InsertionSelector) ckpt：留空则用 config 里 insertion_selector.ckpt_dir+default_cpk_key
# 例: INSERTION_CKPT_PATH="generated_icd_data/checkpoints/insertion_selector_mmlu/min_vl-xxx.ckpt"
INSERTION_CKPT_PATH="${INSERTION_CKPT_PATH:-}"

python icl_inference.py \
  retriever.type=lever_lm \
  insertion_selector.enabled=true \
  retriever.nshot=3 \
  test_subset_ids_path=generated_icd_data/cache/test_500.json \
  lever_lm.device="${LEVER_DEVICE}" \
  insertion_selector.device="${LEVER_DEVICE}" \
  max_samples="${MAX_SAMPLES}" \
  ${CKPT_PATH:+lever_lm.ckpt_path="${CKPT_PATH}"} \
  ${EMBEDDING_PATH:+lever_lm.embedding_path="${EMBEDDING_PATH}"} \
  ${INSERTION_CKPT_PATH:+insertion_selector.ckpt_path="${INSERTION_CKPT_PATH}"}