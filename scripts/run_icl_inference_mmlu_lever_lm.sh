#!/bin/bash
# 使用训练好的 LeverLM 在 MMLU 测试集上进行 ICL 推理

python icl_inference.py \
  retriever.type=lever_lm \
  retriever.nshot=4 \
  lever_lm.ckpt_dir=generated_icd_data/model_cpk/mmlu/debug \
  lever_lm.default_cpk_key=min_vl \
  lever_lm.embedding_path=generated_icd_data/cache/mmlu/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt \
  lever_lm.qwen_model_path=/home/lzh/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-4B \
  lever_lm.device=cuda:0 \
  output_dir=./icl_inference_results
# 快速测试可加: max_samples=100
