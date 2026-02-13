#!/bin/bash
# Stage2: 使用 Stage1 生成的 ICD 序列进行 MMLU 推理和评估

python /home/lzh/llada-icl/test_icd_sequences.py \
  --task mmlu \
  --model llada \
  --sampler text_sim_qwen_mmr \
  --scorer infoscore \
  --construct_order no_order \
  --beam_size 5 \
  --few_shot 4 \
  --candidate_num 64 \
  --sample_num 200 \
  --mc_num 1 \
  --coarse_k 200 \
  --mmr_lambda 0.1 \
  --icd_rank 0 \
  --device cuda:1
