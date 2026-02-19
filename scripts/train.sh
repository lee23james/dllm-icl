#!/bin/bash
# 训练 Lever-LM（需先完成 stage1 生成 JSON + sampler 的 Qwen cache）
# 设备由 trainer_args.devices 控制，默认 1 张 GPU；可加 CUDA_VISIBLE_DEVICES=0 指定卡

source /home/lzh/anaconda3/etc/profile.d/conda.sh && conda activate llada

# wandb 离线模式，无需登录；日志保存在 generated_icd_data/wandb_logs
export WANDB_MODE=offline

python train.py \
  data_files=dummy.json \
  fix_query_last_json=generated_icd_data/generated_data/mmlu-mmlu-llada-random_sampler-scorer:infoscore-construct_order:no_order-beam_size:5-few_shot:4-candidate_num:64-sample_num:200-mc_num:1-coarse_k:200-lambda:0.1.json \
  embedding_path=generated_icd_data/cache/mmlu/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt