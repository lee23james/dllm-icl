#!/bin/bash
# 训练 Lever-LM（需先完成 stage1 生成 JSON + sampler 的 Qwen cache）
# 设备由 trainer_args.devices 控制；可加 CUDA_VISIBLE_DEVICES=0,1 指定使用哪几张卡

source /home/lzh/anaconda3/etc/profile.d/conda.sh && conda activate llada

# wandb 离线模式，无需登录；日志保存在 generated_icd_data/wandb_logs
export WANDB_MODE=offline

python train.py \
  data_files=dummy.json \
  fix_query_last_json=generated_icd_data/generated_data/mmlu-mmlu-llada-random_sampler-scorer_infoscore-construct_order_no_order-beam_size_5-few_shot_3-candidate_num_64-sample_num_5000-mc_num_1.json \
  embedding_path=generated_icd_data/cache/mmlu/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt