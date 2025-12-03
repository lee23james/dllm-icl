#!/bin/bash

# 使用 Accelerate 在多个GPU上运行任务
# 示例：使用2个GPU (cuda:2 和 cuda:3) 处理 mbpp 任务

accelerate launch \
    --multi_gpu \
    --num_processes=1 \
    --gpu_ids=3 \
    --main_process_port=29500 \
    improve/seclect_order.py \
    --task mbpp \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:2 \
    --nshot 4 \
    --steps 128 \
    --gen_length 128 \
    --block_length 128 \
    --temperature 0.0 \
    --data_path ./data/mbpp_test.json \
    --dev_data_path ./data/mbpp_dev.json \
    --dev_samples_num 10

