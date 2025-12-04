#!/bin/bash

# 使用 Accelerate 在多个GPU上运行任务
# 示例：使用2个GPU (cuda:2 和 cuda:3) 处理 mbpp 任务
# 使用 CUDA_VISIBLE_DEVICES 指定可见的 GPU，Accelerate 会将它们映射为 0, 1, 2...

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=3 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    improve/seclect_order.py \
    --task sudoku \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --nshot 2 \
    --steps 32 \
    --gen_length 32 \
    --block_length 32 \
    --temperature 0.0 \
    --data_path ./data/sudoku.csv \
    --dev_data_path ./data/sudoku_dev.csv \
    --dev_samples_num 5

