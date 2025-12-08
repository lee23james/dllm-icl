#!/bin/bash

# 使用 Accelerate 在多个GPU上运行 generate_improve.py
# 示例：使用4个GPU (cuda:0,1,2,3) 处理不同任务
# 使用 CUDA_VISIBLE_DEVICES 指定可见的 GPU，Accelerate 会将它们映射为 0, 1, 2...
#适合gpu实现
#可以先测试一下sudoku和mbpp多,这两个没问题其他基本就没什么问腿
echo "========================================"
echo "Running Generate Improve (Multi-GPU)"
echo "========================================"

# ========================================
# Sudoku Task
# ========================================
# echo "--------------------------------"
# echo "Running Sudoku task"
# echo "--------------------------------"

CUDA_VISIBLE_DEVICES=3 accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    improve/generate_improve.py \
    --task sudoku \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --version answer_token \
    --nshot 4 \
    --steps 32 \
    --gen_length 32 \
    --block_length 32 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/sudoku.csv \
    --result_path ./results/sudoku_results \
    --samples_num 4 \
    --lamda1 1.0 \
    --lamda2 0.0

# ========================================
# Countdown Task
# ========================================
echo "--------------------------------"
echo "Running Countdown task"
echo "--------------------------------"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --num_machines=1 \
#     --mixed_precision=no \
#     --main_process_port=29501 \
#     improve/generate_improve.py \
#     --task countdown \
#     --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --version answer_token \
#     --nshot 4 \
#     --steps 32 \
#     --gen_length 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/countdown.jsonl \
#     --result_path ./results/countdown_results \
#     --samples_num 4 \
#     --lamda1 1 \
#     --lamda2 0

# # ========================================
# # Math500 Task
# # ========================================
echo "--------------------------------"
echo "Running Math500 task"
echo "--------------------------------"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --num_machines=1 \
#     --mixed_precision=no \
#     --main_process_port=29502 \
#     improve/generate_improve.py \
#     --task math500 \
#     --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --version answer_token \
#     --nshot 4 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/math500.jsonl \
#     --result_path ./results/math500_results \
#     --samples_num 10 \
#     --lamda1 1 \
#     --lamda2 0

# ========================================
# MBPP Task
# ========================================
# echo "--------------------------------"
# echo "Running MBPP task"
# echo "--------------------------------"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --num_machines=1 \
#     --mixed_precision=no \
#     --main_process_port=29503 \
#     improve/generate_improve.py \
#     --task mbpp \
#     --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --version answer_token \
#     --nshot 4 \
#     --steps 128 \
#     --gen_length 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/mbpp.json \
#     --result_path ./results/mbpp_results \
#     --samples_num 10 \
#     --lamda1 1.0 \
#     --lamda2 0.0

# python utils/judge_python_code.py \
#     --folder_path results/mbpp_results \
#     --output_path params/conf_params/mbpp/accuracy \
#     --nshot 4 \
#     --steps 128 \
#     --gen_length 128 \
#     --find_not_position
# ========================================
# GSM8K Task
# ========================================
echo "--------------------------------"
echo "Running GSM8K task"
echo "--------------------------------"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --num_machines=1 \
#     --mixed_precision=no \
#     --main_process_port=29504 \
#     improve/generate_improve.py \
#     --task gsm8k \
#     --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --version answer_token \
#     --nshot 4 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/gsm8k.jsonl \
#     --result_path ./results/gsm8k_results \
#     --samples_num 4 \
#     --lamda1 1.0 \
#     --lamda2 0.0

# echo "========================================"
# echo "All tasks completed!"
# echo "========================================"

