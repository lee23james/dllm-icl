#!/bin/bash

# 使用 Accelerate 在多个GPU上运行 upperbound 评估
# 示例：使用4个GPU (cuda:0,1,2,3) 处理不同任务
# 使用 CUDA_VISIBLE_DEVICES 指定可见的 GPU，Accelerate 会将它们映射为 0, 1, 2...
#这里我想的是计算的是每一个位置的upperbound
echo "========================================"
echo "Running Upperbound Evaluation"
echo "========================================"

# ========================================
# Sudoku Task
# ========================================
echo "--------------------------------"
echo "Running Sudoku task (upperbound)"
echo "--------------------------------"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    scripts/eval.py \
    --task sudoku \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --nshot 4 \
    --steps 32 \
    --gen_length 32 \
    --block_length 32 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/sudoku.csv \
    --result_path ./results/sudoku_results \
    --max_samples 4 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --upperbound

# ========================================
# Countdown Task
# ========================================
echo "--------------------------------"
echo "Running Countdown task (upperbound)"
echo "--------------------------------"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29501 \
    scripts/eval.py \
    --task countdown \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --nshot 4 \
    --steps 32 \
    --gen_length 32 \
    --block_length 32 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/countdown.jsonl \
    --result_path ./results/countdown_results \
    --max_samples 4 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --upperbound

# ========================================
# Math500 Task
# ========================================
echo "--------------------------------"
echo "Running Math500 task (upperbound)"
echo "--------------------------------"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29502 \
    scripts/eval.py \
    --task math500 \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --nshot 1 \
    --steps 128 \
    --gen_length 128 \
    --block_length 128 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/math500.jsonl \
    --result_path ./results/math500_results \
    --max_samples 4 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --upperbound

# ========================================
# MBPP Task
# ========================================
#进行测试,看看能不能直接解决
# echo "--------------------------------"
# echo "Running MBPP task (upperbound)"
# echo "--------------------------------"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --num_machines=1 \
#     --mixed_precision=no \
#     --main_process_port=29503 \
#     scripts/eval.py \
#     --task mbpp \
#     --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --nshot 4 \
#     --steps 128 \
#     --gen_length 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/mbpp.json \
#     --result_path ./results/mbpp_results \
#     --max_samples 4 \
#     --seed 1234 \
#     --iscot False \
#     --thread 0.9 \
#     --upperbound

# ========================================
# GSM8K Task
# ========================================
echo "--------------------------------"
echo "Running GSM8K task (upperbound)"
echo "--------------------------------"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29504 \
    scripts/eval.py \
    --task gsm8k \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --nshot 1 \
    --steps 128 \
    --gen_length 128 \
    --block_length 128 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/gsm8k.jsonl \
    --result_path ./results/gsm8k_results \
    --max_samples 4 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --upperbound

# echo "========================================"
# echo "All upperbound evaluations completed!"
# echo "========================================"

