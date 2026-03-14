# Random Position Baseline Experiment
# 随机位置选择的 baseline 实验：每个样本随机选择一个 query 插入位置（seed=42）
# 使用 Accelerate 在多个GPU上运行
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# 配置：可以根据实际情况修改使用的 GPU
# 例如：使用 GPU 2,3 或 0,1,2,3
GPU_LIST="2,3"  # 可以改为 "0,1,2,3" 使用所有 GPU
NUM_GPUS=2      # 对应 GPU 数量

echo "========================================"
echo "Random Position Baseline Evaluation (Multi-GPU)"
echo "Using GPUs: $GPU_LIST"
echo "========================================"

echo "--------------------------------Random Position Baseline: Sudoku--------------------------------"
CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    scripts/eval.py \
    --task sudoku \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 32 \
    --steps 32 \
    --block_length 32 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/sudoku.csv \
    --result_path ./results/sudoku_results_random \
    --nshot 4 \
    --max_samples 2 \
    --random_position \
    --seed 42

echo "--------------------------------Random Position Baseline: Countdown--------------------------------"
CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29501 \
    scripts/eval.py \
    --task countdown \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 32 \
    --steps 32 \
    --block_length 32 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/countdown.jsonl \
    --result_path ./results/countdown_results_random \
    --nshot 4 \
    --max_samples 2 \
    --random_position \
    --seed 42

echo "--------------------------------Random Position Baseline: GSM8K--------------------------------"
CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29502 \
    scripts/eval.py \
    --task gsm8k \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/gsm8k.jsonl \
    --result_path ./results/gsm8k_results_random \
    --nshot 4 \
    --max_samples 2 \
    --random_position \
    --seed 42

echo "--------------------------------Random Position Baseline: MBPP--------------------------------"
CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29504 \
    scripts/eval.py \
    --task mbpp \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/mbpp.json \
    --result_path ./results/mbpp_results_random \
    --nshot 4 \
    --max_samples 2 \
    --random_position \
    --seed 42

echo "--------------------------------Random Position Baseline: Math500--------------------------------"
CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29503 \
    scripts/eval.py \
    --task math500 \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --temperature 0.0 \
    --mode original \
    --data_path ./data/math500.jsonl \
    --result_path ./results/math500_results_random \
    --nshot 4 \
    --max_samples 2 \
    --random_position \
    --seed 42
echo "========================================"
echo "All random position baseline evaluations completed!"
echo "========================================"
