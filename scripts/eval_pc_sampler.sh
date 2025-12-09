#我要进行全代码的评测
echo "--------------------------------"
echo "Running GSM8K task (pc_sampler)"
echo "--------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    scripts/eval.py \
    --task gsm8k \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --nshot 4 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --temperature 0.0 \
    --mode pc_sampler \
    --data_path ./data/gsm8k.jsonl \
    --result_path ./results/gsm8k_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.25 \
    --alpha 10 \

echo "--------------------------------"
echo "Running Countdown task (pc_sampler)"
echo "--------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    scripts/eval.py \
    --task countdown \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --nshot 4 \
    --steps 32 \
    --gen_length 32 \
    --block_length 32 \
    --temperature 0.0 \
    --mode pc_sampler \
    --data_path ./data/countdown.jsonl \
    --result_path ./results/countdown_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.5 \
    --alpha 10 \

echo "--------------------------------"
echo "Running Sudoku task (pc_sampler)"
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
    --mode pc_sampler \
    --data_path ./data/sudoku.csv \
    --result_path ./results/sudoku_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.25 \
    --alpha 10 \

echo "--------------------------------"
echo "Running Math500 task (pc_sampler)"
echo "--------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    scripts/eval.py \
    --task math500 \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --nshot 4 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --temperature 0.0 \
    --mode pc_sampler \
    --data_path ./data/math500.jsonl \
    --result_path ./results/math500_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.25 \
    --alpha 10 \

#mbpp现在可以一键解锁了·
echo "--------------------------------"
echo "Running MBPP task (pc_sampler)"
echo "--------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=29500 \
    scripts/eval.py \
    --task mbpp \
    --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --nshot 4 \
    --steps 128 \
    --gen_length 128 \
    --block_length 128 \
    --temperature 0.0 \
    --mode pc_sampler \
    --data_path ./data/mbpp.json \
    --result_path ./results/mbpp_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.25 \
    --alpha 10 \