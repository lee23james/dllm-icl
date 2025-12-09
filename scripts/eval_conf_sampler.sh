#测试一下自己的代码,看看能不能用conf_sampler进行测试
#这里默认num=0
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
    --mode conf_sampler \
    --data_path ./data/sudoku.csv \
    --result_path ./results/sudoku_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.0 \
    --alpha 10 \

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
    --mode conf_sampler \
    --data_path ./data/gsm8k.jsonl \
    --result_path ./results/gsm8k_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.25 \
    --alpha 10 \

#这里测试的是mbpp(之前的mbpp需要改一下,最好的方法就是一口气就可以进行评测)
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --num_machines=1 \
#     --mixed_precision=no \
#     --main_process_port=29500 \
#     scripts/eval.py \
#     --task mbpp \
#     --model_name /data/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --nshot 4 \
#     --steps 128 \
#     --gen_length 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode conf_sampler \
#     --data_path ./data/mbpp.json \
#     --result_path ./results/mbpp_results \
#     --max_samples 2 \
#     --seed 1234 \
#     --iscot False \
#     --thread 0.9 \
#     --lambd 0.25 \
#     --alpha 10 \

#适配了mbpp
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
    --mode conf_sampler \
    --data_path ./data/mbpp.json \
    --result_path ./results/mbpp_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.25 \
    --alpha 10 \

#这里要对countdown也进行评测
#进行对比实验
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
    --steps 32 \
    --gen_length 32 \
    --block_length 32 \
    --temperature 0.0 \
    --mode conf_sampler \
    --data_path ./data/gsm8k.jsonl \
    --result_path ./results/gsm8k_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.25 \
    --alpha 10 \

