#目前只是短暂的测试,看看能不能把position一起测出来
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

echo "--------------------------------测试gsm8k指令:构造prompt看看不同位置的情况:--------------------------------"
# python eval.py --task gsm8k --model_name GSAI-ML/LLaDA-8B-Base --device cuda:0 --gen_length 256 --steps 256 --block_length 256 --temperature 0.0 --mode original --data_path ./data/gsm8k.json --result_path ./results/gsm8k_results_position_0 --query_position 0
# python eval.py --task gsm8k --model_name GSAI-ML/LLaDA-8B-Base --device cuda:0 --gen_length 256 --steps 256 --block_length 256 --temperature 0.0 --mode original --data_path ./data/gsm8k.json --result_path ./results/gsm8k_results_position_1 --query_position 1
# python eval.py --task gsm8k --model_name GSAI-ML/LLaDA-8B-Base --device cuda:0 --gen_length 256 --steps 256 --block_length 256 --temperature 0.0 --mode original --data_path ./data/gsm8k.json --result_path ./results/gsm8k_results_position_2 --query_position 2
# python eval.py --task gsm8k --model_name GSAI-ML/LLaDA-8B-Base --device cuda:0 --gen_length 256 --steps 256 --block_length 256 --temperature 0.0 --mode original --data_path ./data/gsm8k.json --result_path ./results/gsm8k_results_position_3 --query_position 3
# python eval.py --task gsm8k --model_name GSAI-ML/LLaDA-8B-Base --device cuda:0 --gen_length 256 --steps 256 --block_length 256 --temperature 0.0 --mode original --data_path ./data/gsm8k.json --result_path ./results/gsm8k_results_position_4 --query_position 4
#一般是4shot
# python scripts/eval.py --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 128 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/gsm8k.jsonl \
#     --result_path ./results/gsm8k_results \
#     --nshot 4 \
#     --max_samples 200

# python scripts/eval.py --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 128 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/gsm8k.jsonl \
#     --result_path ./results/gsm8k_results \
#     --nshot 6 \
#     --max_samples 200

# python scripts/eval.py --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 128 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/gsm8k.jsonl \
#     --result_path ./results/gsm8k_results \
#     --nshot 8 \
#     --max_samples 200
echo "--------------------------------eval_conf_base:--------------------------------"
# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --nshot 4 \
#     --max_samples 2

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --nshot 6 \
#     --max_samples 2

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --nshot 8 \
#     --max_samples 200



# echo "--------------------------------测试sudoku指令:构造cot prompt看看不同位置的情况:--------------------------------"
# #sudoku一般是5shot
# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 0 \
#     --seed 1234 \

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 1 \
#     --seed 1234 \


# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 2 \
#     --seed 1234 \

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 3 \
#     --seed 1234 \

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 4 \
#     --seed 1234 \

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 5 \
#     --seed 1234 \

echo "--------------------------------测试256 token:--------------------------------"

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 0 \
#     --seed 1234 \
#     --iscot True

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 1 \
#     --seed 1234 \
#     --iscot True


# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 2 \
#     --seed 1234 \
#     --iscot True

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 3 \
#     --seed 1234 \
#     --iscot True

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 4 \
#     --seed 1234 \
#     --iscot True

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 5 \
#     --seed 1234 \
#     --iscot True


echo "--------------------------------测试countdown指令:构造prompt看看不同位置的情况:--------------------------------"
#countdown一般是3shot
# python scripts/eval.py --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/countdown.jsonl \
#     --result_path ./results/countdown_results \
#     --query_position 0 \
#     --max_samples 500

# python scripts/eval.py --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/countdown.jsonl \
#     --result_path ./results/countdown_results \
#     --query_position 1 \
#     --max_samples 500


# python scripts/eval.py --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/countdown.jsonl \
#     --result_path ./results/countdown_results \
#     --query_position 2 \
#     --max_samples 500


# python scripts/eval.py --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/countdown.jsonl \
#     --result_path ./results/countdown_results \
#     --query_position 3 \
#     --max_samples 500

#现在主要就是生成中方法进行评测和探究,目前不再需要位置了,所有位置都要进行探究
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
    --mode original \
    --data_path ./data/countdown.jsonl \
    --result_path ./results/countdown_results \
    --max_samples 2 \
    --seed 1234 \
    --iscot False \
    --thread 0.9 \
    --lambd 0.25 \
    --alpha 10 \




echo "--------------------------------测试math500指令:构造prompt看看不同位置的情况:--------------------------------"
#math500一般是4shot
# python scripts/eval.py --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/math500.jsonl \
#     --result_path ./results/math500_results \
#     --query_position 0

# python scripts/eval.py --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/math500.jsonl \
#     --result_path ./results/math500_results \
#     --query_position 1

# python scripts/eval.py --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/math500.jsonl \
#     --result_path ./results/math500_results \
#     --query_position 2 

#     python scripts/eval.py --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/math500.jsonl \
#     --result_path ./results/math500_results \
#     --query_position 3 

#     python scripts/eval.py --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/math500.jsonl \
#     --result_path ./results/math500_results \
#     --query_position 4

echo "--------------------------------测试mbpp数据集-------------------------------"
#mbpp一般是4shot
# python scripts/eval.py --task mbpp \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/mbpp.json \
#     --result_path ./results/mbpp_results_position_0 \
#     --query_position 0  \
#     --max_samples 500

# 这个评估代码
# python utils/judge_python_code.py\
#     --folder_path results/mbpp_results_position_0 \
#     --output_path results/mbpp_results_position_0.txt

# python scripts/eval.py --task mbpp \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/mbpp.json \
#     --result_path ./results/mbpp_results_position_1 \
#     --query_position 1 \
#     --max_samples 500

# python utils/judge_python_code.py\
#     --folder_path results/mbpp_results_position_1 \
#     --output_path results/mbpp_results_position_1.txt

# python scripts/eval.py --task mbpp \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/mbpp.json \
#     --result_path ./results/mbpp_results_position_2 \
#     --query_position 2 \
#     --max_samples 500

# python utils/judge_python_code.py\
#     --folder_path results/mbpp_results_position_2 \
#     --output_path results/mbpp_results_position_2.txt

# python scripts/eval.py --task mbpp \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/mbpp.json \
#     --result_path ./results/mbpp_results_position_3 \
#     --query_position 3 \
#     --max_samples 500

# python utils/judge_python_code.py\
#     --folder_path results/mbpp_results_position_3 \
#     --output_path results/mbpp_results_position_3.txt
    
# python scripts/eval.py --task mbpp \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/mbpp.json \
#     --result_path ./results/mbpp_results_position_4 \
#     --query_position 4 \
#     --max_samples 500

# python utils/judge_python_code.py\
#     --folder_path results/mbpp_results_position_4 \
#     --output_path results/mbpp_results_position_4.txt

echo "--------------------------------测试gpqa指令:构造prompt看看不同位置的情况:--------------------------------"
#mbpp一般是5shot
# python scripts/eval.py --task gpqa \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/gpqa.jsonl \
#     --result_path ./results/gpqa_results \
#     --query_position 0 \
#     --nshot 5 