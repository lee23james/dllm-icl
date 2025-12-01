export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
echo "--------------------------------eval_sudoku_base--------------------------------"
echo "--------------------------------测试sudoku指令:构造prompt看看不同位置的情况:--------------------------------"
#sudoku一般是5shot
# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 16 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 0 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 16 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 1 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 16 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 2 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 16 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 3 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 16 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 4 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 16 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 5 \
#     --seed 1234

# echo "--------------------------------eval_conf_instruct--------------------------------"
# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 0 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 1 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 2 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 3 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 4 \
#     --seed 1234

# python scripts/eval.py --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --mode original \
#     --data_path ./data/sudoku.csv \
#     --result_path ./results/sudoku_results \
#     --query_position 5 \
#     --seed 1234

# echo "--------------------------------eval_mbpp--------------------------------"

# python scripts/eval.py \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 256 \
#     --steps 128 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --task mbpp \
#     --nshot 4 \
#     --data_path ./data/mbpp.jsonl \
#     --result_path ./results/mbpp_results \
#     --query_position 5 \
#     # --seed 1234

# python scripts/eval.py \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 256 \
#     --steps 128 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --task mbpp \
#     --nshot 6 \
#     --data_path ./data/mbpp.jsonl \
#     --result_path ./results/mbpp_results \
#     --query_position 5 \
#     # --seed 1234

# python scripts/eval.py \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Instruct \
#     --device cuda:1 \
#     --gen_length 256 \
#     --steps 128 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --mode original \
#     --task mbpp \
#     --nshot 8 \
#     --data_path ./data/mbpp.jsonl \
#     --result_path ./results/mbpp_results \
#     --query_position 5 \
    # --seed 1234

