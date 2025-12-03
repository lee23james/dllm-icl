#这里需要的是顺便添加准确率,来进行一次性的统计
#方便后续画图
# echo "--------------------------------32bits sudoku--------------------------------"
# python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 3 \
#     --paint_num 10 \
#     --samples_num 3 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

#     python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 4 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \


#     python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 5 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

#     python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 6 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

#     python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 7 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 8 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

# echo "--------------------------------32bits countdown--------------------------------"

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 3 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 4 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \


# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 5 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 6 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 7 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 8 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# echo "--------------------------------16bits sudoku--------------------------------"

# python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 3 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

#     python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 4 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \


#     python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 5 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

#     python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 6 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

#     python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 7 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/sudoku.csv \
#     --nshot 8 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/sudoku_results \
#     # --ispaint \

# echo "--------------------------------16bits countdown--------------------------------"

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 3 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 4 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \


# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 5 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 6 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 7 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:2 \
#     --gen_length 32 \
#     --steps 16 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/countdown.jsonl \
#     --nshot 8 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/countdown_results \
    # --ispaint \  

# echo "--------------------------------  测试  --------------------------------"
# python scripts/generate_conf.py \
#     --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --data_path data/gsm8k.jsonl \
#     --nshot 3 \
#     --paint_num 10 \
#     --samples_num 1\
#     --result_path ./results/gsm8k_results \
#     # --ispaint \

# # python scripts/generate_conf.py \
# #     --task math500 \
# #     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
# #     --device cuda:0 \
# #     --gen_length 128 \
# #     --steps 128 \
# #     --block_length 128 \
# #     --temperature 0.0 \
# #     --data_path data/math500.jsonl \
# #     --nshot 1 \
# #     --paint_num 10 \
# #     --samples_num 3 \
# #     --result_path ./results/math500_results \
# #     # --ispaint \


echo "--------------------------------256 bits gsm8k--------------------------------"

# python scripts/generate_conf.py \
#     --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --data_path data/gsm8k.jsonl \
#     --nshot 3 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/gsm8k_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 128 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --data_path data/gsm8k.jsonl \
#     --nshot 4 \
#     --paint_num 10 \
#     --samples_num 2 \
#     --result_path ./results/gsm8k_results \
#     --return_current_conf
#     # --ispaint \


# python scripts/generate_conf.py \
#     --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --data_path data/gsm8k.jsonl \
#     --nshot 5 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/gsm8k_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 128 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --data_path data/gsm8k.jsonl \
#     --nshot 6 \
#     --paint_num 10 \
#     --samples_num 200 \
#     --result_path ./results/gsm8k_results \
#     --return_current_conf
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --data_path data/gsm8k.jsonl \
#     --nshot 7 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/gsm8k_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task gsm8k \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 256 \
#     --steps 256 \
#     --block_length 256 \
#     --temperature 0.0 \
#     --data_path data/gsm8k.jsonl \
#     --nshot 8 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/gsm8k_results \
#     # --ispaint \

echo "--------------------------------math500--------------------------------"

python scripts/generate_conf.py \
    --task math500 \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --temperature 0.0 \
    --data_path data/math500.jsonl \
    --nshot 3 \
    --paint_num 10 \
    --samples_num 500 \
    --result_path ./results/math500_results \
    --return_current_conf
    # --ispaint \

python scripts/generate_conf.py \
    --task math500 \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 256 \
    --steps 128 \
    --block_length 256 \
    --temperature 0.0 \
    --data_path data/math500.jsonl \
    --nshot 3 \
    --paint_num 10 \
    --samples_num 500 \
    --result_path ./results/math500_results \
    --return_current_conf
    # --ispaint \

python scripts/generate_conf.py \
    --task math500 \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 256 \
    --steps 128 \
    --block_length 256 \
    --temperature 0.0 \
    --data_path data/math500.jsonl \
    --nshot 4 \
    --paint_num 10 \
    --samples_num 500 \
    --result_path ./results/math500_results \
    --return_current_conf
    # --ispaint \

python scripts/generate_conf.py \
    --task math500 \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 256 \
    --steps 128 \
    --block_length 256 \
    --temperature 0.0 \
    --data_path data/math500.jsonl \
    --nshot 6 \
    --paint_num 10 \
    --samples_num 500 \
    --result_path ./results/math500_results \
    --return_current_conf
    # --ispaint \

# python scripts/generate_conf.py \
#     --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 512 \
#     --steps 512 \
#     --block_length 512 \
#     --temperature 0.0 \
#     --data_path data/math500.jsonl \
#     --nshot 5 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/math500_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 512 \
#     --steps 512 \
#     --block_length 512 \
#     --temperature 0.0 \
#     --data_path data/math500.jsonl \
#     --nshot 6 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/math500_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 512 \
#     --steps 512 \
#     --block_length 512 \
#     --temperature 0.0 \
#     --data_path data/math500.jsonl \
#     --nshot 7 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/math500_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 512 \
#     --steps 512 \
#     --block_length 512 \
#     --temperature 0.0 \
#     --data_path data/math500.jsonl \
#     --nshot 8 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/math500_results \
#     # --ispaint \

# python scripts/generate_conf.py \
#     --task math500 \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:1 \
#     --gen_length 32 \
#     --steps 32 \
#     --block_length 32 \
#     --temperature 0.0 \
#     --data_path data/math500.jsonl \
#     --nshot 3 \
#     --paint_num 10 \
#     --samples_num 2 \
#     --result_path ./results/math500_results \
#     # --ispaint \

#常识
echo "--------------------------------eval mbpp--------------------------------"
# python scripts/generate_conf.py \
#     --task mbpp \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --data_path data/mbpp.json \
#     --nshot 3 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/mbpp_results \
#     --return_current_conf
#     # --ispaint \

# python utils/judge_python_code.py \
#     --folder_path results/mbpp_results \
#     --output_path params/conf_params/mbpp/accuracy \
#     --return_json \
#     --nshot 3 \
#     --steps 128 \
#     --gen_length 128

# python scripts/generate_conf.py \
#     --task mbpp \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --data_path data/mbpp.json \
#     --nshot 4 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/mbpp_results \
#     --return_current_conf
#     # --ispaint \

# python utils/judge_python_code.py \
#     --folder_path results/mbpp_results \
#     --output_path params/conf_params/mbpp/accuracy \
#     --return_json \
#     --nshot 4 \
#     --steps 128 \
#     --gen_length 128

python scripts/generate_conf.py \
    --task mbpp \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --temperature 0.0 \
    --data_path data/mbpp.json \
    --nshot 6 \
    --paint_num 10 \
    --samples_num 500 \
    --result_path ./results/mbpp_results \
    --return_current_conf
    # --ispaint \

python utils/judge_python_code.py \
    --folder_path results/mbpp_results \
    --output_path params/conf_params/mbpp/accuracy \
    --return_json \
    --nshot 6 \
    --steps 128 \
    --gen_length 128

python scripts/generate_conf.py \
    --task mbpp \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --temperature 0.0 \
    --data_path data/mbpp.json \
    --nshot 8 \
    --paint_num 10 \
    --samples_num 500 \
    --result_path ./results/mbpp_results \
    --return_current_conf
    # --ispaint \

python utils/judge_python_code.py \
    --folder_path results/mbpp_results \
    --output_path params/conf_params/mbpp/accuracy \
    --return_json \
    --nshot 8 \
    --steps 128 \
    --gen_length 128


# python scripts/generate_conf.py \
#     --task mbpp \
#     --model_name /seu_share/home/yangxu/213233851/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --data_path data/mbpp.json \
#     --nshot 6 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/mbpp_results \
#     --return_current_conf
#     # --ispaint \

# python utils/judge_python_code.py \
#     --folder_path results/mbpp_results \
#     --output_path params/conf_params/mbpp/accuracy \
#     --return_json \
#     --nshot 6 \
#     --steps 128 \
#     --gen_length 128

# python scripts/generate_conf.py \
#     --task mbpp \
#     --model_name /seu_share/home/yangxu/213233851/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --data_path data/mbpp.json \
#     --nshot 8 \
#     --paint_num 10 \
#     --samples_num 500 \
#     --result_path ./results/mbpp_results \
#     --return_current_conf
#     # --ispaint \

# python utils/judge_python_code.py \
#     --folder_path results/mbpp_results \
#     --output_path params/conf_params/mbpp/accuracy \
#     --return_json \
#     --nshot 8 \
#     --steps 128 \
#     --gen_length 128
