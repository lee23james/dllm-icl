# python scripts/generate_likelihood.py \
#     --task sudoku \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --data_path data/sudoku.csv \
#     --mc_num 128 \
#     --batch_size 16 \
#     --cfg_scale 0. \
#     --nshot 5 \
#     --samples_num 288 \

python scripts/generate_likelihood.py \
    --task countdown \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:1 \
    --data_path data/countdown.jsonl \
    --mc_num 128 \
    --batch_size 16 \
    --cfg_scale 0. \
    --nshot 3 \
    --samples_num  2\