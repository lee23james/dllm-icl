# cd "$(dirname "$0")" && python generate_heatmap.py \
#     --task countdown \
#     --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
#     --device cuda:0 \
#     --gen_length 128 \
#     --steps 128 \
#     --block_length 128 \
#     --temperature 0.0 \
#     --data_path ../data/countdown.jsonl \
#     --samples_num 200 \
#     --nshot 3

    # python scripts/generate_heatmap.py \
    # --task sudoku \
    # --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    # --device cuda:1 \
    # --gen_length 32 \
    # --steps 32 \
    # --block_length 32 \
    # --temperature 0.0 \
    # --data_path data/sudoku.csv \
    # --samples_num 50 \
    # --nshot 5
echo "--------------------------------paint heatmap for countdown--------------------------------"
# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_0_avg.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_1_avg.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_2_avg.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_3_avg.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_0_single_0.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_0_single_50.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_0_single_100.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_0_single_150.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_1_single_0.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_1_single_50.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_1_single_100.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_1_single_150.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_2_single_0.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_2_single_50.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_2_single_100.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_2_single_150.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_3_single_0.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_3_single_50.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_3_single_100.json \
#     --task countdown

# python utils/paint_heatmap.py \
#     --data_path params/heatmap_params/countdown_3_single_150.json \
#     --task countdown

echo "--------------------------------paint heatmap for sudoku--------------------------------"

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_0_avg.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_1_avg.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_2_avg.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_3_avg.json \
    --task sudoku  

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_4_avg.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_5_avg.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_0_single_0.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_0_single_1.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_0_single_2.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_0_single_3.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_1_single_0.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_1_single_1.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_1_single_2.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_1_single_3.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_2_single_0.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_2_single_1.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_2_single_2.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_2_single_3.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_3_single_0.json \
    --task sudoku   

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_3_single_1.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_3_single_2.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_3_single_3.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_4_single_0.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_4_single_1.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_4_single_2.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_4_single_3.json \
    --task sudoku
python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_5_single_0.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_5_single_1.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_5_single_2.json \
    --task sudoku

python utils/paint_heatmap.py \
    --data_path params/heatmap_params/sudoku_5_single_3.json \
    --task sudoku

