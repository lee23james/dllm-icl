echo "--------------------------------generate rollout for sudoku--------------------------------"
    python scripts/generate_rollout.py \
    --task sudoku \
    --model_name /home/share/model_weight/llada/LLaDA-8B-Base \
    --device cuda:0 \
    --gen_length 32 \
    --steps 32 \
    --block_length 32 \
    --temperature 0.0 \
    --data_path data/sudoku.csv \
    --samples_num 50 \
    --nshot 5 \
    --step_dis 1

#第一个数字是位置,第二个数字是data的index
#现在开始进行绘图,我想的是先把所有的json文件绘制出来

# python utils/paint_rollout.py \
#     --json_file params/rollout_params/sudoku_0_0.json \
#     --output_dir rollout_results/heatmap \
#     --basename sudoku_0_0

# python utils/paint_rollout.py \
#     --json_file params/rollout_params/sudoku_1_0.json \
#     --output_dir rollout_results/heatmap \
#     --basename sudoku_1_0

# python utils/paint_rollout.py \
#     --json_file params/rollout_params/sudoku_2_0.json \
#     --output_dir rollout_results/heatmap \
#     --basename sudoku_2_0

# python utils/paint_rollout.py \
#     --json_file params/rollout_params/sudoku_3_0.json \
#     --output_dir rollout_results/heatmap \
#     --basename sudoku_3_0

# python utils/paint_rollout.py \
#     --json_file params/rollout_params/sudoku_4_0.json \
#     --output_dir rollout_results/heatmap \
#     --basename sudoku_4_0

# python utils/paint_rollout.py \
#     --json_file params/rollout_params/sudoku_5_0.json \
#     --output_dir rollout_results/heatmap \
#     --basename sudoku_5_0

python utils/paint_rollout.py \
    --layer_analysis True \
    --heatmap_generate False \
    --max_samples 50 \
    --nshot 5 \
    --task sudoku