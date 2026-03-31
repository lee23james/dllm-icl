# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 0 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 1 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 2 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 3 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 4 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 5 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku
    
# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 6 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 7 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 8 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 9 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku

# python utils/paint_conf.py \
#     --json_dir params/conf_params \
#     --sample_index 10 \
#     --task sudoku \
#     --output_dir conf_results/conf_analysis/sudoku\

python utils/paint_conf.py \
    --json_dir params/conf_params/sudoku \
    --task sudoku \
    --paint_currentconf_acc \
    --num_steps 32 \
    --num_gen_lengths 32 \
    --num_shots 3 4 6 8 \
    --output_dir conf_results/conf_analysis/sudoku \