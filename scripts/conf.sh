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
#     --output_dir conf_results/conf_analysis/sudoku

echo "--------------------------------paint countdown conf accuracy--------------------------------"
python utils/paint_conf.py \
    --json_dir params/conf_params \
    --task countdown \
    --output_dir conf_results/conf_analysis/countdown \
    --paint_conf_acc

echo "--------------------------------paint sudoku conf accuracy--------------------------------"
python utils/paint_conf.py \
    --json_dir params/conf_params \
    --task sudoku \
    --output_dir conf_results/conf_analysis/sudoku \
    --paint_conf_acc