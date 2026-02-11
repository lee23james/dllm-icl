#测试生成baseline并进行记录
python3 test_baseline_icd_sequences.py \
    --task mmlu \
    --model llada \
    --sampler text_sim_qwen_mmr \
    --scorer infoscore \
    --construct_order no_order \
    --beam_size 5 \
    --few_shot 4 \
    --candidate_num 64 \
    --sample_num 100 \
    --icd_rank 0 \
    --device cuda:1 \
    --mask_length 2 \
    --block_length 2 \
    --gen_length 2 \
    --steps 2 \
    --mc_num 1 \
    --coarse_k 200 \
    --mmr_lambda 0.1