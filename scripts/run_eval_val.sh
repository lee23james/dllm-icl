#!/bin/bash
# 在验证集上评估训练好的 LeverLM（min_vl checkpoint）

# 自动查找 min_vl checkpoint，或手动指定
CKPT_DIR="generated_icd_data/model_cpk/mmlu/debug"
CKPT="${1:-}"
if [ -z "$CKPT" ]; then
    CKPT=$(ls -t "${CKPT_DIR}"/min_vl-*.ckpt 2>/dev/null | head -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "用法: $0 [ckpt_path]"
    echo "未找到 min_vl checkpoint，请指定: $0 generated_icd_data/model_cpk/mmlu/debug/min_vl-epoch=0-xxx.ckpt"
    exit 1
fi

python eval_lever_lm_val.py \
  fix_query_last_json=generated_icd_data/generated_data/mmlu-mmlu-llada-random_sampler-scorer_infoscore-construct_order_no_order-beam_size_5-few_shot_3-candidate_num_64-sample_num_5000-mc_num_1.json \
  embedding_path=generated_icd_data/cache/mmlu/mmlu-mmlu-Qwen3-Embedding-4B-TextFeatures.pt \
  ckpt_path="${CKPT}" \
  device=cuda:0 \
  eval_batch_size=32
