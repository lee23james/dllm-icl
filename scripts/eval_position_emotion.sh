export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

echo "=========================================="
echo "开始情感分析任务评估 - 测试不同query位置的影响"
echo "=========================================="

# 设置默认参数
MODEL_NAME="/home/share/model_weight/llada/LLaDA-8B-Base"
DEVICE="cuda:1"
SEED=1234
TEMPERATURE=0.0

echo ""
echo "--------------------------------测试SST2任务:情感分析(2分类)--------------------------------"
# SST2: 4-shot
# python scripts/llada_main.py \
#     --config config/sst2.yaml \
#     --model ${MODEL_NAME} \
#     --device ${DEVICE} \
#     --seed ${SEED} \
#     --temperature ${TEMPERATURE} \
#     --output ./results/sst2_results \
#     --train_sample_mode balance \
#     --test_data_path ./data/sst2/dev_subsample.jsonl \
#     --nshot 2 \
#     --gen_length 1 \
#     --steps 1 \
#     --block_length 1 \
#     --task sst2

echo ""
echo "--------------------------------测试AGNews任务:新闻分类(4分类)--------------------------------"
# # AGNews: 4-shot
python scripts/llada_main.py \
    --config config/agnews.yaml \
    --model ${MODEL_NAME} \
    --device ${DEVICE} \
    --seed ${SEED} \
    --temperature ${TEMPERATURE} \
    --output ./results/agnews_results \
    --train_sample_mode balance \
    --test_data_path ./data/agnews/dev_subsample.jsonl \
    --nshot 2 \
    --gen_length 1 \
    --steps 1 \
    --block_length 1 \
    --task agnews

echo ""
echo "--------------------------------测试TREC任务:问题分类(6分类)--------------------------------"
# ; # TREC: 6-shot(因为是按标签随机取样,虽哦咦nshot=1->实际上有6个例子)
# python scripts/llada_main.py \
#     --config config/trec.yaml \
#     --model ${MODEL_NAME} \
#     --device ${DEVICE} \
#     --seed ${SEED} \
#     --temperature ${TEMPERATURE} \
#     --output ./results/trec_results \
#     --train_sample_mode balance \
#     --test_data_path ./data/trec/dev_subsample.jsonl \
#     --nshot 2 \
#     --gen_length 1 \
#     --steps 1 \
#     --block_length 1 \
#     --task trec

echo ""
echo "--------------------------------测试SUBJ任务:主客观分类(2分类)--------------------------------"
# # SUBJ: 4-shot
# python scripts/llada_main.py \
#     --config config/subj.yaml \
#     --model ${MODEL_NAME} \
#     --device ${DEVICE} \
#     --seed ${SEED} \
#     --temperature ${TEMPERATURE} \
#     --output ./results/subj_results \
#     --train_sample_mode balance \
#     --test_data_path ./data/subj/dev_subsample.jsonl \
#     --nshot 2 \
#     --gen_length 1 \
#     --steps 1 \
#     --block_length 1 \
#     --task subj

echo ""
echo "=========================================="
echo "所有情感分析任务评估完成！"
echo "=========================================="
