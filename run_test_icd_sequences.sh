#!/bin/bash
# 运行ICD序列测试脚本
# 使用方法: bash run_test_icd_sequences.sh

# 设置默认参数
TASK="gsm8k"
MODEL="llada"
SAMPLER="random"
SCORER="infoscore"
CONSTRUCT_ORDER="no_order"
BEAM_SIZE=3
FEW_SHOT=4
CANDIDATE_NUM=10
SAMPLE_NUM=10
ICD_RANK=0  # 选择第几名score的ICD序列：0=最高分；2=第三名（你现在要测的）

# 评测参数（可选，会从config读取默认值）
# 注意：以下参数如果不填写（保持注释状态），会自动从 configs/generate_data.yaml 读取
# configs/generate_data.yaml 会引用 configs/infer_model/llada.yaml 中的配置
# 如果填写了，会覆盖config中的值
DEVICE="cuda:1"
# MODEL_PATH=""  # 默认从 configs/infer_model/llada.yaml 的 model_path 读取
# MASK_LENGTH=""  # 默认从 configs/infer_model/llada.yaml 的 mask_length 读取（默认256）
# MASK_ID=""  # 默认从 configs/infer_model/llada.yaml 的 mask_id 读取（默认126336）
# BLOCK_LENGTH=""  # 默认从 configs/infer_model/llada.yaml 的 generation_kwargs.block_length 读取（默认128）
# GEN_LENGTH=""  # 默认从 configs/infer_model/llada.yaml 的 generation_kwargs.gen_length 读取（默认128）
# STEPS=""  # 默认从 configs/infer_model/llada.yaml 的 generation_kwargs.steps 读取（默认128）
# TEMPERATURE=""  # 默认从 configs/infer_model/llada.yaml 的 generation_kwargs.temperature 读取（默认0.0）
MODE="original"

# 构建命令
CMD="python3 test_icd_sequences.py \
    --task ${TASK} \
    --model ${MODEL} \
    --sampler ${SAMPLER} \
    --scorer ${SCORER} \
    --construct_order ${CONSTRUCT_ORDER} \
    --beam_size ${BEAM_SIZE} \
    --few_shot ${FEW_SHOT} \
    --candidate_num ${CANDIDATE_NUM} \
    --sample_num ${SAMPLE_NUM} \
    --icd_rank ${ICD_RANK} \
    --device ${DEVICE} \
    --mode ${MODE}"

# 添加可选参数（如果设置了）
if [ ! -z "$MODEL_PATH" ]; then
    CMD="$CMD --model_path $MODEL_PATH"
fi

if [ ! -z "$MASK_LENGTH" ]; then
    CMD="$CMD --mask_length $MASK_LENGTH"
fi

if [ ! -z "$MASK_ID" ]; then
    CMD="$CMD --mask_id $MASK_ID"
fi

if [ ! -z "$BLOCK_LENGTH" ]; then
    CMD="$CMD --block_length $BLOCK_LENGTH"
fi

if [ ! -z "$GEN_LENGTH" ]; then
    CMD="$CMD --gen_length $GEN_LENGTH"
fi

if [ ! -z "$STEPS" ]; then
    CMD="$CMD --steps $STEPS"
fi

if [ ! -z "$TEMPERATURE" ]; then
    CMD="$CMD --temperature $TEMPERATURE"
fi

# 打印命令
echo "=========================================="
echo "运行ICD序列测试脚本"
echo "=========================================="
echo "命令: $CMD"
echo "=========================================="
echo ""

# 执行命令
eval $CMD
