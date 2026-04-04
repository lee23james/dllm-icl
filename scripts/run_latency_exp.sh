#!/bin/bash
# =============================================================================
# run_latency_exp.sh
# 实验：记录 Vanilla ICL（3种位置策略）vs Auto-ICL 在 5 个数据集上的 Latency & Accuracy
#
# Vanilla ICL - 放在最后（对照组）：p=4 尾部位置，共 200 次生成
#   → scripts/eval.py  --query_position 4
#
# Vanilla ICL - 放在最前：p=0 首部位置，共 200 次生成
#   → scripts/eval.py  --query_position 0
#
# Vanilla ICL - 随机放置：每个样本随机选 p∈{0..4}，共 200 次生成
#   → scripts/eval.py  --random_position
#
# Auto-ICL（实验组）：枚举 p∈{0,1,2,3,4}，共 200×6=1200 次生成
#   （5次 select_position + 1次 final generate）
#   → improve/generate_improve.py
#
# 配置（steps=gen_length=block_length）：
#   gsm8k     : gen_length=256
#   sudoku    : gen_length=32
#   countdown : gen_length=32
#   math500   : gen_length=256
#   mbpp      : gen_length=128
# =============================================================================

set -e

MODEL=/hy-tmp/dllm-icl/model/LLaDA-8B-Base
DEVICE=cuda:0
NSHOT=4
SAMPLES=200
MODE=original
VERSION=all

# 切换到项目根目录，保证相对路径正确
cd /hy-tmp/dllm-icl

echo "================================================================"
echo " Latency Experiment  |  $(date)"
echo "================================================================"

# -----------------------------------------------------------------------
# Helper：运行单个 task 的四组实验
# 参数：$1=task  $2=gen_length  $3=data_path  $4=result_root
# -----------------------------------------------------------------------
run_task() {
    local TASK=$1
    local GL=$2          # gen_length = steps = block_length
    local DATA=$3
    local RES_ROOT=$4

    echo ""
    echo "================================================================"
    echo "  TASK: ${TASK}   gen_length=${GL}"
    echo "================================================================"

    # ---- Vanilla ICL - 放在最后（对照组） ----
    echo ""
    echo ">>> [Vanilla ICL - Last] task=${TASK}  position=4  samples=${SAMPLES}"
    python scripts/eval.py \
        --task        "${TASK}" \
        --model_name  "${MODEL}" \
        --device      "${DEVICE}" \
        --gen_length  "${GL}" \
        --steps       "${GL}" \
        --block_length "${GL}" \
        --temperature 0.0 \
        --mode        "${MODE}" \
        --data_path   "${DATA}" \
        --result_path "${RES_ROOT}/vanilla_last/${TASK}" \
        --nshot       "${NSHOT}" \
        --query_position 4 \
        --max_samples "${SAMPLES}"

    # ---- Vanilla ICL - 放在最前 ----
    echo ""
    echo ">>> [Vanilla ICL - First] task=${TASK}  position=0  samples=${SAMPLES}"
    python scripts/eval.py \
        --task        "${TASK}" \
        --model_name  "${MODEL}" \
        --device      "${DEVICE}" \
        --gen_length  "${GL}" \
        --steps       "${GL}" \
        --block_length "${GL}" \
        --temperature 0.0 \
        --mode        "${MODE}" \
        --data_path   "${DATA}" \
        --result_path "${RES_ROOT}/vanilla_first/${TASK}" \
        --nshot       "${NSHOT}" \
        --query_position 0 \
        --max_samples "${SAMPLES}"

    # ---- Vanilla ICL - 随机放置 ----
    echo ""
    echo ">>> [Vanilla ICL - Random] task=${TASK}  random_position  samples=${SAMPLES}"
    python scripts/eval.py \
        --task        "${TASK}" \
        --model_name  "${MODEL}" \
        --device      "${DEVICE}" \
        --gen_length  "${GL}" \
        --steps       "${GL}" \
        --block_length "${GL}" \
        --temperature 0.0 \
        --mode        "${MODE}" \
        --data_path   "${DATA}" \
        --result_path "${RES_ROOT}/vanilla_random/${TASK}" \
        --nshot       "${NSHOT}" \
        --random_position \
        --max_samples "${SAMPLES}"

    echo ""
    echo ">>> [Auto-ICL]   task=${TASK}  enumerate p∈{0..4}  samples=${SAMPLES}"
    python improve/generate_improve.py \
        --task        "${TASK}" \
        --model_name  "${MODEL}" \
        --device      "${DEVICE}" \
        --gen_length  "${GL}" \
        --steps       "${GL}" \
        --block_length "${GL}" \
        --temperature 0.0 \
        --mode        "${MODE}" \
        --version     "${VERSION}" \
        --data_path   "${DATA}" \
        --result_path "${RES_ROOT}/auto_icl/${TASK}" \
        --nshot       "${NSHOT}" \
        --samples_num "${SAMPLES}" \
        --lamda1      0.8 \
        --lamda2      0.2
}

# -----------------------------------------------------------------------
# 逐 task 跑
# -----------------------------------------------------------------------
run_task gsm8k     256 ./data/gsm8k.jsonl      ./results/latency_exp
run_task sudoku     32 ./data/sudoku.csv        ./results/latency_exp
run_task countdown  32 ./data/countdown.jsonl   ./results/latency_exp
run_task math500   256 ./data/math500.jsonl     ./results/latency_exp
run_task mbpp      128 ./data/mbpp.jsonl        ./results/latency_exp

echo ""
echo "================================================================"
echo "  ALL DONE  |  $(date)"
echo "================================================================"
