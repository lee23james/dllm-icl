#!/bin/bash
# Full experiment with 200 samples per dataset using accelerate for multi-GPU
set -e

MODEL=/hy-tmp/dllm-icl/model/LLaDA-8B-Base
NSHOT=4
SAMPLES=200
MODE=original
VERSION=all
LAMDA1=1.0
LAMDA2=0.0

cd /hy-tmp/dllm-icl

echo "================================================================"
echo " Full Latency Experiment (200 samples)  |  $(date)"
echo " Using accelerate for multi-GPU support"
echo " lamda1=${LAMDA1} lamda2=${LAMDA2}"
echo "================================================================"

# Function to run a single task
run_task() {
    local TASK=$1
    local GL=$2
    local DATA=$3
    local RES_ROOT=$4

    echo ""
    echo "================================================================"
    echo "  TASK: ${TASK}   gen_length=${GL}   samples=${SAMPLES}"
    echo "================================================================"

    # ---- Vanilla ICL ----
    echo ""
    echo ">>> [Vanilla ICL] task=${TASK}  position=4"
    accelerate launch --num_processes 2 scripts/eval.py \
        --task        "${TASK}" \
        --model_name  "${MODEL}" \
        --gen_length  "${GL}" \
        --steps       "${GL}" \
        --block_length "${GL}" \
        --temperature 0.0 \
        --mode        "${MODE}" \
        --data_path   "${DATA}" \
        --result_path "${RES_ROOT}/vanilla/${TASK}" \
        --nshot       "${NSHOT}" \
        --query_position 4 \
        --max_samples "${SAMPLES}"

    # ---- Auto-ICL ----
    echo ""
    echo ">>> [Auto-ICL]   task=${TASK}  enumerate p∈{0..4}"
    accelerate launch --num_processes 2 improve/generate_improve.py \
        --task        "${TASK}" \
        --model_name  "${MODEL}" \
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
        --lamda1      "${LAMDA1}" \
        --lamda2      "${LAMDA2}"
}

# Run all 5 datasets sequentially (accelerate handles multi-GPU within each task)
run_task gsm8k     256 ./data/gsm8k.jsonl      ./results/latency_exp_200
run_task sudoku     32 ./data/sudoku.csv        ./results/latency_exp_200
run_task countdown  32 ./data/countdown.jsonl   ./results/latency_exp_200
run_task math500   256 ./data/math500.jsonl     ./results/latency_exp_200
run_task mbpp      128 ./data/mbpp.jsonl        ./results/latency_exp_200

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE  |  $(date)"
echo "================================================================"
