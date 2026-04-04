#!/bin/bash
set -e

MODEL=/hy-tmp/dllm-icl/model/LLaDA-8B-Base
DEVICE=cuda:0
NSHOT=4
SAMPLES=2
MODE=original
VERSION=all

cd /hy-tmp/dllm-icl

echo "================================================================"
echo " Latency Experiment TEST (2 samples per dataset)  |  $(date)"
echo " lamda1=1.0 lamda2=0.0 (confidence only)"
echo "================================================================"

run_task() {
    local TASK=$1
    local GL=$2
    local DATA=$3
    local RES_ROOT=$4

    echo ""
    echo "================================================================"
    echo "  TASK: ${TASK}   gen_length=${GL}   samples=${SAMPLES}"
    echo "================================================================"

    echo ""
    echo ">>> [Vanilla ICL] task=${TASK}  position=4"
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
        --result_path "${RES_ROOT}/vanilla/${TASK}" \
        --nshot       "${NSHOT}" \
        --query_position 4 \
        --max_samples "${SAMPLES}"

    echo ""
    echo ">>> [Auto-ICL]   task=${TASK}  enumerate p∈{0..4}"
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
        --lamda1      1.0 \
        --lamda2      0.0
}

run_task gsm8k     256 ./data/gsm8k.jsonl      ./results/latency_exp_test
run_task sudoku     32 ./data/sudoku.csv        ./results/latency_exp_test
run_task countdown  32 ./data/countdown.jsonl   ./results/latency_exp_test
run_task math500   256 ./data/math500.jsonl     ./results/latency_exp_test
run_task mbpp      128 ./data/mbpp.jsonl        ./results/latency_exp_test

echo ""
echo "================================================================"
echo "  TEST COMPLETE  |  $(date)"
echo "================================================================"
