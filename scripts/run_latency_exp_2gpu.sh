#!/bin/bash
# Full experiment with 200 samples using 2 GPUs via accelerate
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
echo " Full Latency Experiment (200 samples, 2 GPUs)  |  $(date)"
echo " Using accelerate with num_processes=2"
echo " lamda1=${LAMDA1} lamda2=${LAMDA2}"
echo "================================================================"

export ACCELERATE_CONFIG_FILE=/hy-tmp/dllm-icl/accelerate_config_2gpu.yaml

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
    accelerate launch --config_file /hy-tmp/dllm-icl/accelerate_config_2gpu.yaml scripts/eval.py \
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

    echo ""
    echo ">>> [Auto-ICL]   task=${TASK}  enumerate p∈{0..4}"
    accelerate launch --config_file /hy-tmp/dllm-icl/accelerate_config_2gpu.yaml improve/generate_improve.py \
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

run_task gsm8k     256 ./data/gsm8k.jsonl      ./results/latency_exp_2gpu
run_task sudoku     32 ./data/sudoku.csv        ./results/latency_exp_2gpu
run_task countdown  32 ./data/countdown.jsonl   ./results/latency_exp_2gpu
run_task math500   256 ./data/math500.jsonl     ./results/latency_exp_2gpu
run_task mbpp      128 ./data/mbpp.jsonl        ./results/latency_exp_2gpu

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE  |  $(date)"
echo "================================================================"
