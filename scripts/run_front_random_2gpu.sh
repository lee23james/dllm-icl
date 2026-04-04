#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-/hy-tmp/dllm-icl/model/LLaDA-8B-Base}
NSHOT=${NSHOT:-4}
SAMPLES=${SAMPLES:-100}
MODE=${MODE:-original}
RESULT_ROOT=${RESULT_ROOT:-./results/front_random_2gpu}
LOG_ROOT=${LOG_ROOT:-./logs/front_random_2gpu}

cd /hy-tmp/dllm-icl
mkdir -p "${LOG_ROOT}"

run_eval() {
    local TASK=$1
    local GL=$2
    local DATA=$3
    local PLACEMENT=$4
    local RESULT_PATH="${RESULT_ROOT}/${PLACEMENT}/${TASK}"
    local LOG_PATH="${LOG_ROOT}/${TASK}_${PLACEMENT}.log"

    local EXTRA_ARGS=()
    if [[ "${PLACEMENT}" == "first" ]]; then
        EXTRA_ARGS+=(--query_position 0)
    elif [[ "${PLACEMENT}" == "random" ]]; then
        EXTRA_ARGS+=(--random_position)
    else
        echo "Unsupported placement: ${PLACEMENT}" >&2
        exit 1
    fi

    echo ""
    echo "================================================================"
    echo " TASK=${TASK} placement=${PLACEMENT} samples=${SAMPLES} gen_length=${GL}"
    echo " LOG=${LOG_PATH}"
    echo "================================================================"

    stdbuf -oL -eL accelerate launch \
        --config_file /hy-tmp/dllm-icl/accelerate_config_2gpu.yaml \
        scripts/eval.py \
        --task "${TASK}" \
        --model_name "${MODEL}" \
        --gen_length "${GL}" \
        --steps "${GL}" \
        --block_length "${GL}" \
        --temperature 0.0 \
        --mode "${MODE}" \
        --data_path "${DATA}" \
        --result_path "${RESULT_PATH}" \
        --nshot "${NSHOT}" \
        --max_samples "${SAMPLES}" \
        "${EXTRA_ARGS[@]}" | tee "${LOG_PATH}"
}

run_task() {
    local TASK=$1
    local GL=$2
    local DATA=$3
    run_eval "${TASK}" "${GL}" "${DATA}" first
    run_eval "${TASK}" "${GL}" "${DATA}" random
}

echo "================================================================"
echo " Front + Random 2GPU Experiment | $(date)"
echo " model=${MODEL}"
echo " nshot=${NSHOT} samples=${SAMPLES} mode=${MODE}"
echo " result_root=${RESULT_ROOT}"
echo " log_root=${LOG_ROOT}"
echo "================================================================"

run_task gsm8k 256 ./data/gsm8k.jsonl
run_task sudoku 32 ./data/sudoku.csv
run_task countdown 32 ./data/countdown.jsonl
run_task math500 256 ./data/math500.jsonl
run_task mbpp 128 ./data/mbpp.jsonl

echo ""
echo "================================================================"
echo " All front/random experiments complete | $(date)"
echo "================================================================"
