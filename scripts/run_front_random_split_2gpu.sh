#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-/hy-tmp/dllm-icl/model/LLaDA-8B-Base}
NSHOT=${NSHOT:-4}
SAMPLES=${SAMPLES:-100}
MODE=${MODE:-original}
RESULT_ROOT=${RESULT_ROOT:-./results/front_random_split_2gpu}
LOG_ROOT=${LOG_ROOT:-./logs/front_random_split_2gpu}

cd /hy-tmp/dllm-icl
mkdir -p "${LOG_ROOT}"

part_a_samples=$(((SAMPLES + 1) / 2))
part_b_samples=$((SAMPLES / 2))

run_one_half() {
    local GPU_ID=$1
    local TASK=$2
    local GL=$3
    local DATA=$4
    local PLACEMENT=$5
    local MAX_SAMPLES=$6
    local RESULT_PATH=$7
    local LOG_PATH=$8

    local EXTRA_ARGS=()
    if [[ "${PLACEMENT}" == "first" ]]; then
        EXTRA_ARGS+=(--query_position 0)
    elif [[ "${PLACEMENT}" == "random" ]]; then
        EXTRA_ARGS+=(--random_position)
    else
        echo "Unsupported placement: ${PLACEMENT}" >&2
        exit 1
    fi

    env CUDA_VISIBLE_DEVICES="${GPU_ID}" stdbuf -oL -eL python scripts/eval.py \
        --task "${TASK}" \
        --model_name "${MODEL}" \
        --device cuda:0 \
        --gen_length "${GL}" \
        --steps "${GL}" \
        --block_length "${GL}" \
        --temperature 0.0 \
        --mode "${MODE}" \
        --data_path "${DATA}" \
        --result_path "${RESULT_PATH}" \
        --nshot "${NSHOT}" \
        --max_samples "${MAX_SAMPLES}" \
        "${EXTRA_ARGS[@]}" > "${LOG_PATH}" 2>&1
}

run_eval() {
    local TASK=$1
    local GL=$2
    local DATA_A=$3
    local DATA_B=$4
    local PLACEMENT=$5

    local RESULT_BASE="${RESULT_ROOT}/${PLACEMENT}/${TASK}"
    local LOG_A="${LOG_ROOT}/${TASK}_${PLACEMENT}_gpu0.log"
    local LOG_B="${LOG_ROOT}/${TASK}_${PLACEMENT}_gpu1.log"
    local RESULT_A="${RESULT_BASE}/part_a"
    local RESULT_B="${RESULT_BASE}/part_b"
    local META_PATH="${RESULT_BASE}/run_meta.json"

    mkdir -p "${RESULT_BASE}"

    echo ""
    echo "================================================================"
    echo " TASK=${TASK} placement=${PLACEMENT} total_samples=${SAMPLES} gen_length=${GL}"
    echo " GPU0: data=${DATA_A} samples=${part_a_samples} log=${LOG_A}"
    echo " GPU1: data=${DATA_B} samples=${part_b_samples} log=${LOG_B}"
    echo "================================================================"

    local start_ts
    start_ts=$(date +%s)

    run_one_half 0 "${TASK}" "${GL}" "${DATA_A}" "${PLACEMENT}" "${part_a_samples}" "${RESULT_A}" "${LOG_A}" &
    local pid_a=$!
    run_one_half 1 "${TASK}" "${GL}" "${DATA_B}" "${PLACEMENT}" "${part_b_samples}" "${RESULT_B}" "${LOG_B}" &
    local pid_b=$!

    local rc_a=0
    local rc_b=0
    wait "${pid_a}" || rc_a=$?
    wait "${pid_b}" || rc_b=$?

    local end_ts
    end_ts=$(date +%s)

    python - <<PY
import json
from pathlib import Path
meta = {
    "task": "${TASK}",
    "placement": "${PLACEMENT}",
    "samples_total": ${SAMPLES},
    "samples_part_a": ${part_a_samples},
    "samples_part_b": ${part_b_samples},
    "gen_length": ${GL},
    "mode": "${MODE}",
    "result_part_a": "${RESULT_A}",
    "result_part_b": "${RESULT_B}",
    "log_part_a": "${LOG_A}",
    "log_part_b": "${LOG_B}",
    "start_ts": ${start_ts},
    "end_ts": ${end_ts},
    "wall_seconds": ${end_ts} - ${start_ts},
    "return_code_part_a": ${rc_a},
    "return_code_part_b": ${rc_b},
}
path = Path("${META_PATH}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
PY

    if [[ ${rc_a} -ne 0 || ${rc_b} -ne 0 ]]; then
        echo "Run failed for task=${TASK} placement=${PLACEMENT}: rc_a=${rc_a} rc_b=${rc_b}" >&2
        exit 1
    fi
}

run_task() {
    local TASK=$1
    local GL=$2
    local DATA_A=$3
    local DATA_B=$4
    run_eval "${TASK}" "${GL}" "${DATA_A}" "${DATA_B}" first
    run_eval "${TASK}" "${GL}" "${DATA_A}" "${DATA_B}" random
}

echo "================================================================"
echo " Front + Random Split 2GPU Experiment | $(date)"
echo " model=${MODEL}"
echo " nshot=${NSHOT} total_samples=${SAMPLES} mode=${MODE}"
echo " result_root=${RESULT_ROOT}"
echo " log_root=${LOG_ROOT}"
echo "================================================================"

run_task gsm8k 256 ./data/gsm8k_part_a.jsonl ./data/gsm8k_part_b.jsonl
run_task sudoku 32 ./data/sudoku_part_a.csv ./data/sudoku_part_b.csv
run_task countdown 32 ./data/countdown_part_a.jsonl ./data/countdown_part_b.jsonl
run_task math500 256 ./data/math500_part_a.jsonl ./data/math500_part_b.jsonl
run_task mbpp 128 ./data/mbpp_part_a.jsonl ./data/mbpp_part_b.jsonl

echo ""
echo "================================================================"
echo " All split front/random experiments complete | $(date)"
echo "================================================================"
