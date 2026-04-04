#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-/hy-tmp/dllm-icl/model/Dream-7B-Base}
NSHOT=${NSHOT:-4}
QUERY_POSITION=${QUERY_POSITION:-0}
SAMPLES=${SAMPLES:-100}
RESULT_ROOT=${RESULT_ROOT:-./results/dream_last_split_2gpu}
LOG_ROOT=${LOG_ROOT:-./logs/dream_last_split_2gpu}

cd /hy-tmp/dllm-icl
mkdir -p "${LOG_ROOT}"

part_a_samples=$(((SAMPLES + 1) / 2))
part_b_samples=$((SAMPLES / 2))

run_one_half() {
    local GPU_ID=$1
    local TASK=$2
    local GL=$3
    local DATA=$4
    local MAX_SAMPLES=$5
    local RESULT_PATH=$6
    local LOG_PATH=$7

    env CUDA_VISIBLE_DEVICES="${GPU_ID}" stdbuf -oL -eL python scripts/eval_dream.py \
        --task "${TASK}" \
        --model_name "${MODEL}" \
        --device cuda:0 \
        --gen_length "${GL}" \
        --steps "${GL}" \
        --temperature 0.0 \
        --data_path "${DATA}" \
        --result_path "${RESULT_PATH}" \
        --nshot "${NSHOT}" \
        --query_position "${QUERY_POSITION}" \
        --max_samples "${MAX_SAMPLES}" > "${LOG_PATH}" 2>&1
}

run_eval() {
    local TASK=$1
    local GL=$2
    local DATA_A=$3
    local DATA_B=$4

    local RESULT_BASE="${RESULT_ROOT}/${TASK}"
    local LOG_A="${LOG_ROOT}/${TASK}_gpu0.log"
    local LOG_B="${LOG_ROOT}/${TASK}_gpu1.log"
    local RESULT_A="${RESULT_BASE}/part_a"
    local RESULT_B="${RESULT_BASE}/part_b"
    local META_PATH="${RESULT_BASE}/run_meta.json"

    mkdir -p "${RESULT_BASE}"

    echo ""
    echo "================================================================"
    echo " TASK=${TASK} position=${QUERY_POSITION} total_samples=${SAMPLES} gen_length=${GL}"
    echo " GPU0: data=${DATA_A} samples=${part_a_samples} log=${LOG_A}"
    echo " GPU1: data=${DATA_B} samples=${part_b_samples} log=${LOG_B}"
    echo "================================================================"

    local start_ts
    start_ts=$(date +%s)

    run_one_half 0 "${TASK}" "${GL}" "${DATA_A}" "${part_a_samples}" "${RESULT_A}" "${LOG_A}" &
    local pid_a=$!
    run_one_half 1 "${TASK}" "${GL}" "${DATA_B}" "${part_b_samples}" "${RESULT_B}" "${LOG_B}" &
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
    "query_position": ${QUERY_POSITION},
    "samples_total": ${SAMPLES},
    "samples_part_a": ${part_a_samples},
    "samples_part_b": ${part_b_samples},
    "gen_length": ${GL},
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
        echo "Run failed for task=${TASK}: rc_a=${rc_a} rc_b=${rc_b}" >&2
        exit 1
    fi
}

echo "================================================================"
echo " Dream Last-Position Split 2GPU Experiment | $(date)"
echo " model=${MODEL}"
echo " nshot=${NSHOT} query_position=${QUERY_POSITION} total_samples=${SAMPLES}"
echo " result_root=${RESULT_ROOT}"
echo " log_root=${LOG_ROOT}"
echo "================================================================"

run_eval gsm8k 256 ./data/gsm8k_part_a.jsonl ./data/gsm8k_part_b.jsonl
run_eval sudoku 32 ./data/sudoku_part_a.csv ./data/sudoku_part_b.csv
run_eval countdown 32 ./data/countdown_part_a.jsonl ./data/countdown_part_b.jsonl
run_eval math500 256 ./data/math500_part_a.jsonl ./data/math500_part_b.jsonl
run_eval mbpp 128 ./data/mbpp_part_a.jsonl ./data/mbpp_part_b.jsonl

echo ""
echo "================================================================"
echo " All Dream last-position experiments complete | $(date)"
echo "================================================================"
