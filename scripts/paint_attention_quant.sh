#!/usr/bin/env bash

set -euo pipefail

TASK=${TASK:-sudoku}
NSHOT=${NSHOT:-5}
MAX_SAMPLES=${MAX_SAMPLES:-50}
POSITION_MODE=${POSITION_MODE:-reverse}
PARAMS_DIR=${PARAMS_DIR:-params/rollout_params}
OUT_DIR=${OUT_DIR:-rollout_results/sudoku}

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  if [[ "${MAX_STEPS}" != "None" ]]; then
    EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
  fi
fi

python3 utils/plot_attention_quant.py \
  --task "${TASK}" \
  --nshot "${NSHOT}" \
  --max_samples "${MAX_SAMPLES}" \
  --position_mode "${POSITION_MODE}" \
  --params_dir "${PARAMS_DIR}" \
  --out_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"
