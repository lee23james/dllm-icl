#!/usr/bin/env bash

set -euo pipefail

# 可选环境变量：
#   TASK       (默认: sudoku)
#   NSHOT      (默认: 5)
#   MAX_SAMPLES(默认: 50)
#   MAX_STEPS  (如果设置为整数则生效；为空或 'None' 则不加 --max_steps)
#   MODE       (默认: pos_vs_example)
#   POSITION_MODE (默认: reverse)
#   PARAMS_DIR (默认: params/rollout_params)
#   OUT        (默认: rollout_results/sudoku/recency_pos_vs_example.png)
#   EXPORT_TABLES (默认: 1；为 1/true/yes 时导出 CSV)
#   TABLE_PREFIX  (默认: 空；为空时使用 OUT 去掉后缀作为 CSV 前缀)

TASK=${TASK:-sudoku}
NSHOT=${NSHOT:-5}
MAX_SAMPLES=${MAX_SAMPLES:-50}
MODE=${MODE:-pos_vs_example}
POSITION_MODE=${POSITION_MODE:-reverse}
PARAMS_DIR=${PARAMS_DIR:-params/rollout_params}
OUT=${OUT:-rollout_results/sudoku/recency_pos_vs_example.png}
EXPORT_TABLES=${EXPORT_TABLES:-1}
TABLE_PREFIX=${TABLE_PREFIX:-}

EXTRA_ARGS=()

# 只有在 MAX_STEPS 是非空且不为 'None' 时，才传给脚本，避免 argparse 报错
if [[ -n "${MAX_STEPS:-}" ]]; then
  if [[ "${MAX_STEPS}" != "None" ]]; then
    EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
  fi
fi

case "${EXPORT_TABLES}" in
  1|true|TRUE|yes|YES)
    EXTRA_ARGS+=(--save_tables)
    ;;
esac

if [[ -n "${TABLE_PREFIX}" ]]; then
  EXTRA_ARGS+=(--table_prefix "${TABLE_PREFIX}")
fi

python3 utils/plot_recency_heatmap.py \
  --task "${TASK}" \
  --nshot "${NSHOT}" \
  --max_samples "${MAX_SAMPLES}" \
  --mode "${MODE}" \
  --position_mode "${POSITION_MODE}" \
  --params_dir "${PARAMS_DIR}" \
  --out "${OUT}" \
  "${EXTRA_ARGS[@]}"
