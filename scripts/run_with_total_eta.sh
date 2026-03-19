#!/usr/bin/env bash
set -euo pipefail

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-./.venv/bin/python}
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=${PYTHON_BIN:-python3}
else
  PYTHON_BIN=${PYTHON_BIN:-python}
fi
OUT_ROOT=${OUT_ROOT:-./output}
mkdir -p "$OUT_ROOT"

# MODE: admm | gpadmm | all
MODE=${1:-all}

ADMM_JOBS=(
  "$PYTHON_BIN main.py --dataset mnist --model lenet --num_pre_epochs 3 --num_epochs 10 --num_re_epochs 3 --lr 1e-3 --output-dir $OUT_ROOT/admm_mnist_baseline.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model alexnet --num_pre_epochs 5 --num_epochs 20 --num_re_epochs 5 --lr 1e-3 --output-dir $OUT_ROOT/admm_cifar10_baseline.jsonl"
)

GPADMM_JOBS=(
  "$PYTHON_BIN main.py --dataset mnist --model lenet --use-rigl-admm --sparsity-method uniform --sparsity 0.99 --num-cycles 3 --grow-interval 5 --grow-frac 0.10 --num_re_epochs 3 --lr 1e-3 --output-dir $OUT_ROOT/gpadmm_mnist_gf010.jsonl"
  "$PYTHON_BIN main.py --dataset mnist --model lenet --use-rigl-admm --sparsity-method erk --sparsity 0.99 --num-cycles 4 --grow-interval 5 --grow-frac 0.05 --num_re_epochs 3 --lr 1e-3 --output-dir $OUT_ROOT/gpadmm_mnist_erk.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model alexnet --use-rigl-admm --sparsity-method uniform --sparsity 0.95 --num-cycles 4 --grow-interval 10 --grow-frac 0.10 --num_re_epochs 5 --num_pre_epochs 5 --lr 1e-3 --output-dir $OUT_ROOT/gpadmm_cifar10_u.jsonl"
)

ALL_JOBS=()
if [[ "$MODE" == "admm" || "$MODE" == "all" ]]; then
  ALL_JOBS+=("${ADMM_JOBS[@]}")
fi
if [[ "$MODE" == "gpadmm" || "$MODE" == "all" ]]; then
  ALL_JOBS+=("${GPADMM_JOBS[@]}")
fi

if [[ ${#ALL_JOBS[@]} -eq 0 ]]; then
  echo "No jobs selected. Usage: $0 [admm|gpadmm|all]"
  exit 1
fi

format_hms() {
  local secs=$1
  if (( secs < 0 )); then secs=0; fi
  local h=$((secs / 3600))
  local m=$(((secs % 3600) / 60))
  local s=$((secs % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

TOTAL=${#ALL_JOBS[@]}
BATCH_START=$(date +%s)

for i in "${!ALL_JOBS[@]}"; do
  IDX=$((i + 1))
  CMD="${ALL_JOBS[$i]}"

  RUN_START=$(date +%s)
  echo "[Batch $IDX/$TOTAL] $CMD"
  eval "$CMD"
  RUN_END=$(date +%s)

  ELAPSED=$((RUN_END - BATCH_START))
  AVG=$((ELAPSED / IDX))
  REMAIN=$((AVG * (TOTAL - IDX)))

  echo "[Batch] elapsed=$(format_hms "$ELAPSED") avg/job=$(format_hms "$AVG") eta=$(format_hms "$REMAIN")"
  echo

done

BATCH_END=$(date +%s)
TOTAL_ELAPSED=$((BATCH_END - BATCH_START))
echo "All jobs finished. Total elapsed: $(format_hms "$TOTAL_ELAPSED")"
