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

run_jobs() {
  local tag=$1
  shift
  local -a jobs=("$@")

  local i=0
  local total=${#jobs[@]}
  for cmd in "${jobs[@]}"; do
    i=$((i + 1))
    echo "[$tag] (${i}/${total}) $cmd"
    eval "$cmd"
  done
}

case "$MODE" in
  admm)
    run_jobs "ADMM" "${ADMM_JOBS[@]}"
    ;;
  gpadmm)
    run_jobs "GPADMM" "${GPADMM_JOBS[@]}"
    ;;
  all)
    run_jobs "ADMM" "${ADMM_JOBS[@]}"
    run_jobs "GPADMM" "${GPADMM_JOBS[@]}"
    ;;
  *)
    echo "Unsupported mode: $MODE"
    echo "Usage: $0 [admm|gpadmm|all]"
    exit 1
    ;;
esac

echo "All jobs finished."
