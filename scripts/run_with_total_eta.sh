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
  # MNIST/LeNet (Extreme Sparsity)
  "$PYTHON_BIN main.py --dataset mnist --model lenet --sparsity 0.95 --num_epochs 20 --num_re_epochs 5 --lr 1e-3 --output-dir $OUT_ROOT/admm_mnist_s95.jsonl"
  "$PYTHON_BIN main.py --dataset mnist --model lenet --sparsity 0.98 --num_epochs 20 --num_re_epochs 5 --lr 1e-3 --output-dir $OUT_ROOT/admm_mnist_s98.jsonl"
  "$PYTHON_BIN main.py --dataset mnist --model lenet --sparsity 0.99 --num_epochs 20 --num_re_epochs 5 --lr 1e-3 --output-dir $OUT_ROOT/admm_mnist_s99.jsonl"

  # CIFAR10/VGG19 (Standard Bench)
  "$PYTHON_BIN main.py --dataset cifar10 --model vgg19 --sparsity 0.90 --num_epochs 40 --num_re_epochs 10 --lr 1e-3 --output-dir $OUT_ROOT/admm_c10_vgg_s90.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model vgg19 --sparsity 0.95 --num_epochs 40 --num_re_epochs 10 --lr 1e-3 --output-dir $OUT_ROOT/admm_c10_vgg_s95.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model vgg19 --sparsity 0.98 --num_epochs 40 --num_re_epochs 10 --lr 1e-3 --output-dir $OUT_ROOT/admm_c10_vgg_s98.jsonl"

  # CIFAR10/ResNet20 (Residual Connection)
  "$PYTHON_BIN main.py --dataset cifar10 --model resnet20 --sparsity 0.90 --num_epochs 40 --num_re_epochs 10 --lr 5e-4 --output-dir $OUT_ROOT/admm_c10_r20_s90.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model resnet20 --sparsity 0.95 --num_epochs 40 --num_re_epochs 10 --lr 5e-4 --output-dir $OUT_ROOT/admm_c10_r20_s95.jsonl"
)

# ---------------------------------------------------------
# 2. GPADMM (Proposed: RigL + ADMM)
# ---------------------------------------------------------
GPADMM_JOBS=(
  # [MNIST/LeNet] 극한의 희소성 테스트 (Interval 10)
  "$PYTHON_BIN main.py --dataset mnist --model lenet --use-rigl-admm --sparsity 0.95 --num-cycles 4 --grow-interval 10 --num_re_epochs 5 --lr 1e-3 --output-dir $OUT_ROOT/gp_mnist_s95_i10.jsonl"
  "$PYTHON_BIN main.py --dataset mnist --model lenet --use-rigl-admm --sparsity 0.98 --num-cycles 4 --grow-interval 10 --num_re_epochs 5 --lr 1e-3 --output-dir $OUT_ROOT/gp_mnist_s98_i10.jsonl"
  "$PYTHON_BIN main.py --dataset mnist --model lenet --use-rigl-admm --sparsity 0.99 --num-cycles 4 --grow-interval 10 --num_re_epochs 5 --lr 1e-3 --output-dir $OUT_ROOT/gp_mnist_s99_i10.jsonl"

  # [CIFAR10/VGG19] 층이 깊은 모델의 수렴력 테스트 (Interval 20)
  "$PYTHON_BIN main.py --dataset cifar10 --model vgg19 --use-rigl-admm --sparsity 0.90 --num-cycles 4 --grow-interval 20 --num_re_epochs 10 --lr 1e-3 --output-dir $OUT_ROOT/gp_c10_vgg_s90_i20.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model vgg19 --use-rigl-admm --sparsity 0.95 --num-cycles 4 --grow-interval 20 --num_re_epochs 10 --lr 1e-3 --output-dir $OUT_ROOT/gp_c10_vgg_s95_i20.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model vgg19 --use-rigl-admm --sparsity 0.98 --num-cycles 4 --grow-interval 25 --num_re_epochs 10 --lr 5e-4 --output-dir $OUT_ROOT/gp_c10_vgg_s98_i25_lr5e4.jsonl"

  # [CIFAR10/ResNet20] Skip-connection 모델의 복구력 테스트 (Interval 20)
  "$PYTHON_BIN main.py --dataset cifar10 --model resnet20 --use-rigl-admm --sparsity 0.90 --num-cycles 5 --grow-interval 15 --num_re_epochs 10 --lr 5e-4 --output-dir $OUT_ROOT/gp_c10_r20_s90_i15.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model resnet20 --use-rigl-admm --sparsity 0.95 --num-cycles 5 --grow-interval 20 --num_re_epochs 10 --lr 5e-4 --output-dir $OUT_ROOT/gp_c10_r20_s95_i20.jsonl"
  "$PYTHON_BIN main.py --dataset cifar10 --model resnet20 --use-rigl-admm --sparsity 0.98 --num-cycles 5 --grow-interval 25 --num_re_epochs 10 --lr 3e-4 --output-dir $OUT_ROOT/gp_c10_r20_s98_i25_lr3e4.jsonl"

  # [CIFAR100/ResNet18] 고난이도 데이터셋 (Interval 25)
  "$PYTHON_BIN main.py --dataset cifar100 --model resnet18 --use-rigl-admm --sparsity 0.80 --num-cycles 4 --grow-interval 20 --num_re_epochs 10 --lr 1e-4 --output-dir $OUT_ROOT/gp_c100_r18_s80_i20.jsonl"
  "$PYTHON_BIN main.py --dataset cifar100 --model resnet18 --use-rigl-admm --sparsity 0.90 --num-cycles 4 --grow-interval 25 --num_re_epochs 15 --lr 1e-4 --output-dir $OUT_ROOT/gp_c100_r18_s90_i25.jsonl"
  "$PYTHON_BIN main.py --dataset cifar100 --model resnet18 --use-rigl-admm --sparsity 0.95 --num-cycles 4 --grow-interval 30 --num_re_epochs 15 --lr 5e-5 --output-dir $OUT_ROOT/gp_c100_r18_s95_i30_lr5e5.jsonl"

  # [ImageNet/ResNet50] 실제 스케일 테스트 (Interval 5~10)
  "$PYTHON_BIN main.py --dataset imagenet --model resnet50 --use-rigl-admm --sparsity 0.70 --num-cycles 3 --grow-interval 5 --num_re_epochs 5 --lr 1e-4 --output-dir $OUT_ROOT/gp_img_r50_s70.jsonl --data_dir /home/users/taehyeok/ILSVRC_2012"
  "$PYTHON_BIN main.py --dataset imagenet --model resnet50 --use-rigl-admm --sparsity 0.80 --num-cycles 3 --grow-interval 8 --num_re_epochs 5 --lr 1e-4 --output-dir $OUT_ROOT/gp_img_r50_s80.jsonl --data_dir /home/users/taehyeok/ILSVRC_2012"
  "$PYTHON_BIN main.py --dataset imagenet --model resnet50 --use-rigl-admm --sparsity 0.90 --num-cycles 3 --grow-interval 10 --num_re_epochs 5 --lr 5e-5 --output-dir $OUT_ROOT/gp_img_r50_s90_lr5e5.jsonl --data_dir /home/users/taehyeok/ILSVRC_2012"
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
