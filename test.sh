#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./output

# # =========================
# # Default ADMM (MNIST)
# # =========================
# python main.py --dataset mnist --output-dir "./output/MNIST_#0.json"

# # =========================
# # RIGL ADMM GrowFrac Test (MNIST)
# # =========================
# python main.py --dataset mnist --output-dir "./output/MNIST_#1.json"  --use-rigl-admm --grow-frac 0.1
# python main.py --dataset mnist --output-dir "./output/MNIST_#2.json"  --use-rigl-admm --grow-frac 0.2
# python main.py --dataset mnist --output-dir "./output/MNIST_#3.json"  --use-rigl-admm --grow-frac 0.05
# python main.py --dataset mnist --output-dir "./output/MNIST_#4.json"  --use-rigl-admm --grow-frac 0.01

# # =========================
# # RIGL ADMM Num Cycle Test (MNIST)
# # =========================
# python main.py --dataset mnist --output-dir "./output/MNIST_#5.json"  --use-rigl-admm --num-cycles 3
# python main.py --dataset mnist --output-dir "./output/MNIST_#6.json"  --use-rigl-admm --num-cycles 4
# python main.py --dataset mnist --output-dir "./output/MNIST_#7.json"  --use-rigl-admm --num-cycles 5
# python main.py --dataset mnist --output-dir "./output/MNIST_#8.json"  --use-rigl-admm --num-cycles 2

# # =========================
# # RIGL ADMM Grow Interval Test (MNIST)
# # =========================
# python main.py --dataset mnist --output-dir "./output/MNIST_#9.json"   --use-rigl-admm --grow-interval 5
# python main.py --dataset mnist --output-dir "./output/MNIST_#10.json"  --use-rigl-admm --grow-interval 10
# python main.py --dataset mnist --output-dir "./output/MNIST_#11.json"  --use-rigl-admm --grow-interval 15
# python main.py --dataset mnist --output-dir "./output/MNIST_#12.json"  --use-rigl-admm --grow-interval 20

# # =========================
# # RIGL ADMM Retrain Epoch Test (MNIST)
# # =========================
# python main.py --dataset mnist --output-dir "./output/MNIST_#13.json" --use-rigl-admm --num_re_epochs 3
# python main.py --dataset mnist --output-dir "./output/MNIST_#14.json" --use-rigl-admm --num_re_epochs 4
# python main.py --dataset mnist --output-dir "./output/MNIST_#15.json" --use-rigl-admm --num_re_epochs 5

# # =========================
# # RIGL ADMM Init Method Test (MNIST)
# # (Random -> --init-method normal 로 매핑)
# # =========================
# python main.py --dataset mnist --output-dir "./output/MNIST_#16.json" --use-rigl-admm --init-method erk
# python main.py --dataset mnist --output-dir "./output/MNIST_#17.json" --use-rigl-admm --init-method er
# python main.py --dataset mnist --output-dir "./output/MNIST_#18.json" --use-rigl-admm --init-method random

# # =========================
# # RIGL ADMM Learning Rate Test (MNIST)
# # =========================
# python main.py --dataset mnist --output-dir "./output/MNIST_#19.json" --use-rigl-admm --lr 1e-3
# python main.py --dataset mnist --output-dir "./output/MNIST_#20.json" --use-rigl-admm --lr 5e-3
# python main.py --dataset mnist --output-dir "./output/MNIST_#21.json" --use-rigl-admm --lr 5e-4


# =========================
# RIGL ADMM GrowFrac Test (CIFAR10)
# =========================
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#1.json"  --use-rigl-admm --grow-frac 0.1 --num_epochs 20 --num_re_epochs 5 --num_pre_epochs 5

# =========================
# RIGL ADMM Init Method Test (CIFAR10)
# (Random -> --init-method normal 로 매핑)
# =========================
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#2.json" --use-rigl-admm --init-method random --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#3.json" --use-rigl-admm --init-method er --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#4.json" --use-rigl-admm --init-method erk --num_re_epochs 5 --num_pre_epochs 5

# =========================
# RIGL ADMM Retrain Epoch Test (CIFAR10)
# =========================
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#5.json" --use-rigl-admm --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#6.json" --use-rigl-admm --num_re_epochs 4 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#7.json" --use-rigl-admm --num_re_epochs 3 --num_pre_epochs 5

# =========================
# RIGL ADMM Grow Interval Test (CIFAR10)
# =========================
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#8.json"  --use-rigl-admm --grow-interval 20 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#9.json"  --use-rigl-admm --grow-interval 15 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#10.json" --use-rigl-admm --grow-interval 10 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#11.json" --use-rigl-admm --grow-interval 5 --num_re_epochs 5 --num_pre_epochs 5

# =========================
# RIGL ADMM Num Cycle Test (CIFAR10)
# =========================
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#12.json" --use-rigl-admm --num-cycles 2 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#13.json" --use-rigl-admm --num-cycles 5 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#14.json" --use-rigl-admm --num-cycles 4 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#15.json" --use-rigl-admm --num-cycles 3 --num_re_epochs 5 --num_pre_epochs 5

# =========================
# RIGL ADMM GrowFrac Variants (CIFAR10)
# =========================
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#16.json" --use-rigl-admm --grow-frac 0.01 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#17.json" --use-rigl-admm --grow-frac 0.05 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#18.json" --use-rigl-admm --grow-frac 0.2 --num_re_epochs 5 --num_pre_epochs 5

# =========================
# RIGL ADMM Learning Rate Test (CIFAR10)
# (순번 유지: #19=1e-3, #21=5e-4, #20=5e-3)
# =========================
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#19.json" --use-rigl-admm --lr 1e-3 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#21.json" --use-rigl-admm --lr 5e-4 --num_re_epochs 5 --num_pre_epochs 5
python main.py --dataset cifar10 --output-dir "./output/CIFAR10_#20.json" --use-rigl-admm --lr 5e-3 --num_re_epochs 5 --num_pre_epochs 5
