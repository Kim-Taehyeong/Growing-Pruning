#!/usr/bin/env python3
"""Run a 100-point CIFAR-10 pruning-LR x retraining-LR sweep."""

import argparse
import copy
import itertools
import subprocess
import tempfile
from pathlib import Path

import yaml


SPARSITIES = (0.8, 0.9, 0.95, 0.98)
TRAIN_LRS = (0.001, 0.005, 0.01, 0.025, 0.05)
RETRAIN_LRS = (0.001, 0.005, 0.01, 0.025, 0.05)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4 sparsities x 5 training LRs x 5 retraining LRs (100 runs)."
    )
    parser.add_argument("--model", choices=("resnet56", "vgg16"), default="resnet56")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--base-config",
        default="",
        help="Base YAML; defaults to the pretrained global SGD config for --model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most this many selected experiments; 0 runs all 100.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned experiments without launching training.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Run even when the output JSONL already exists and is non-empty.",
    )
    return parser.parse_args()


def lr_tag(value):
    return format(value, "g").replace(".", "p")


def load_base_config(path):
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "common" not in config or "gpadmm" not in config:
        raise ValueError(f"Invalid GP-ADMM config: {path}")
    return config


def build_experiment(base_config, model, gpu, sparsity, train_lr, retrain_lr):
    config = copy.deepcopy(base_config)
    name = (
        f"gpadmm_cifar10_{sparsity:g}_{model}_pretrained_global_"
        f"train_lr_{lr_tag(train_lr)}_retrain_lr_{lr_tag(retrain_lr)}"
    )

    config["common"].update(
        {
            "model": model,
            "device": f"cuda:{gpu}",
            "lr": train_lr,
            "retrain_lr": retrain_lr,
            "output_dir": f"./output/{name}.jsonl",
            "save_dir": f"./runs/{name}",
        }
    )
    config["gpadmm"]["sparsity"] = sparsity
    return name, config


def run_experiment(config):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8", delete=False
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        config_path = Path(handle.name)
    try:
        subprocess.run(
            ["python", "main.py", "--config", str(config_path)],
            check=True,
        )
    finally:
        config_path.unlink(missing_ok=True)


def main():
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater")

    base_path = Path(args.base_config) if args.base_config else Path(
        f"configs/gpadmm_cifar10_{args.model}_pretrained_global_sgd_default.yaml"
    )
    base_config = load_base_config(base_path)

    grid = list(itertools.product(SPARSITIES, TRAIN_LRS, RETRAIN_LRS))
    if args.limit:
        grid = grid[: args.limit]

    print(f"Base config: {base_path}")
    print(f"Planned experiments: {len(grid)} / 100 on cuda:{args.gpu}")

    launched = skipped = 0
    for index, (sparsity, train_lr, retrain_lr) in enumerate(grid, start=1):
        name, config = build_experiment(
            base_config, args.model, args.gpu, sparsity, train_lr, retrain_lr
        )
        output_path = Path(config["common"]["output_dir"])
        label = (
            f"[{index:03d}/{len(grid):03d}] {name} "
            f"(s={sparsity:g}, train_lr={train_lr:g}, retrain_lr={retrain_lr:g})"
        )

        if output_path.exists() and output_path.stat().st_size > 0 and not args.rerun_existing:
            print(f"SKIP {label}: {output_path} already exists")
            skipped += 1
            continue

        print(f"RUN  {label}")
        launched += 1
        if not args.dry_run:
            run_experiment(config)

    print(f"Done: launched={launched}, skipped={skipped}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
