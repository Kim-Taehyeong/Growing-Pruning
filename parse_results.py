#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


SUMMARY_COLUMNS = [
    "method",
    "gpadmm_prune_scope",
    "sparsity_method",
    "dataset",
    "model",
    "seed",
    "target_sparsity",
    "measured_sparsity",
    "top1_acc",
    "top5_acc",
    "delta_top1",
    "loss",
    "total_params_M",
    "nonzero_params_M",
    "gflops",
    "stage",
    "epoch",
    "optimizer",
    "lr",
    "rho",
    "c",
    "num_cycles",
    "grow_interval",
    "num_re_epochs",
    "is_lr_sweep",
    "best_lr_candidate",
    "rank_within_group",
    "unstable_flag",
    "output_file",
]

BEST_LR_COLUMNS = [
    "dataset",
    "model",
    "method",
    "target_sparsity",
    "recommended_lr",
    "top1_acc",
    "measured_sparsity",
    "output_file",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize pruning experiment JSONL files.")
    parser.add_argument("--input-glob", default="./output/**/*.jsonl", help="Glob for JSONL result files")
    parser.add_argument("--out", default="results_summary.csv", help="Output CSV path")
    parser.add_argument(
        "--best-lr-out",
        default="",
        help="Optional CSV path for recommended LR config per dataset/model/method/sparsity",
    )
    return parser.parse_args()


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping invalid JSON {path}:{line_no}: {exc}")


def stage_text(record):
    return str(record.get("stage") or record.get("prefix") or "").strip().lower()


def get_value(record, key, default=None):
    if key in record:
        return record.get(key)
    args = record.get("args")
    if isinstance(args, dict):
        return args.get(key, default)
    return default


def to_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def stage_priority(record):
    stage = stage_text(record)
    epoch = to_int(record.get("epoch"), default=-1)
    if stage == "retraining":
        return (4, epoch)
    if "final" in stage:
        return (3, epoch)
    if "post-pruning" in stage or "post_pruning" in stage:
        return (2, epoch)
    if stage == "dense-eval":
        return (1, epoch)
    return (0, epoch)


def select_final_record(records):
    scored = []
    for index, record in enumerate(records):
        if record.get("top1_acc") is None and record.get("loss") is None:
            continue
        priority = stage_priority(record)
        scored.append((priority[0], priority[1], index, record))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return scored[0][3]


def select_dense_record(records):
    dense_records = [record for record in records if stage_text(record) == "dense-eval"]
    if not dense_records:
        return None
    return max(dense_records, key=lambda record: to_float(record.get("top1_acc"), default=-math.inf))


def compute_measured_sparsity(record):
    total_params = to_float(record.get("total_params_M"))
    nonzero_params = to_float(record.get("nonzero_params_M"))
    if total_params in (None, 0.0) or nonzero_params is None:
        stored = to_float(record.get("measured_sparsity"))
        return stored
    return 1.0 - (nonzero_params / total_params)


def detect_unstable_flag(records):
    retraining = [record for record in records if stage_text(record) == "retraining" and record.get("top1_acc") is not None]
    retraining.sort(key=lambda record: to_int(record.get("epoch"), default=-1))
    if len(retraining) < 2:
        return 0

    top1_values = [to_float(record.get("top1_acc"), default=0.0) for record in retraining]
    if not top1_values:
        return 0

    spread = max(top1_values) - min(top1_values)
    max_jump = 0.0
    for prev, cur in zip(top1_values, top1_values[1:]):
        max_jump = max(max_jump, abs(cur - prev))

    final_top1 = top1_values[-1]
    peak_top1 = max(top1_values)
    if max_jump >= 3.0:
        return 1
    if spread >= 5.0 and final_top1 < (peak_top1 - 1.0):
        return 1
    return 0


def is_lr_sweep_path(path_str):
    normalized = path_str.replace("\\", "/").lower()
    return "/lr_sweep/" in normalized or "/hparam/" in normalized


def lr_numeric(lr_value):
    value = to_float(lr_value)
    return value if value is not None else math.inf


def sort_lr_group(rows):
    def key(row):
        top1 = to_float(row.get("top1_acc"), default=-math.inf)
        measured = to_float(row.get("measured_sparsity"), default=math.inf)
        target = to_float(row.get("target_sparsity"), default=math.inf)
        sparsity_gap = abs(measured - target) if measured is not None and target is not None else math.inf
        lr_value = lr_numeric(row.get("lr"))
        return (-top1, sparsity_gap, lr_value)

    return sorted(rows, key=key)


def build_summary_row(final_record, dense_record, output_file, unstable_flag):
    row = {column: "" for column in SUMMARY_COLUMNS}
    record_args = final_record.get("args") if isinstance(final_record.get("args"), dict) else {}
    for key in SUMMARY_COLUMNS:
        if key == "output_file":
            continue
        row[key] = get_value(final_record, key, "")

    row["method"] = get_value(final_record, "method", "")
    row["gpadmm_prune_scope"] = get_value(final_record, "gpadmm_prune_scope", "")
    row["sparsity_method"] = get_value(final_record, "sparsity_method", "")
    row["dataset"] = get_value(final_record, "dataset", "")
    row["model"] = get_value(final_record, "model", "")
    row["seed"] = get_value(final_record, "seed", "")
    row["target_sparsity"] = get_value(final_record, "target_sparsity", "")
    row["measured_sparsity"] = compute_measured_sparsity(final_record)
    row["stage"] = final_record.get("stage", "")
    row["epoch"] = final_record.get("epoch", "")
    row["optimizer"] = get_value(final_record, "optimizer", "")
    row["loss"] = final_record.get("loss", "")
    row["top1_acc"] = final_record.get("top1_acc", "")
    row["top5_acc"] = final_record.get("top5_acc", "")
    row["total_params_M"] = final_record.get("total_params_M", "")
    row["nonzero_params_M"] = final_record.get("nonzero_params_M", "")
    row["gflops"] = final_record.get("gflops", "")
    row["lr"] = record_args.get("lr", get_value(final_record, "lr", ""))
    row["rho"] = get_value(final_record, "rho", "")
    row["c"] = get_value(final_record, "c", "")
    row["num_cycles"] = get_value(final_record, "num_cycles", "")
    row["grow_interval"] = get_value(final_record, "grow_interval", "")
    row["num_re_epochs"] = get_value(final_record, "num_re_epochs", "")
    row["is_lr_sweep"] = 1 if is_lr_sweep_path(output_file) else 0
    row["unstable_flag"] = unstable_flag
    row["output_file"] = output_file

    dense_top1 = to_float(dense_record.get("top1_acc")) if dense_record else None
    final_top1 = to_float(final_record.get("top1_acc"))
    if dense_top1 is not None and final_top1 is not None:
        row["delta_top1"] = final_top1 - dense_top1
    else:
        row["delta_top1"] = ""

    return row


def write_csv(path, rows, fieldnames):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob, recursive=True))
    if not paths:
        print(f"[WARN] No files matched: {args.input_glob}")
        write_csv(args.out, [], SUMMARY_COLUMNS)
        if args.best_lr_out:
            write_csv(args.best_lr_out, [], BEST_LR_COLUMNS)
        return

    rows = []
    dense_by_group = {}
    per_file_rows = []

    for path in paths:
        records = list(iter_jsonl(path))
        if not records:
            continue

        dense_record = select_dense_record(records)
        final_record = select_final_record(records)
        unstable_flag = detect_unstable_flag(records)
        if final_record is None:
            continue

        row = build_summary_row(final_record, dense_record, path, unstable_flag)
        per_file_rows.append(row)

        if dense_record is not None:
            dense_key = (
                get_value(dense_record, "dataset", ""),
                get_value(dense_record, "model", ""),
                str(get_value(dense_record, "seed", "")),
            )
            previous = dense_by_group.get(dense_key)
            if previous is None or to_float(dense_record.get("top1_acc"), default=-math.inf) > to_float(previous.get("top1_acc"), default=-math.inf):
                dense_by_group[dense_key] = dense_record

    for row in per_file_rows:
        dense_key = (row["dataset"], row["model"], str(row["seed"]))
        dense_record = dense_by_group.get(dense_key)
        dense_top1 = to_float(dense_record.get("top1_acc")) if dense_record else None
        final_top1 = to_float(row.get("top1_acc"))
        if dense_top1 is not None and final_top1 is not None:
            row["delta_top1"] = final_top1 - dense_top1
        rows.append(row)

    lr_groups = defaultdict(list)
    for row in rows:
        if int(row.get("is_lr_sweep") or 0) != 1:
            continue
        key = (
            row.get("dataset", ""),
            row.get("model", ""),
            str(row.get("target_sparsity", "")),
            row.get("method", ""),
        )
        lr_groups[key].append(row)

    best_lr_rows = []
    for group_rows in lr_groups.values():
        ranked = sort_lr_group(group_rows)
        if not ranked:
            continue
        best_lr = ranked[0].get("lr", "")
        for index, row in enumerate(ranked, start=1):
            row["best_lr_candidate"] = best_lr
            row["rank_within_group"] = index
        best_lr_rows.append(
            {
                "dataset": ranked[0].get("dataset", ""),
                "model": ranked[0].get("model", ""),
                "method": ranked[0].get("method", ""),
                "target_sparsity": ranked[0].get("target_sparsity", ""),
                "recommended_lr": best_lr,
                "top1_acc": ranked[0].get("top1_acc", ""),
                "measured_sparsity": ranked[0].get("measured_sparsity", ""),
                "output_file": ranked[0].get("output_file", ""),
            }
        )

    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("model", ""), str(row.get("seed", "")), str(row.get("target_sparsity", "")), row.get("method", ""), row.get("output_file", "")))
    write_csv(args.out, rows, SUMMARY_COLUMNS)
    print(f"Wrote {len(rows)} rows to {args.out}")

    if args.best_lr_out:
        best_lr_rows.sort(key=lambda row: (row["dataset"], row["model"], str(row["target_sparsity"]), row["method"]))
        write_csv(args.best_lr_out, best_lr_rows, BEST_LR_COLUMNS)
        print(f"Wrote {len(best_lr_rows)} recommended LR rows to {args.best_lr_out}")


if __name__ == "__main__":
    main()
