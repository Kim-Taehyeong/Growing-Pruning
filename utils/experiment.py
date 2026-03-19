import time
import sys
import atexit
from datetime import datetime
from pathlib import Path

import torch


def setup_experiment(args):
    mode = "gpadmm" if getattr(args, "use_rigl_admm", False) else "admm"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = getattr(args, "run_name", "") or f"{mode}_{args.dataset}_{args.model}_{timestamp}"

    base_dir = Path(getattr(args, "save_dir", "./runs")) / run_name
    ckpt_dir = base_dir / "checkpoints"
    model_dir = base_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    output_dir = getattr(args, "output_dir", "")
    if output_dir:
        metrics_path = Path(output_dir)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        metrics_path = base_dir / "metrics.jsonl"
        args.output_dir = str(metrics_path)

    exp = {
        "run_name": run_name,
        "base_dir": str(base_dir),
        "checkpoints_dir": str(ckpt_dir),
        "models_dir": str(model_dir),
        "log_file": str(base_dir / "run.log"),
        "started_at": time.time(),
    }
    args.experiment = exp
    return exp


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def redirect_output_to_log(args):
    """Redirect stdout/stderr to run.log; keep original stdout for live ETA line."""
    exp = args.experiment
    log_fp = open(exp["log_file"], "a", encoding="utf-8", buffering=1)

    exp["_orig_stdout"] = sys.stdout
    exp["_orig_stderr"] = sys.stderr
    exp["_console_stream"] = sys.__stdout__
    exp["_log_fp"] = log_fp

    sys.stdout = log_fp
    sys.stderr = log_fp

    def _cleanup():
        try:
            finish_live_eta(args)
        except Exception:
            pass
        try:
            sys.stdout = exp.get("_orig_stdout", sys.__stdout__)
            sys.stderr = exp.get("_orig_stderr", sys.__stderr__)
        except Exception:
            pass
        try:
            fp = exp.get("_log_fp")
            if fp and not fp.closed:
                fp.flush()
                fp.close()
        except Exception:
            pass

    atexit.register(_cleanup)


def update_live_eta(args, completed_epochs, total_epochs, stage="train"):
    min_interval = float(getattr(args, "eta_update_interval", 1.0))
    now = time.time()
    last = float(args.experiment.get("_eta_last_ts", 0.0))
    if (now - last) < min_interval:
        # Too soon to refresh terminal line; still return current estimate.
        elapsed = now - args.experiment["started_at"]
        eta = 0.0
        if completed_epochs > 0:
            eta = (elapsed / completed_epochs) * max(total_epochs - completed_epochs, 0)
        return elapsed, eta

    args.experiment["_eta_last_ts"] = now
    elapsed = time.time() - args.experiment["started_at"]
    eta = 0.0
    if completed_epochs > 0:
        eta = (elapsed / completed_epochs) * max(total_epochs - completed_epochs, 0)

    # 고정 폭 포맷으로 길이 변화(소수점/사이클명)로 인한 잔문자 발생을 최소화
    line = (
        f"\r[ETA] {stage:<18} | "
        f"{float(completed_epochs):7.2f}/{float(total_epochs):7.2f} | "
        f"elapsed {_format_duration(elapsed)} | eta {_format_duration(eta)}"
    )
    prev_len = int(args.experiment.get("_eta_prev_len", 0))
    cur_len = len(line) - 1  # '\r' 제외
    if prev_len > cur_len:
        line += " " * (prev_len - cur_len)
    args.experiment["_eta_prev_len"] = max(prev_len, cur_len)
    console = args.experiment.get("_console_stream", sys.__stdout__)
    console.write(line)
    console.flush()
    return elapsed, eta


def finish_live_eta(args):
    console = args.experiment.get("_console_stream", sys.__stdout__)
    prev_len = int(args.experiment.get("_eta_prev_len", 0))
    if prev_len > 0:
        console.write("\r" + (" " * prev_len) + "\r")
    console.write("\n")
    console.flush()


def log_epoch_eta(start_time, completed_epochs, total_epochs, prefix):
    elapsed = time.time() - start_time
    eta = 0.0
    if completed_epochs > 0:
        avg = elapsed / completed_epochs
        eta = avg * max(total_epochs - completed_epochs, 0)

    print(
        f"[{prefix}] Elapsed {_format_duration(elapsed)} | "
        f"ETA {_format_duration(eta)} | Progress {completed_epochs}/{total_epochs}"
    )
    return elapsed, eta


def save_checkpoint(args, model, optimizer, stage, stage_epoch, global_epoch, metrics=None):
    every = int(getattr(args, "save_checkpoint_every", 1))
    if every <= 0 or (global_epoch % every) != 0:
        return None

    checkpoints_dir = Path(args.experiment["checkpoints_dir"])
    checkpoint_path = checkpoints_dir / f"ep_{global_epoch:04d}_{stage}.pt"

    state = {
        "stage": stage,
        "stage_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics or {},
    }
    torch.save(state, checkpoint_path)
    print(f"[Checkpoint] Saved: {checkpoint_path}")
    return str(checkpoint_path)


def save_final_model(args, model, optimizer=None, tag="final"):
    model_dir = Path(args.experiment["models_dir"])
    final_path = model_dir / f"{tag}_model.pt"

    state = {
        "tag": tag,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(state, final_path)
    print(f"[Model] Saved final model: {final_path}")
    return str(final_path)
