import argparse
from pathlib import Path
import torch

from utils.loader import load_dataset, load_model, load_optimizer
from methology.admm import admm
from methology.GPadmm import gpadmm
from utils.experiment import setup_experiment, redirect_output_to_log, finish_live_eta, save_checkpoint
from model.pretrain import pretrain
from utils.utils import testAndSave


DEFAULT_CONFIG = {
    "dataset": "mnist",
    "output_dir": "",
    "save_dir": "./runs",
    "run_name": "",
    "data_dir": "./data",
    "save_checkpoint_every": 1,
    "eta_update_interval": 1.0,
    "dense_checkpoint": "",
    "pretrained_checkpoint": "",
    "cifar_pretrained": False,
    "cifar_pretrained_repo": "chenyaofo/pytorch-cifar-models",
    "min_dense_acc": 0.0,
    "method": "admm",
    "model": "lenet",
    "batch_size": 64,
    "test_batch_size": 128,
    "percent": [0.8, 0.92, 0.991, 0.93],
    "alpha": 5e-4,
    "rho": 1e-2,
    "l1": False,
    "l2": False,
    "num_pre_epochs": 3,
    "num_epochs": 10,
    "num_re_epochs": 3,
    "lr": 1e-3,
    "optimizer": "adam",
    "adam_epsilon": 1e-8,
    "momentum": 0.0,
    "dampening": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
    "device": "auto",
    "gpu_id": None,
    "no_cuda": False,
    "seed": 1,
    "use_rigl_admm": False,
    "sparsity": 0.99,
    "sparsity_method": "uniform",
    "gpadmm_prune_scope": "global",
    "min_layer_keep_ratio": 0.01,
    "num_cycles": 3,
    "grow_interval": 5,
    "c": 0.5,
}

DATASET_CHOICES = {"mnist", "cifar10", "cifar100", "imagenet"}
MODEL_CHOICES = {
    "lenet",
    "alexnet",
    "resnet20",
    "resnet56",
    "resnet18",
    "resnet50",
    "vgg16",
    "vgg19",
    "mobilenet_v2",
}
METHOD_CHOICES = {"admm", "admm_layerwise", "gpadmm", "gp_admm", "gpadmm_full", "rigl_admm"}
SPARSITY_METHOD_CHOICES = {"erk", "er", "uniform"}
GPADMM_PRUNE_SCOPE_CHOICES = {"global", "layerwise"}
OPTIMIZER_CHOICES = {"adam", "sgd"}


def _parse_config_scalar(value):
    value = value.strip()
    if value == "":
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {'""', "''"}:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_config_scalar(item.strip()) for item in inner.split(",")]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(text):
    root = {}
    stack = [(-1, root)]

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped_comment = raw_line.split("#", 1)[0].rstrip()
        if not stripped_comment.strip():
            continue
        indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
        content = stripped_comment.strip()
        if ":" not in content:
            raise ValueError(f"Unsupported YAML syntax at line {line_no}: {raw_line}")

        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Missing YAML key at line {line_no}: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML indentation at line {line_no}: {raw_line}")

        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_config_scalar(value)

    return root


def _load_yaml_config(path):
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError:
        with config_path.open("r", encoding="utf-8") as f:
            config = _parse_simple_yaml(f.read())
    else:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return _coerce_loaded_config(config)


def _coerce_loaded_config(value):
    if isinstance(value, str):
        return _parse_config_scalar(value)
    if isinstance(value, dict):
        return {key: _coerce_loaded_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_loaded_config(item) for item in value]
    return value


def _normalize_config_key(key):
    return str(key).replace("-", "_")


def _normalize_config_mapping(mapping):
    return {_normalize_config_key(k): v for k, v in mapping.items()}


def _flatten_experiment_config(config):
    if not config:
        return {}

    blocked_sections = {"common", "admm", "gpadmm", "gp_admm"}
    flat = {}

    common = config.get("common", {})
    if common:
        if not isinstance(common, dict):
            raise ValueError("Config section `common` must be a mapping.")
        flat.update(_normalize_config_mapping(common))

    for key, value in config.items():
        norm_key = _normalize_config_key(key)
        if norm_key in blocked_sections:
            continue
        flat[norm_key] = value

    method = str(flat.get("method", "")).lower()
    if method in {"gpadmm", "gp_admm", "gpadmm_full", "rigl_admm"}:
        flat["use_rigl_admm"] = True
        method_section = config.get("gpadmm", config.get("gp_admm", {}))
    elif method in {"admm", "admm_layerwise", ""}:
        if method:
            flat["use_rigl_admm"] = False
        method_section = config.get("admm", {})
    else:
        raise ValueError(f"Unsupported config method: {flat.get('method')}")

    if method_section:
        if not isinstance(method_section, dict):
            raise ValueError("Method-specific config section must be a mapping.")
        flat.update(_normalize_config_mapping(method_section))

    return flat


def _parse_config_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR with ADMM or RigL+ADMM')
    parser.add_argument('--config', type=str, required=True, help='YAML config file for experiment parameters')
    return parser.parse_args()


def _build_args_from_config(config_path):
    config_values = _flatten_experiment_config(_load_yaml_config(config_path))
    merged = dict(DEFAULT_CONFIG)
    merged.update(config_values)
    args = argparse.Namespace(**merged)
    args.config = config_path
    args.config_values = config_values
    _normalize_method_flags(args)
    _validate_args(args)
    return args


def _normalize_method_flags(args):
    if args.method in {"gpadmm", "gp_admm", "gpadmm_full", "rigl_admm"}:
        args.use_rigl_admm = True
    elif args.method in {"admm", "admm_layerwise"}:
        args.use_rigl_admm = False
    args.method = "gpadmm_full" if args.use_rigl_admm else "admm"


def _validate_args(args):
    if args.dataset not in DATASET_CHOICES:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if args.model not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model: {args.model}")
    if args.method not in {"admm", "gpadmm_full"}:
        raise ValueError(f"Unsupported normalized method: {args.method}")
    if args.sparsity_method not in SPARSITY_METHOD_CHOICES:
        raise ValueError(f"Unsupported sparsity_method: {args.sparsity_method}")
    if args.gpadmm_prune_scope not in GPADMM_PRUNE_SCOPE_CHOICES:
        raise ValueError(f"Unsupported gpadmm_prune_scope: {args.gpadmm_prune_scope}")
    if args.optimizer not in OPTIMIZER_CHOICES:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    if not isinstance(args.percent, (list, tuple)):
        args.percent = [float(args.percent)]
    args.percent = [float(value) for value in args.percent]


def _resolve_device(args):
    requested = str(getattr(args, "device", "auto") or "auto").lower()
    gpu_id = getattr(args, "gpu_id", None)

    if getattr(args, "no_cuda", False):
        requested = "cpu"
    elif requested == "auto":
        if gpu_id is not None:
            requested = f"cuda:{int(gpu_id)}"
        else:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda" and gpu_id is not None:
        requested = f"cuda:{int(gpu_id)}"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device `{requested}`, but CUDA is not available.")
        if ":" in requested:
            index = int(requested.split(":", 1)[1])
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested device `{requested}`, but only {torch.cuda.device_count()} CUDA device(s) are available."
                )

    args.resolved_device = requested
    return torch.device(requested)


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint is not a state_dict or a dict containing model_state_dict/state_dict/model")


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _align_wrapper_prefix(model, state_dict):
    model_keys = model.state_dict().keys()
    if any(k.startswith("backbone.") for k in model_keys) and not any(k.startswith("backbone.") for k in state_dict):
        return {f"backbone.{k}": v for k, v in state_dict.items()}
    return state_dict


def _load_dense_checkpoint(args, model, device):
    checkpoint_path = getattr(args, "dense_checkpoint", "") or getattr(args, "pretrained_checkpoint", "")
    args.checkpoint_loaded = False
    args.checkpoint_path = checkpoint_path
    if not checkpoint_path:
        return model

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Dense checkpoint not found: {path}")

    print(f"[Checkpoint] Loading dense/pretrained checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = _align_wrapper_prefix(model, _strip_module_prefix(_checkpoint_state_dict(checkpoint)))
    model_state = model.state_dict()

    missing_in_checkpoint = sorted(set(model_state) - set(state_dict))
    unexpected_in_checkpoint = sorted(set(state_dict) - set(model_state))
    shape_mismatch = []
    compatible_state = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state[key] = value

    if missing_in_checkpoint:
        print(f"[Checkpoint][WARN] Missing keys in checkpoint: {missing_in_checkpoint}")
    if unexpected_in_checkpoint:
        print(f"[Checkpoint][WARN] Unexpected keys in checkpoint: {unexpected_in_checkpoint}")
    if shape_mismatch:
        print("[Checkpoint][ERROR] Shape mismatches:")
        for key, ckpt_shape, model_shape in shape_mismatch:
            print(f"  - {key}: checkpoint={ckpt_shape}, model={model_shape}")
        raise RuntimeError("Checkpoint shape mismatch. Refusing to continue with a random/mismatched head.")

    incompatible = model.load_state_dict(compatible_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint load produced incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    args.checkpoint_loaded = True
    args.pretrained_loaded = True
    print(f"[Checkpoint] Loaded dense checkpoint successfully: {path}")
    return model


def _requires_cifar_dense_training(args):
    return args.dataset in {"cifar10", "cifar100"} and args.model in {"vgg16", "vgg19", "mobilenet_v2"}


def _validate_dense_ready(args):
    if getattr(args, "checkpoint_loaded", False) or getattr(args, "cifar_pretrained_loaded", False):
        return
    if _requires_cifar_dense_training(args) and args.num_pre_epochs <= 0:
        raise RuntimeError(
            f"{args.dataset}/{args.model} requires a trained dense checkpoint or dense training before pruning. "
            "Pass --dense-checkpoint/--pretrained-checkpoint, or set --num_pre_epochs > 0."
        )


def main():
    cli_args = _parse_config_args()
    args = _build_args_from_config(cli_args.config)
    args.sparsity_was_set = "sparsity" in args.config_values
    args.checkpoint_loaded = False
    args.checkpoint_path = args.dense_checkpoint or args.pretrained_checkpoint
    args.pretrained_loaded = False

    # 랜덤 시드 비활성화
    torch.manual_seed(args.seed)
    device = _resolve_device(args)
    use_cuda = device.type == "cuda"
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 실험 로그/체크포인트 저장 경로 초기화
    setup_experiment(args)
    redirect_output_to_log(args)

    # 데이터셋 로드
    train_loader, test_loader = load_dataset(args, kwargs)


    # 모델 초기화
    model = load_model(args, kwargs).to(device)
    model = _load_dense_checkpoint(args, model, device)

    # 모델별 사전학습 정책 강제 적용
    forced_pre_epochs = {
        "lenet": 3,
        "alexnet": 5,
        "resnet20": 5,
    }

    if getattr(args, "checkpoint_loaded", False):
        # 로컬 dense checkpoint를 사용한 경우 사전학습 스킵
        args.num_pre_epochs = 0
        print(f"[Pretrain] Disabled because dense checkpoint was loaded: {args.checkpoint_path}")
    elif getattr(args, "cifar_pretrained_loaded", False):
        args.num_pre_epochs = 0
        print(f"[Pretrain] Disabled because CIFAR pretrained model was loaded: {args.dataset}/{args.model}")
    elif getattr(args, "torch_pretrained_loaded", False) and args.dataset == "imagenet":
        # ImageNet torchvision pretrained를 사용한 경우 사전학습 스킵
        args.num_pre_epochs = 0
        print(f"[Pretrain] Disabled for ImageNet pretrained model: {args.model}")
    elif args.model in forced_pre_epochs:
        # pretrained가 없는 모델은 모델별로 pre-epoch를 강제 설정
        args.num_pre_epochs = forced_pre_epochs[args.model]
        print(f"[Pretrain] Forced num_pre_epochs={args.num_pre_epochs} for model: {args.model}")
    elif _requires_cifar_dense_training(args):
        print(
            f"[Pretrain] {args.dataset}/{args.model} has a dataset-specific classifier head. "
            f"Will run dense training for {args.num_pre_epochs} epochs before pruning."
        )

    _validate_dense_ready(args)

    # Optimizer 초기화
    base_optimizer_cls = load_optimizer(args, kwargs)

    # Pretrain Phase
    if args.num_pre_epochs > 0:
        model = pretrain(
            args,
            kwargs,
            model,
            device,
            train_loader,
            test_loader,
            base_optimizer_cls,
            total_epochs=args.num_pre_epochs + args.num_epochs + args.num_re_epochs,
        )

    dense_optimizer = base_optimizer_cls(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
    args._extra_metrics = {
        "global_epoch": 0,
        "total_epochs": args.num_epochs + args.num_re_epochs,
        "stage": "dense-eval",
    }
    dense_metrics = testAndSave(args, model, device, test_loader, "Dense Model Test", optimizer=dense_optimizer)
    save_checkpoint(args, model, dense_optimizer, stage="dense_eval", stage_epoch=0,
                    global_epoch=1, metrics=dense_metrics)

    min_dense_acc = max(float(getattr(args, "min_dense_acc", 0.0)), 20.0 if _requires_cifar_dense_training(args) else 0.0)
    if min_dense_acc > 0.0 and dense_metrics["top1_acc"] < min_dense_acc:
        message = (
            f"[Dense][ERROR] {args.dataset}/{args.model} dense Top-1 is {dense_metrics['top1_acc']:.2f}%, "
            f"below required minimum {min_dense_acc:.2f}%. "
            "Stopping before pruning because this is not a trustworthy pruning baseline."
        )
        print(message)
        raise RuntimeError(message)

    try:
        if args.use_rigl_admm:
            gpadmm(args, model, device, train_loader, test_loader, base_optimizer_cls)
        else: # 기존 ADMM 파이프 라인
            admm(args, model, device, train_loader, test_loader, base_optimizer_cls)
    finally:
        finish_live_eta(args)

if __name__ == "__main__":
    main()
