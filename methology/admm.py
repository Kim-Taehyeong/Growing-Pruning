import torch
import torch.nn.functional as F
import time

from utils.utils import admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune, testAndSave
from utils.experiment import save_checkpoint, save_final_model, update_live_eta


_DEFAULT_PERCENT = [0.8, 0.92, 0.991, 0.93]


def _model_percent_template(model_name):
    templates = {
        "lenet": [0.0, 0.65, 0.85, 0.90],
        "alexnet": [0.0, 0.50, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95],
        "resnet20": [0.0, 0.20, 0.28, 0.35, 0.42, 0.50, 0.58, 0.66, 0.72, 0.78, 0.84, 0.88],
        "resnet18": [0.0, 0.25, 0.33, 0.40, 0.48, 0.55, 0.62, 0.68, 0.74, 0.80, 0.85],
        "resnet50": [0.0, 0.20, 0.28, 0.35, 0.42, 0.50, 0.58, 0.65, 0.72, 0.78, 0.83, 0.88],
        "vgg19": [0.0, 0.35, 0.45, 0.55, 0.62, 0.68, 0.74, 0.79, 0.84, 0.88, 0.92, 0.95],
        "mobilenet_v2": [0.0, 0.25, 0.35, 0.45, 0.55, 0.63, 0.70, 0.76, 0.82, 0.87, 0.91],
    }
    return templates.get(model_name)


def _resize_percent_template(template, target_len):
    if target_len <= 0:
        return []
    if len(template) == 1:
        return [template[0]] * target_len
    if target_len == 1:
        return [template[0]]

    src_last = len(template) - 1
    out = []
    for i in range(target_len):
        t = float(i) / float(target_len - 1)
        pos = t * src_last
        left = int(pos)
        right = min(left + 1, src_last)
        w = pos - left
        out.append((1.0 - w) * template[left] + w * template[right])
    return out


def _prepare_admm_percent(args, model):
    prunable_layers = [
        name for name, param in model.named_parameters()
        if name.split('.')[-1] == "weight" and param.dim() >= 2
    ]
    n_layers = len(prunable_layers)

    raw = getattr(args, "percent", None)
    if isinstance(raw, (int, float)):
        percents = [float(raw)]
    elif isinstance(raw, (list, tuple)):
        percents = [float(p) for p in raw]
    else:
        percents = []

    # 사용자 지정이 없거나 기본값이면 모델 템플릿 우선 적용
    use_template = (not percents) or (list(percents) == _DEFAULT_PERCENT)
    template = _model_percent_template(getattr(args, "model", "")) if use_template else None

    if template:
        percents = _resize_percent_template(template, n_layers)
        print(f"[ADMM] Using model template percent for {args.model} (layers={n_layers})")
    elif not percents:
        fallback = float(getattr(args, "sparsity", 0.9))
        percents = [fallback]

    if len(percents) < n_layers:
        fill = percents[-1]
        percents = percents + [fill] * (n_layers - len(percents))
    elif len(percents) > n_layers:
        percents = percents[:n_layers]

    args.percent = [min(0.9999, max(0.0, float(p))) for p in percents]


def admm(args, model, device, train_loader, test_loader, base_optimizer_cls):
    _prepare_admm_percent(args, model)
    optimizer = base_optimizer_cls(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
    total_epochs = args.num_epochs + args.num_re_epochs
    global_epoch = _train_admm(args, model, device, train_loader, test_loader, optimizer, total_epochs)

    mask = apply_l1_prune(model, device, args) if args.l1 else apply_prune(model, device, args)
    print_prune(model, args)
    args._extra_metrics = {"global_epoch": global_epoch, "total_epochs": total_epochs, "stage": "post-pruning"}
    post_metrics = testAndSave(args, model, device, test_loader, "ADMM-Post-Pruning", optimizer=optimizer)
    save_checkpoint(args, model, optimizer, stage="admm_post_pruning", stage_epoch=0,
                    global_epoch=max(global_epoch, 1), metrics=post_metrics)

    for epoch in range(args.num_re_epochs):
        print('Re epoch: {}'.format(epoch + 1))
        model.train()
        running_loss = 0.0
        num_batches = max(len(train_loader), 1)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if args.dataset == "imagenet":
                loss = F.cross_entropy(output, target)
            else:
                loss = F.nll_loss(output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.prune_step(mask)

            completed = global_epoch + float(batch_idx + 1) / float(num_batches)
            update_live_eta(args, completed, total_epochs, stage="admm-retrain")
        global_epoch += 1
        elapsed = time.time() - args.experiment["started_at"]
        eta = 0.0
        if global_epoch > 0:
            eta = (elapsed / global_epoch) * max(total_epochs - global_epoch, 0)
        update_live_eta(args, global_epoch, total_epochs, stage="admm-retrain")
        args._extra_metrics = {
            "train_loss": running_loss / max(len(train_loader), 1),
            "global_epoch": global_epoch,
            "total_epochs": total_epochs,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "stage": "retraining",
        }
        metrics = testAndSave(args, model, device, test_loader, "ADMM-Re-Training", epoch, optimizer=optimizer)
        save_checkpoint(args, model, optimizer, stage="admm_retrain", stage_epoch=epoch + 1,
                        global_epoch=global_epoch, metrics=metrics)

    save_final_model(args, model, optimizer=optimizer, tag="admm_final")


def _train_admm(args, model, device, train_loader, test_loader, optimizer, total_epochs):
    global_epoch = 0

    Z, U = initialize_Z_and_U(model)
    for epoch in range(args.num_epochs):
        model.train()
        print('Epoch: {}'.format(epoch + 1))
        running_loss = 0.0
        num_batches = max(len(train_loader), 1)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = admm_loss(args, device, model, Z, U, output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            completed = global_epoch + float(batch_idx + 1) / float(num_batches)
            update_live_eta(args, completed, total_epochs, stage="admm-main")
        X = update_X(model)
        Z = update_Z_l1(X, U, args) if args.l1 else update_Z(X, U, args)
        U = update_U(U, X, Z)
        print_convergence(model, X, Z)
        global_epoch += 1
        elapsed = time.time() - args.experiment["started_at"]
        eta = 0.0
        if global_epoch > 0:
            eta = (elapsed / global_epoch) * max(total_epochs - global_epoch, 0)
        update_live_eta(args, global_epoch, total_epochs, stage="admm-main")
        args._extra_metrics = {
            "train_loss": running_loss / max(len(train_loader), 1),
            "global_epoch": global_epoch,
            "total_epochs": total_epochs,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "stage": "main-training",
        }
        metrics = testAndSave(args, model, device, test_loader, "ADMM-Train", epoch, optimizer=optimizer)
        save_checkpoint(args, model, optimizer, stage="admm_train", stage_epoch=epoch + 1,
                        global_epoch=global_epoch, metrics=metrics)

    return global_epoch