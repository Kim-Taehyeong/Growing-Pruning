from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from optimizer import PruneAdam
from model import LeNet, AlexNet
from utils import apply_prune_random, regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune, apply_prune_rigl
from torchvision import datasets, transforms
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score


# ==============================================================
# 학습에 마스크를 적용할 "가중치 파라미터"만 수집
# ==============================================================
def _collect_maskable_params(model):
    named = []
    for name, p in model.named_parameters():
        if p.dim() >= 2 and name.endswith("weight"):
            named.append((name, p))
    return named

# ==============================================================
# ### 파라미터 별 IN/OUT Shape Info 저장
# ==============================================================
def _layer_shape_info(param):
    shape = tuple(param.shape)
    if len(shape) == 4:  # Conv
        out_c, in_c, kH, kW = shape
        n_in = in_c * kH * kW
        n_out = out_c
        k2 = kH * kW
        N = out_c * in_c * kH * kW
        return dict(kind="conv", n_in=n_in, n_out=n_out, k2=k2, N=N)
    elif len(shape) == 2:  # Linear
        out_f, in_f = shape
        n_in = in_f
        n_out = out_f
        k2 = 1
        N = out_f * in_f
        return dict(kind="linear", n_in=n_in, n_out=n_out, k2=k2, N=N)
    else:
        raise ValueError(f"Unsupported shape for masking: {shape}")

# ==============================================================
# ### ER/ERK 초기 희소화 분배
#   * ER  : p_l ∝ (n_in + n_out) / (n_in * n_out)
#   * ERK : p_l ∝ (n_in + n_out) / (n_in * n_out * k^2)
# - 글로벌 sparsity(예: 0.9) → 레이어별 density로 변환
# ==============================================================
def _allocate_density_erk_er(params, target_sparsity, method="erk"):
    assert 0.0 <= target_sparsity < 1.0
    infos, total_N = {}, 0
    for name, p in params:
        info = _layer_shape_info(p)
        infos[name] = info
        total_N += info["N"]
    target_nonzero = int(round((1.0 - target_sparsity) * total_N))

    f = {}
    for name, _ in params:
        info = infos[name]
        if method == "erk":
            f[name] = (info["n_in"] + info["n_out"]) / (info["n_in"] * info["n_out"] * info["k2"])
        else:  # "er"
            f[name] = (info["n_in"] + info["n_out"]) / (info["n_in"] * info["n_out"])

    # 이분 탐색으로 eps를 찾아 Sparsity에 근사하도록 조정
    lo, hi = 0.0, 1e9
    for _ in range(60):
        mid = (lo + hi) / 2.0
        nonzeros = 0.0
        for name, _ in params:
            p_l = min(1.0, mid * f[name])
            nonzeros += p_l * infos[name]["N"]
        if nonzeros > target_nonzero:
            hi = mid
        else:
            lo = mid
    eps = lo

    densities = {}
    for name, _ in params:
        densities[name] = min(1.0, eps * f[name])
    return densities

# ==============================================================
# ### 마스크 초기화 및 ER, ERK Sparsity 계산
# ==============================================================
def initialize_masks(model, target_sparsity=0.9, method="erk", device="cpu"):
    maskable = _collect_maskable_params(model)
    densities = _allocate_density_erk_er(maskable, target_sparsity, method)
    masks = {}
    with torch.no_grad():
        for name, p in maskable:
            N = p.numel()
            keep = int(round(densities[name] * N))
            if keep <= 0:
                mask = torch.zeros_like(p, dtype=torch.bool)
            elif keep >= N:
                mask = torch.ones_like(p, dtype=torch.bool)
            else:
                idx = torch.randperm(N, device=p.device)[:keep]
                flat = torch.zeros(N, device=p.device, dtype=torch.bool)
                flat[idx] = True
                mask = flat.view_as(p)
            masks[name] = mask.to(device)
            p.data.mul_(masks[name].to(p.device))  # pruned=0
    return masks, densities


def enforce_weight_mask(model, masks):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.data.mul_(masks[name].to(p.device))


def mask_grads(model, masks):
    for name, p in model.named_parameters():
        if name in masks and p.grad is not None:
            p.grad.mul_(masks[name].to(p.device))


# ==============================================================
# ### Growing에 활용할 그래드언트 수집 (Default : Step 50)
# ==============================================================
def collect_grad_for_growth(model, device, data_loader, steps=50):
    model.train()
    grad_buffers = {}
    seen = 0
    it = iter(data_loader)
    while seen < steps:
        try:
            data, target = next(it)
        except StopIteration:
            it = iter(data_loader)
            data, target = next(it)
        data, target = data.to(device), target.to(device)
        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if name not in grad_buffers:
                grad_buffers[name] = p.grad.detach().abs().clone()
            else:
                grad_buffers[name] += p.grad.detach().abs()
        seen += 1
    # 평균 |grad|
    for k in list(grad_buffers.keys()):
        grad_buffers[k] /= float(steps)
    return grad_buffers


# ==============================================================
# ### TopK (Grow_Frac) 비율 만큼 Growing
# ==============================================================
def rigl_grow_once(model, masks, grow_frac, grad_buffers):
    # 현재 활성 수 기준으로 성장 개수 결정
    active_total = sum(m.sum().item() for m in masks.values())
    g = int(max(1, round(active_total * float(grow_frac))))
    if g == 0:
        return masks

    # pruned 위치의 |grad| 풀링
    scores_all, refs = [], []
    for name, p in model.named_parameters():
        if name not in masks:
            continue
        m = masks[name].view(-1).bool()
        pruned = (~m)
        if not pruned.any():
            continue
        grad_abs = grad_buffers.get(name, None)
        if grad_abs is None:
            continue
        grad_abs = grad_abs.view(-1)
        sel = grad_abs[pruned]
        if sel.numel() == 0:
            continue
        scores_all.append(sel)
        flat_idx = torch.nonzero(pruned, as_tuple=False).view(-1)
        refs += [(name, int(i.item())) for i in flat_idx]

    if len(scores_all) == 0:
        return masks

    scores_all = torch.cat(scores_all, dim=0)
    g = min(g, scores_all.numel())
    _, topk_idx = torch.topk(scores_all, k=g, largest=True, sorted=False)
    chosen = [refs[int(i)] for i in topk_idx.tolist()]

    with torch.no_grad():
        for (name, flat_i) in chosen:
            mask = masks[name].view(-1)
            if not mask[flat_i]:
                mask[flat_i] = True
                masks[name] = mask.view_as(masks[name])
                # 가중치 0으로 초기화
                p = dict(model.named_parameters())[name]
                p.view(-1)[flat_i] = 0.0

    # 마스크 적용
    enforce_weight_mask(model, masks)
    return masks


# ==============================================================
# ### [RIGL+ADMM ADDED] 현재 모델에서 "마스크 복원" (apply_prune 이후 사용)
# - 요구사항: ADMM 프루닝이 끝난 뒤, 다음 사이클에서 쓸 마스크를 모델에서 재생성
# ==============================================================
def rebuild_masks_from_weights(model, device="cpu"):
    masks = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2 and name.endswith("weight"):
                masks[name] = (p.data != 0).to(device)
    return masks


# ==============================================================
# ### ADMM Pruning 사이클 (Interval 크기만큼 Epoch 반복)
# ==============================================================
def admm_prune_stage(args, model, device, train_loader, test_loader, masks, optimizer, densities, epochs, cycle):
    Z, U = initialize_Z_and_U(model)
    for epoch in range(epochs):
        print(f'[RigL+ADMM] ADMM epoch {epoch+1}/{epochs}')
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = admm_loss(args, device, model, Z, U, output, target)
            loss.backward()
            mask_grads(model, masks)
            optimizer.step()
            enforce_weight_mask(model, masks)  # pruned=0 유지
        # ADMM 업데이트(에폭 말)
        X = update_X(model)
        Z = update_Z_l1(X, U, args) if args.l1 else update_Z(X, U, args)
        U = update_U(U, X, Z)
        print_convergence(model, X, Z)
        testAndSave(args, model, device, test_loader, f"ADMM-Cycle {cycle}", epoch)

    # 목표 희소도까지 프루닝
    if args.init_method == "random":
        if args.l1:
            masks = apply_l1_prune(model, device, args)
        else:
            masks = apply_prune(model, device, args)
    else:
        masks = apply_prune_rigl(model, device, densities)

    # 프루닝 후 모델에서 새 마스크 복원 (다음 사이클에 사용)
    new_masks = rebuild_masks_from_weights(model, device=device)
    print_prune(model, args)
    testAndSave(args, model, device, test_loader, f"ADMM-Cycle {cycle} Post-Pruning")
    return new_masks


# ==============================================================
# 기존 ADMM 파이프라인(원본 유지)
# ==============================================================
def train_admm_full(args, model, device, train_loader, test_loader, optimizer):
    for epoch in range(args.num_pre_epochs):
        print('Pre epoch: {}'.format(epoch + 1))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = regularized_nll_loss(args, model, output, target)
            loss.backward()
            optimizer.step()
        testAndSave(args, model, device, test_loader, "ADMM-Pretraining", epoch)

    Z, U = initialize_Z_and_U(model)
    for epoch in range(args.num_epochs):
        model.train()
        print('Epoch: {}'.format(epoch + 1))
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = admm_loss(args, device, model, Z, U, output, target)
            loss.backward()
            optimizer.step()
        X = update_X(model)
        Z = update_Z_l1(X, U, args) if args.l1 else update_Z(X, U, args)
        U = update_U(U, X, Z)
        print_convergence(model, X, Z)
        testAndSave(args, model, device, test_loader, "ADMM-Train", epoch)


def testAndSave(args, model, device, test_loader, prefix, epoch=None):
    model.eval()
    test_loss, correct, correct_top5 = 0.0, 0, 0
    all_preds, all_targets = [], []

    if not args.output_dir:
        args.output_dir = "./metrics.json"

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Top-1
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Top-5
            top5 = output.topk(5, dim=1)[1]
            correct_top5 += top5.eq(target.view(-1,1)).sum().item()

            # Save preds for precision/recall/f1
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 평균 loss
    test_loss /= len(test_loader.dataset)

    # Accuracy
    acc1 = 100. * correct / len(test_loader.dataset)
    acc5 = 100. * correct_top5 / len(test_loader.dataset)

    # Precision / Recall / F1
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    # === JSON 저장 ===
    metrics = {
        "prefix" : prefix,
        "epoch": epoch if epoch is not None else -1,
        "loss": test_loss,
        "top1_acc": acc1,
        "top5_acc": acc5,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "args": vars(args) if args is not None else {}
    }

    json_file = Path(args.output_dir)
    with open(json_file, mode="a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"[Test] Epoch {metrics['epoch']} | "
          f"Loss {test_loss:.4f}, Top1 {acc1:.2f}%, Top5 {acc5:.2f}%, "
          f"P {precision:.3f}, R {recall:.3f}, F1 {f1:.3f}")

    return metrics


# ==============================================================
# ADMM + RigL 학습 파이프라인
# ==============================================================
def rigl_admm_cycle_train(args, model, device, train_loader, test_loader, base_optimizer_cls):
    optimizer = base_optimizer_cls(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)

    # 사전 학습
    for epoch in range(args.num_pre_epochs):
        print('Pre epoch: {}'.format(epoch + 1))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = regularized_nll_loss(args, model, output, target)
            loss.backward()
            optimizer.step()
        testAndSave(args, model, device, test_loader, "Pretraining", epoch)

    # (0) 초기 희소 모델 생성(ER/ERK)
    densities = None
    if args.init_method == "random":
        masks = apply_prune_random(model, device, args)
    else:
        masks, densities = initialize_masks(
            model,
            target_sparsity=args.sparsity,
            method=args.init_method,
            device=device
        )
    enforce_weight_mask(model, masks)


    # 사이클 반복
    for c in tqdm(range(args.num_cycles)):
        print(f'[RigL+ADMM] Cycle {c+1}/{args.num_cycles}')

        # (1) 성장용 |grad| 수집 (업데이트 없이 평균)
        grad_buf = collect_grad_for_growth(model, device, train_loader, steps=args.grow_grad_steps)

        # (2) RigL 성장: pruned 중 |grad| 상위 → 활성화(가중치 0으로 초기화)
        masks = rigl_grow_once(model, masks, grow_frac=args.grow_frac, grad_buffers=grad_buf)

        # (3) ADMM 프루닝 스테이지: 짧게 학습 후 apply_prune로 목표 희소율 달성
        masks = admm_prune_stage(args, model, device, train_loader, test_loader, masks,
                                 optimizer, densities, epochs=args.grow_interval, cycle=c + 1)

    print('[RigL+ADMM] Final fixed-mask retraining...')

    # (4) 최종 재학습(마스크 고정, 성장/프루닝 없음)
    for epoch in range(args.num_re_epochs):
        print(f'[RigL+ADMM] Re epoch: {epoch+1}/{args.num_re_epochs}')
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            mask_grads(model, masks)     # pruned 고정
            optimizer.step()
            enforce_weight_mask(model, masks)
        testAndSave(args, model, device, test_loader, "ADMM-Re-Training", epoch)

    print_prune(model, args)
    return masks


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR with ADMM or RigL+ADMM')

    # Dataset
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "cifar10"], metavar='D')
    parser.add_argument('--output-dir', type=str, default="", metavar='O', help='Directory to save metrics JSON files')

    # Batches
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')

    # 기존 ADMM(풀런) 파라미터
    parser.add_argument('--percent', type=list, default=[0.8, 0.92, 0.991, 0.93], metavar='P',
                        help='(ADMM) per-layer pruning percentage list')
    parser.add_argument('--alpha', type=float, default=5e-4, metavar='L')
    parser.add_argument('--rho', type=float, default=1e-2, metavar='R')
    parser.add_argument('--l1', action='store_true', default=False,
                        help='use l1 ADMM regularization/pruning (instead of cardinality)')
    parser.add_argument('--l2', action='store_true', default=False)
    parser.add_argument('--num_pre_epochs', type=int, default=3, metavar='P')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N')
    parser.add_argument('--num_re_epochs', type=int, default=3, metavar='R')

    # 공통 옵티마이저
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--save-model', action='store_true', default=False)

    # RigL + ADMM 파라미터
    parser.add_argument('--use-rigl-admm', action='store_true', default=False,
                        help='enable Grow (RigL) -> ADMM pruning cycles')
    parser.add_argument('--sparsity', type=float, default=0.98, help='global target sparsity (0~1)')
    parser.add_argument('--init-method', type=str, default='random', choices=['erk', 'er', 'random'])
    parser.add_argument('--num-cycles', type=int, default=3)
    parser.add_argument('--grow-interval', type=int, default=5)
    parser.add_argument('--grow-frac', type=float, default=0.1)
    parser.add_argument('--grow-grad-steps', type=int, default=50)

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # 랜덤 시드 비활성화
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        
    # elif args.dataset == "imagenet":
    #     train_tf = transforms.Compose([
    #         transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    #     ])
    #     val_tf = transforms.Compose([
    #         transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    #     ])
    #     train_loader = torch.utils.data.DataLoader(
    #         datasets.ImageFolder(args.imagenet_train, transform=train_tf),
    #         batch_size=args.batch_size, shuffle=True, **kwargs)
    #     test_loader = torch.utils.data.DataLoader(
    #         datasets.ImageFolder(args.imagenet_val, transform=val_tf),
    #         batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        args.percent = [0.8, 0.92, 0.93, 0.94, 0.95, 0.99, 0.99, 0.93]
        # args.num_pre_epochs = 5
        # args.num_epochs = 20
        # args.num_re_epochs = 5
        # args.num_cycles = 4
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.batch_size, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.test_batch_size, **kwargs)

    model = LeNet().to(device) if args.dataset == "mnist" else AlexNet().to(device)
    BaseOpt = PruneAdam

    if args.use_rigl_admm:
        rigl_admm_cycle_train(args, model, device, train_loader, test_loader, BaseOpt)
    else:
        optimizer = BaseOpt(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
        train_admm_full(args, model, device, train_loader, test_loader, optimizer)
        mask = apply_l1_prune(model, device, args) if args.l1 else apply_prune(model, device, args)
        print_prune(model, args)
        testAndSave(args, model, device, test_loader, "ADMM-Post-Pruning")
        for epoch in range(args.num_re_epochs):
            print('Re epoch: {}'.format(epoch + 1))
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.prune_step(mask)
            testAndSave(args, model, device, test_loader, "ADMM-Re-Training", epoch)


if __name__ == "__main__":
    main()
