import torch
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
from thop import profile
from scipy import optimize as opt



def regularized_nll_loss(args, model, output, target):
    index = 0
    if args.dataset == "imagenet":
        loss = F.cross_entropy(output, target)
    else:
        loss = F.nll_loss(output, target)
    if args.l2:
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight" and param.dim() >= 2:
                loss += args.alpha * param.norm()
                index += 1
    return loss


def admm_loss(args, device, model, Z, U, output, target):
    idx = 0
    if args.dataset == "imagenet":
        loss = F.cross_entropy(output, target)
    else:
        loss = F.nll_loss(output, target)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            u = U[idx].to(device)
            z = Z[idx].to(device)
            loss += args.rho / 2 * (param - z + u).norm()
            if args.l2:
                loss += args.alpha * param.norm()
            idx += 1
    return loss


def initialize_Z_and_U(model):
    Z = ()
    U = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            Z += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
    return Z, U


def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            X += (param.detach().cpu().clone(),)
    return X


def update_Z(X, U, args):
    new_Z = ()
    idx = 0
    for x, u in zip(X, U):
        z = x + u
        pcen = np.percentile(abs(z), 100*args.percent[idx])
        under_threshold = abs(z) < pcen
        z.data[under_threshold] = 0
        new_Z += (z,)
        idx += 1
    return new_Z


def update_Z_l1(X, U, args):
    new_Z = ()
    delta = args.alpha / args.rho
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if (z > delta).sum() != 0:
            new_z[z > delta] = z[z > delta] - delta
        if (z < -delta).sum() != 0:
            new_z[z < -delta] = z[z < -delta] + delta
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z += (new_z,)
    return new_Z


def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
    return new_U


def prune_weight(weight, device, percent):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    pcen = np.percentile(abs(weight_numpy), 100*percent)
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask

def prune_weight_random(weight, device, percent, generator: torch.Generator):
    p = float(max(0.0, min(1.0, percent)))
    N = weight.numel()
    k_prune = int(round(p * N))

    mask_flat = torch.ones(N, device=device, dtype=weight.dtype)

    if k_prune <= 0:
        return mask_flat.view_as(weight)
    if k_prune >= N:
        return torch.zeros_like(weight, dtype=weight.dtype, device=device)

    idx = torch.randperm(N, device=device, generator=generator)[:k_prune]
    mask_flat[idx] = 0
    return mask_flat.view_as(weight)

def prune_l1_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask


def apply_prune(model, device, args):
    # returns dictionary of non_zero_values' indices
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            print(name, idx)
            mask = prune_weight(param, device, args.percent[idx])
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def apply_prune_random(model, device, args):
    print("Apply Pruning based Random")
    dict_mask = {}
    idx = 0

    # Fix Seed
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            mask = prune_weight_random(param, device, args.percent[idx], gen)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def apply_prune_rigl(model, device, densities):
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_weight(param, device, 1 - densities[name])
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def apply_l1_prune(model, device, args):
    delta = args.alpha / args.rho
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def print_convergence(model, X, Z):
    idx = 0
    print("normalized norm of (weight - projection)")
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            x, z = X[idx], Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
            idx += 1


def print_prune(model, args):
    prune_param, total_param = 0, 0
    json_dict = {}
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if name.split('.')[-1] == "weight":
            json_dict[name] = {
                "percentage" : 100 * (abs(param) == 0).sum().item() / param.numel(),
                "nonzero parameters" : "{} / {}".format((param != 0).sum().item(), param.numel())
            }
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    json_dict["total"] = {
        "percentage" : 100 * (total_param - prune_param) / total_param,
        "nonzero parameters" : "{} / {}".format(prune_param, total_param)
    }
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))
    
    with open(args.output_dir, 'a') as f:
        json.dump(json_dict, f)
        f.write("\n")

# 온라인 |grad| EMA 누적기
def online_accumulate_grad_ema(model, grad_ma, beta=0.95, include_bias=False):
    # grad_ma: dict[name] -> tensor (EMA 상태)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            if (not include_bias) and name.endswith(".bias"):
                continue  # conv/linear의 weight 위주로

            g = p.grad.detach().abs()
            if name not in grad_ma:
                grad_ma[name] = g.clone()
            else:
                # EMA: m = beta*m + (1-beta)*|g|
                grad_ma[name].mul_(beta).add_(g, alpha=(1.0 - beta))

def enforce_weight_mask(model, masks):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.data.mul_(masks[name].to(p.device))

def mask_grads(model, masks):
    for name, p in model.named_parameters():
        if name in masks and p.grad is not None:
            p.grad.mul_(masks[name].to(p.device))



def _count_sparse_flops(model, sample_input):
    """
    Forward hooks를 사용하여 희소 모델(pruned model)의 FLOPs를 계산하는 함수.
    가중치가 0인 연산은 FLOPs 계산에서 제외합니다.

    NOTE: 이 계산은 근사치이며, 모든 종류의 레이어를 지원하지 않을 수 있습니다.
          주로 Conv2d와 Linear 레이어에 초점을 맞춥니다.
    """
    total_flops = 0

    def multiply_adds_hook(module, input, output):
        nonlocal total_flops
        # torch.nn.Linear
        if isinstance(module, torch.nn.Linear):
            batch_size = input[0].shape[0] if input[0].dim() > 1 else 1
            nonzero_weights = torch.count_nonzero(module.weight)
            # 가중치에 대한 곱셈-누산(Multiply-Accumulate) 연산
            macs = nonzero_weights * batch_size
            total_flops += 2 * macs # 1 MAC = 2 FLOPs (곱셈 1, 덧셈 1)
            # Bias 덧셈 연산
            if module.bias is not None:
                total_flops += module.bias.numel() * batch_size

        # torch.nn.Conv2d
        elif isinstance(module, torch.nn.Conv2d):
            batch_size, _, output_h, output_w = output.shape
            nonzero_weights = torch.count_nonzero(module.weight)
            # 커널의 0이 아닌 가중치에 대한 MACs
            macs = output_h * output_w * (nonzero_weights / module.groups) * batch_size
            total_flops += 2 * macs # 1 MAC = 2 FLOPs
            # Bias 덧셈 연산
            if module.bias is not None:
                total_flops += module.bias.numel() * output_h * output_w * batch_size

    hooks = []
    # 모든 Conv2d와 Linear 레이어에 hook 등록
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(multiply_adds_hook))

    # FLOPs 계산을 위해 dummy forward pass 실행
    with torch.no_grad():
        model(sample_input)

    # 등록된 hook 제거
    for hook in hooks:
        hook.remove()

    return total_flops

def testAndSave(args, model, device, test_loader, prefix, epoch=None, optimizer=None):
    """
    모델을 테스트하고 성능 지표를 계산하여 JSON 파일에 저장하는 함수.
    파라미터 수와 GFLOPs 계산 기능이 추가되었습니다.
    """
    model.eval()
    test_loss, correct, correct_top5 = 0.0, 0, 0
    all_preds, all_targets = [], []

    if not args.output_dir:
        args.output_dir = "./metrics.json"

    # --- 파라미터 및 FLOPs 계산 ---
    # 샘플 입력을 가져와서 GFLOPs 계산에 사용
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input.to(device)

    # 0인 가중치를 제외하고 FLOPs 계산
    flops = _count_sparse_flops(model, sample_input)
    gflops = flops / 1e9  # GFLOPs 단위로 변환

    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    # 0이 아닌 가중치(pruning된 모델 고려)의 파라미터 수 계산
    nonzero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)
    # --- 계산 종료 ---

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if args.dataset == "imagenet":
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
            else:
                test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Top-1 정확도
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Top-5 정확도
            top5 = output.topk(5, dim=1)[1]
            correct_top5 += top5.eq(target.view(-1,1)).sum().item()

            # Precision/Recall/F1을 위한 예측값 저장
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 평균 손실
    test_loss /= len(test_loader.dataset)

    # 정확도
    acc1 = 100. * correct / len(test_loader.dataset)
    acc5 = 100. * correct_top5 / len(test_loader.dataset)

    # Precision / Recall / F1
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    current_lr = None
    if optimizer is not None:
        current_lr = optimizer.param_groups[0]['lr']

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
        "total_params_M": total_params / 1e6,
        "nonzero_params_M": nonzero_params.item() / 1e6,
        "gflops": gflops,
        "lr": current_lr,
        "args": vars(args) if args is not None else {}
    }

    json_file = Path(args.output_dir)
    with open(json_file, mode="a") as f:
        f.write(json.dumps(metrics) + "\n")

    # --- 출력 포맷 수정 ---
    print(f"[{prefix}] Epoch {metrics['epoch']} | "
          f"Params(M): {metrics['nonzero_params_M']:.2f}/{metrics['total_params_M']:.2f} | "
          f"GFLOPs: {metrics['gflops']:.2f} | "
          f"Loss {test_loss:.4f}, Top1 {acc1:.2f}%, Top5 {acc5:.2f}%, "
          f"P {precision:.3f}, R {recall:.3f}, F1 {f1:.3f} | "
          f"LR {current_lr}"
    )

    return metrics

def equation_to_solve(alpha, target_t, n, c):
    calculated_t = ((1 - alpha)**n) * ((1 + c * alpha)**(n - 1))
    return target_t - calculated_t

def find_alpha_for_t(target_t, n, c):
    """
    주어진 T, N, c에 대해 α를 계산하는 함수
    """
    try:
        # brentq를 사용하여 [0, 1] 범위 내에서 해를 찾음
        alpha_solution = opt.brentq(
            f=equation_to_solve,
            a=0,
            b=1,
            args=(target_t, n, c)
        )
        return alpha_solution
    except ValueError:
        # 해를 찾지 못한 경우 (거의 발생하지 않음)
        return None