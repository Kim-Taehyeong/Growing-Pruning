import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import tqdm

try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except Exception:
    precision_score = None
    recall_score = None
    f1_score = None



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

def admm_penalty_loss(args, device, model, Z, U):
    penalty = 0.0
    idx = 0
    for name, param in model.named_parameters():
        # 프루닝 대상 레이어(보통 weight)만 골라냅니다.
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            u = U[idx].to(device)
            z = Z[idx].to(device)
            
            # ADMM 수식: (rho / 2) * ||W - Z + U||_2^2
            # .pow(2).sum() 혹은 .norm()**2 를 사용하세요.
            diff = param - z + u
            penalty += (args.rho / 2) * torch.sum(diff**2)
            
            # 추가적인 L2 정규화가 필요하다면 여기서 더해줍니다.
            if args.l2:
                penalty += args.alpha * torch.sum(param**2)
                
            idx += 1
    return penalty


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

def update_Z_global(X, U, args):
    new_Z = []
    temp_Z_list = []
    all_abs_values = []

    # 1. 모든 레이어에 대해 (X + U)를 계산하고 절댓값을 한데 모음
    for x, u in zip(X, U):
        z_temp = x + u
        temp_Z_list.append(z_temp)
        # 전역 임계값을 구하기 위해 모든 값을 1차원으로 펴서 수집
        all_abs_values.append(z_temp.abs().view(-1))

    all_abs_tensor = torch.cat(all_abs_values)

    k = int(all_abs_tensor.numel() * args.global_target_sparsity)
    if k > 0:
        threshold, _ = torch.kthvalue(all_abs_tensor, k)
    else:
        threshold = -1.0

    for z_temp in temp_Z_list:
        mask = (z_temp.abs() > threshold).float()
        z_final = z_temp * mask
        new_Z.append(z_final)

    return tuple(new_Z)


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

def apply_global_prune(model, device, args):
    print(f"Apply Global Pruning")
    dict_mask = {}
    all_weights = []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            all_weights.append(param.data.abs().view(-1))
    
    all_weights_tensor = torch.cat(all_weights)

    # 전역 희소도 설정값을 우선순위대로 확인
    global_sparsity = getattr(args, "global_target_sparsity", None)
    if global_sparsity is None:
        global_sparsity = getattr(args, "target_global_percent", None)
    if global_sparsity is None:
        global_sparsity = getattr(args, "sparsity", 0.0)

    # 가중치 개수 계산
    k = int(all_weights_tensor.numel() * float(global_sparsity))
    if k > 0:
        threshold, _ = torch.kthvalue(all_weights_tensor, k)
    else:
        threshold = -1.0

    # 전역 프루닝 후 특정 레이어가 전부 0이 되는 것을 방지
    min_keep_ratio = float(getattr(args, "min_layer_keep_ratio", 0.0))

    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim() >= 2:
            mask = (param.data.abs() > threshold).float().to(device)

            if min_keep_ratio > 0.0:
                keep_count = int(mask.sum().item())
                min_keep = max(1, int(param.numel() * min_keep_ratio))
                min_keep = min(min_keep, param.numel())
                if keep_count < min_keep:
                    flat_abs = param.data.abs().view(-1)
                    topk_idx = torch.topk(flat_abs, k=min_keep, largest=True, sorted=False).indices
                    flat_mask = torch.zeros_like(flat_abs, device=device)
                    flat_mask[topk_idx] = 1.0
                    mask = flat_mask.view_as(param)

            param.data.mul_(mask)
            dict_mask[name] = mask

            actual_sparsity = 1.0 - (mask.sum().item() / mask.numel())
            print(f"{name} sparsity: {actual_sparsity:.4f}")
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
            num = (x - z).norm().item()
            den = x.norm().item()
            if den <= 1e-12:
                ratio = 0.0 if num <= 1e-12 else float("inf")
            else:
                ratio = num / den
            print("({}): {:.4f}".format(name, ratio))
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
    total_flops = 0.0

    def multiply_adds_hook(module, input, output):
        nonlocal total_flops
        # torch.nn.Linear
        if isinstance(module, torch.nn.Linear):
            batch_size = input[0].shape[0] if input[0].dim() > 1 else 1
            nonzero_weights = torch.count_nonzero(module.weight).item()
            # 가중치에 대한 곱셈-누산(Multiply-Accumulate) 연산
            macs = nonzero_weights * batch_size
            total_flops += 2 * macs # 1 MAC = 2 FLOPs (곱셈 1, 덧셈 1)
            # Bias 덧셈 연산
            if module.bias is not None:
                total_flops += module.bias.numel() * batch_size

        # torch.nn.Conv2d
        elif isinstance(module, torch.nn.Conv2d):
            batch_size, _, output_h, output_w = output.shape
            nonzero_weights = torch.count_nonzero(module.weight).item()
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

    return float(total_flops)


def _to_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]
    # Fallback for arbitrary runtime objects (e.g., file handles)
    if hasattr(obj, "__dict__"):
        return str(obj)
    return obj


def _format_hms(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_args_dict(args):
    if args is None:
        return {}

    raw = vars(args)
    safe = {}
    for key, value in raw.items():
        # runtime/private 필드는 메트릭 JSON에서 제외
        if key.startswith("_"):
            continue
        if key == "experiment":
            continue
        safe[key] = _to_json_serializable(value)
    return safe

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
    if precision_score is not None and recall_score is not None and f1_score is not None:
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    else:
        labels = np.unique(np.concatenate((np.array(all_targets), np.array(all_preds))))
        p_list, r_list, f1_list = [], [], []
        for label in labels:
            tp = np.sum((np.array(all_preds) == label) & (np.array(all_targets) == label))
            fp = np.sum((np.array(all_preds) == label) & (np.array(all_targets) != label))
            fn = np.sum((np.array(all_preds) != label) & (np.array(all_targets) == label))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f)
        precision = float(np.mean(p_list)) if p_list else 0.0
        recall = float(np.mean(r_list)) if r_list else 0.0
        f1 = float(np.mean(f1_list)) if f1_list else 0.0

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
        "args": _safe_args_dict(args)
    }

    extra_metrics = getattr(args, "_extra_metrics", None)
    if isinstance(extra_metrics, dict) and extra_metrics:
        metrics.update(extra_metrics)
        args._extra_metrics = {}

    json_file = Path(args.output_dir)
    with open(json_file, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(_to_json_serializable(metrics)) + "\n")

        elapsed_sec = metrics.get("elapsed_sec")
        eta_sec = metrics.get("eta_sec")
        time_part = ""
        if elapsed_sec is not None and eta_sec is not None:
          time_part = f" | Elapsed {_format_hms(elapsed_sec)} ETA {_format_hms(eta_sec)}"

        # --- 출력 포맷 수정 ---
        print(f"[{prefix}] Epoch {metrics['epoch']} | "
            f"Params(M): {metrics['nonzero_params_M']:.2f}/{metrics['total_params_M']:.2f} | "
            f"GFLOPs: {metrics['gflops']:.2f} | "
            f"Loss {test_loss:.4f}, Top1 {acc1:.2f}%, Top5 {acc5:.2f}%, "
            f"P {precision:.3f}, R {recall:.3f}, F1 {f1:.3f} | "
            f"LR {current_lr}{time_part}"
        )

    return metrics

def equation_to_solve(alpha, target_t, n, c):
    calculated_t = ((1 - alpha)**n) * ((1 + c * alpha)**(n - 1))
    return target_t - calculated_t

def find_alpha_for_t(target_t, n, c):
    """
    주어진 T, N, c에 대해 α를 계산하는 함수
    """
    left, right = 0.0, 1.0
    f_left = equation_to_solve(left, target_t, n, c)
    f_right = equation_to_solve(right, target_t, n, c)

    if f_left == 0:
        return left
    if f_right == 0:
        return right
    if f_left * f_right > 0:
        return None

    for _ in range(120):
        mid = (left + right) / 2.0
        f_mid = equation_to_solve(mid, target_t, n, c)
        if abs(f_mid) < 1e-10:
            return mid
        if f_left * f_mid <= 0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid

    return (left + right) / 2.0

def rigl_grow_once_global(model, masks, global_grow_frac, grad_buffers):
    all_pruned_grads = []
    location_map = []

    total_active_params = 0
    
    for name, p in model.named_parameters():
        if name not in masks:
            continue
        mask = masks[name]
        total_active_params += mask.sum().item()

        mask_flat = mask.view(-1)
        pruned_idx = (mask_flat == 0).nonzero(as_tuple=False).view(-1)

        if pruned_idx.numel() == 0:
            continue
            
        gabs = grad_buffers.get(name, None)
        if gabs is not None:
            pruned_grads = gabs.view(-1)[pruned_idx]
            all_pruned_grads.append(pruned_grads)
            location_map.append((name, pruned_idx))

    if not all_pruned_grads:
        return masks
    
    g_global = int(round(total_active_params * global_grow_frac))

    all_grads_tensor = torch.cat(all_pruned_grads)

    k = min(g_global, all_grads_tensor.numel())
    if k <= 0:
        return masks
    
    _, topk_global_indices = torch.topk(all_grads_tensor, k=k, largest=True, sorted=False)

    lengths = [len(g) for g in all_pruned_grads]
    cum_lengths = torch.cumsum(torch.tensor([0] + lengths), dim=0)

    params_dict = dict(model.named_parameters())

    with torch.no_grad():
        for i in range(len(location_map)):
            name, pruned_indices = location_map[i]
            start_idx = cum_lengths[i]
            end_idx = cum_lengths[i + 1]

            mask_in_this_layer = (topk_global_indices >= start_idx) & (topk_global_indices < end_idx)
            local_topk_indices = topk_global_indices[mask_in_this_layer] - start_idx

            if local_topk_indices.numel() > 0:
                actual_grow_indices = pruned_indices[local_topk_indices]
                masks[name].view(-1).index_fill_(0, actual_grow_indices, 1.0)

                params_dict[name].view(-1).index_fill_(0, actual_grow_indices, 0.0)

    enforce_weight_mask(model, masks)
    return masks


def rigl_grow_once_chore_layer_wise(model, masks, grow_fracs_list, grad_buffers, show_tqdm: bool = True):
    # --- 사전 계산: 마스크가 적용된 파라미터 목록 ---
    param_list = [(n, p) for n, p in model.named_parameters() if n in masks]

    # --- 입력 검증 및 grow_frac 매핑 ---
    # grow_fracs_list의 길이가 레이어 수와 맞는지 확인
    if len(grow_fracs_list) != len(param_list):
        raise ValueError(
            f"Length of grow_fracs_list ({len(grow_fracs_list)}) must match "
            f"the number of prunable layers ({len(param_list)})."
        )
    # 사용하기 쉽도록 리스트를 레이어 이름 기반의 딕셔너리로 변환
    grow_fracs = {name: frac for (name, _), frac in zip(param_list, grow_fracs_list)}

    # --- 각 레이어별로 순회하며 성장 진행 ---
    pbar = tqdm(param_list, total=len(param_list), desc="Layer-wise Growing",
                disable=not show_tqdm, dynamic_ncols=True, leave=False)

    with torch.no_grad():
        for name, p in pbar:
            # 이 레이어에 할당된 성장률(grow_frac) 가져오기
            layer_grow_frac = grow_fracs.get(name, 0.0)
            if layer_grow_frac <= 0:
                continue

            # 1. 레이어별 성장 개수(g_layer) 계산
            #    (기존) 전역 g를 분배 -> (변경) 레이어의 활성 파라미터 수에 직접 성장률 곱하기
            active_params_layer = masks[name].sum().item()
            g_layer = int(round(active_params_layer * layer_grow_frac))
            if g_layer <= 0:
                continue

            # 2. 이 레이어의 Pruned 위치와 그래디언트 절대값 수집
            mask_flat = masks[name].view(-1)
            pruned_idx = (~mask_flat.bool()).nonzero(as_tuple=False).view(-1)

            # 성장시킬 파라미터가 없으면(전부 활성화 상태이면) 건너뜀
            if pruned_idx.numel() == 0:
                continue

            gabs = grad_buffers.get(name, None)
            if gabs is None:
                continue

            pruned_grads = gabs.view(-1)[pruned_idx]

            # 3. 이 레이어 내에서만 top-k 선택
            # 성장시킬 수 있는 최대 개수는 현재 pruned된 파라미터 개수를 넘을 수 없음
            k = min(g_layer, pruned_grads.numel())
            if k <= 0:
                continue

            _, topk_local_idx = torch.topk(pruned_grads, k=k, largest=True, sorted=False)

            # 4. 실제 마스크 상의 인덱스 가져오기 및 갱신
            grow_indices = pruned_idx[topk_local_idx]
            mask_flat.index_fill_(0, grow_indices, True)
            
            # 새로 성장한 가중치는 0으로 초기화
            p_flat = p.view(-1)
            p_flat.index_fill_(0, grow_indices, 0.0)

    # step 후 마스크를 가중치에 적용하여 0으로 유지
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