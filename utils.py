import torch
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path



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
            if args.dataset == "imagenet":
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
            else:
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