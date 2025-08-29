import torch
import torch.nn.functional as F
import numpy as np
import json


def regularized_nll_loss(args, model, output, target):
    index = 0
    loss = F.nll_loss(output, target)
    if args.l2:
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight":
                loss += args.alpha * param.norm()
                index += 1
    return loss


def admm_loss(args, device, model, Z, U, output, target):
    idx = 0
    loss = F.nll_loss(output, target)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
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
        if name.split('.')[-1] == "weight":
            Z += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
    return Z, U


def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
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
        if name.split('.')[-1] == "weight":
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
        if name.split('.')[-1] == "weight":
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
        if name.split('.')[-1] == "weight":
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def print_convergence(model, X, Z):
    idx = 0
    print("normalized norm of (weight - projection)")
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight":
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
