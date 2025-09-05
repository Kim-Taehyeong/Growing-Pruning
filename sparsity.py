import torch.nn as nn
import numpy as np

def get_pruning_sparsities_uniform(model, args):
    sparsities = []
    first = True
    for name, param in model.named_parameters():
        if name.endswith("weight") and param.dim() >= 2:
            if first:
                sparsities.append(0.0)   # 첫 레이어 dense
                first = False
            else:
                sparsities.append(args.sparsity)
    return sparsities

def get_pruning_sparsities_erk(model, args, include_kernel=True, erk_power_scale=1.0):
    all_layers = []
    first_layer = True
    
    for name, param in model.named_parameters():
        if name.endswith("weight") and param.dim() >= 2:
            shape = list(param.shape)
            all_layers.append((name, shape, first_layer))
            first_layer = False

    # 기본 sparsity
    default_sparsity = args.sparsity

    dense_layers = set()
    is_eps_valid = False

    while not is_eps_valid:
        divisor = 0
        rhs = 0
        raw_probabilities = {}

        for idx, (layer, shape, is_first) in enumerate(all_layers):
            n_param = np.prod(shape)

            if idx in dense_layers or is_first:
                # 첫번째 레이어는 무조건 dense
                continue

            # 기본 희소율 적용 시 남기는 수
            n_zeros = int(default_sparsity * n_param)
            n_ones = n_param - n_zeros
            rhs += n_ones

            if include_kernel:
                raw_prob = (np.sum(shape) / np.prod(shape)) ** erk_power_scale
            else:
                n_in, n_out = shape[-2:]
                raw_prob = (n_in + n_out) / (n_in * n_out)
            raw_probabilities[idx] = raw_prob
            divisor += raw_prob * n_param

        eps = rhs / divisor if divisor > 0 else 0.0

        max_prob_one = max([raw_prob * eps for raw_prob in raw_probabilities.values()], default=0)
        if max_prob_one > 1:
            is_eps_valid = False
            # 가장 큰 layer를 dense로 전환
            max_idx = max(raw_probabilities, key=lambda k: raw_probabilities[k])
            dense_layers.add(max_idx)
        else:
            is_eps_valid = True

    # sparsities 리스트 생성
    sparsities = []
    for idx, (layer, shape, is_first) in enumerate(all_layers):
        n_param = np.prod(shape)
        if is_first or idx in dense_layers:
            sparsities.append(0.0)  # dense
        else:
            prob_one = eps * raw_probabilities[idx]
            sparsities.append(1.0 - prob_one)

    return sparsities
        