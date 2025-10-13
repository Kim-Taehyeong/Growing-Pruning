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
    # 1. Prunable Layer 목록 및 총 가중치 수 계산
    all_layers = []
    prunable_params = 0
    first_layer_found = False
    
    for name, param in model.named_parameters():
        if name.endswith("weight") and param.dim() >= 2:
            shape = list(param.shape)
            
            # 첫 번째 'prunable' 레이어는 보통 Input Layer로 간주하여 희소율 0.0 (dense)로 처리
            is_first = not first_layer_found
            all_layers.append((name, shape, is_first))
            
            if is_first:
                first_layer_found = True
            else:
                # 첫 번째 레이어는 희소화하지 않으므로 prunable_params에 포함하지 않음
                prunable_params += np.prod(shape)

    # 목표로 하는 전체 남길 가중치의 수 (1.0 - target_sparsity)
    # *첫 번째 레이어를 제외한* 전체 가중치 수를 기준으로 계산해야 함
    target_ones = prunable_params * (1.0 - args.sparsity)

    # 2. ERK 계수 (raw_probabilities) 계산
    raw_probabilities = {}
    for idx, (layer, shape, is_first) in enumerate(all_layers):
        if is_first:
            continue
            
        if include_kernel:
            # 커널 크기를 포함한 ERK
            # (H + W) / (H * W) 형태를 일반화: sum(shape) / prod(shape)
            raw_prob = (np.sum(shape) / np.prod(shape)) ** erk_power_scale
        else:
            # In/Out 채널만 고려한 ERK
            n_in, n_out = shape[-2:]
            raw_prob = (n_in + n_out) / (n_in * n_out)
            
        raw_probabilities[idx] = raw_prob
    
    # 3. Dense Layer 결정 및 EPS (epsilon) 계산 (While Loop)
    dense_layers = set() # 희소율 0.0으로 고정될 레이어 인덱스 집합
    is_eps_valid = False

    while not is_eps_valid:
        # a. 현재 남길 가중치 수 (target_ones) 계산
        # dense_layers에 포함된 레이어는 희소율 0.0, 즉 100% 남김.
        # 따라서, target_ones에서 이 레이어들이 남기는 가중치 수를 제외해야 함.
        current_target_ones = target_ones
        
        # dense_layers에 포함된 레이어의 가중치 수만큼 target_ones에서 차감
        for idx in dense_layers:
            layer_shape = all_layers[idx][1]
            current_target_ones -= np.prod(layer_shape)

        # b. 현재 Prunable Layers의 총 ERK 계수 * 가중치 수 (divisor) 계산
        divisor = 0
        current_prunable_params = 0
        
        for idx, (layer, shape, is_first) in enumerate(all_layers):
            if is_first or idx in dense_layers:
                continue
            
            n_param = np.prod(shape)
            divisor += raw_probabilities[idx] * n_param
            current_prunable_params += n_param
        
        # c. EPS 계산
        # 남길 비율 = current_target_ones / current_prunable_params (이론적 평균 남길 비율)
        # EPS = (남길 비율) / (평균 raw_prob)
        # 즉, EPS = current_target_ones / divisor
        
        eps = current_target_ones / divisor if divisor > 0 else 0.0

        # d. 유효성 검사 (max(raw_prob * eps) <= 1)
        max_prob_one = 0
        if raw_probabilities:
            # dense_layers에 없는 레이어들만 대상으로 검사
            prunable_raw_probs = [raw_probabilities[idx] for idx in raw_probabilities if idx not in dense_layers]
            max_prob_one = max([raw_prob * eps for raw_prob in prunable_raw_probs], default=0)

        if max_prob_one > 1:
            is_eps_valid = False
            # 가장 높은 raw_prob * eps 값을 가지는 레이어 (즉, 희소율이 음수가 되는 레이어)를 찾아 dense로 전환
            
            max_idx = -1
            max_val = -1
            
            for idx, raw_prob in raw_probabilities.items():
                if idx not in dense_layers and (raw_prob * eps) > max_val:
                    max_val = raw_prob * eps
                    max_idx = idx
            
            if max_idx != -1:
                 # 해당 레이어를 dense_layers에 추가하여 다음 반복에서 희소율 0.0으로 고정
                 dense_layers.add(max_idx)
            else:
                 # 모든 레이어가 이미 dense_layers에 추가된 경우 (희귀하지만, 방어 코드)
                 is_eps_valid = True 
        else:
            is_eps_valid = True

    # 4. 최종 희소율 (Sparsities) 계산
    sparsities = []
    final_ones = 0
    
    for idx, (layer, shape, is_first) in enumerate(all_layers):
        n_param = np.prod(shape)
        
        if is_first or idx in dense_layers:
            # 첫 번째 레이어나 dense로 고정된 레이어는 희소율 0.0
            sparsity = 0.0
        else:
            # 최종 남길 확률 = eps * raw_probability
            prob_one = min(1.0, eps * raw_probabilities[idx]) # 1.0을 넘지 않도록 안전 장치
            sparsity = 1.0 - prob_one # 희소율 = 1 - 남길 확률
        
        sparsities.append(sparsity)
        final_ones += n_param * (1.0 - sparsity)
        
    # 최종적으로 계산된 전체 희소율 확인 (옵션)
    # total_sparsity = 1.0 - (final_ones / prunable_params)
    # print(f"Target Sparsity: {args.sparsity:.4f}, Calculated Sparsity: {total_sparsity:.4f}")
    
    return sparsities
        