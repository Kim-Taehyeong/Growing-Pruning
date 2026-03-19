
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
    for name, p in tqdm(model.named_parameters()):
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
        for (name, flat_i) in tqdm(chosen):
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


def rigl_grow_once_chore(model, masks, grow_frac, grad_buffers, show_tqdm=True):
    # --- 사전 계산: 파라미터/레이어 목록 & 맵 ---
    param_list = [(n, p) for (n, p) in model.named_parameters() if n in masks]
    name2idx   = {n: i for i, (n, _) in enumerate(param_list)}
    idx2name   = {i: n for n, i in name2idx.items()}

    # 성장 개수 g
    active_total = sum(m.sum().item() for m in masks.values())
    g = int(max(1, round(active_total * float(grow_frac))))
    if g <= 0:
        return masks

    # --- 1) pruned 위치와 그 |grad| 수집 (tensor로만, Python 튜플 금지) ---
    chunks = []
    owners = []     # 각 엘리먼트가 어느 레이어(정수 id)에 속하는지
    flats  = []     # 레이어 안에서의 flat index
    pbar = tqdm(param_list, total=len(param_list), desc="Collect pruned grads",
                disable=not show_tqdm, dynamic_ncols=True, leave=False)
    for name, p in pbar:
        m = masks[name].view(-1).bool()
        pruned_idx = (~m).nonzero(as_tuple=False).view(-1)
        if pruned_idx.numel() == 0:
            continue
        gabs = grad_buffers.get(name, None)
        if gabs is None:
            continue
        gabs = gabs.view(-1)[pruned_idx]  # pruned 위치의 |grad|
        if gabs.numel() == 0:
            continue

        chunks.append(gabs)                                   # scores
        owners.append(torch.full((gabs.numel(),), name2idx[name],
                                 dtype=torch.int32, device=gabs.device))
        flats.append(pruned_idx.to(torch.int64))              # local flat indices

    if not chunks:
        return masks

    scores_all = torch.cat(chunks, dim=0)
    owners_all = torch.cat(owners, dim=0)
    flats_all  = torch.cat(flats,  dim=0)

    # --- 2) 전역 top-k 선택 ---
    k = min(g, scores_all.numel())
    _, topk_idx = torch.topk(scores_all, k=k, largest=True, sorted=False)
    top_owners  = owners_all.index_select(0, topk_idx)
    top_flats   = flats_all.index_select(0, topk_idx)

    # --- 3) 선택된 인덱스를 레이어별로 묶어서 한 번에 적용 ---
    unique_layers = torch.unique(top_owners)
    apply_pbar = tqdm(unique_layers.tolist(), desc="Activate connections",
                      disable=not show_tqdm, dynamic_ncols=True, leave=False)
    with torch.no_grad():
        # param 딕셔너리는 한 번만 생성
        name2param = {n: p for (n, p) in param_list}
        for lid in apply_pbar:
            lid = int(lid)
            name = idx2name[lid]
            sel_mask = (top_owners == lid)
            sel_idx  = top_flats[sel_mask]             # 이 레이어 내 flat index들

            # 마스크/파라미터 갱신 (벡터화)
            mask_flat = masks[name].view(-1)
            mask_flat.index_fill_(0, sel_idx, True)
            masks[name] = mask_flat.view_as(masks[name])

            p = name2param[name].view(-1)
            p.index_fill_(0, sel_idx, 0.0)             # 새로 켠 가중치 0으로 초기화

    # step 후 값만 0 유지 (forward에서는 mask 곱 X 권장)
    enforce_weight_mask(model, masks)
    return masks








# ==============================================================
# ADMM + RigL 학습 파이프라인
# ==============================================================
def rigl_admm_cycle_train(args, model, device, train_loader, test_loader, base_optimizer_cls):
    optimizer = base_optimizer_cls(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_cycles)
    retrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_re_epochs)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[main_scheduler, retrain_scheduler], milestones=[args.num_cycles])
    grad_buf = {}

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
            online_accumulate_grad_ema(model, grad_buf, beta=0.95)
            optimizer.step()
        testAndSave(args, model, device, test_loader, "Pretraining", epoch)
    testAndSave(args, model, device, test_loader, "Dense Model Test", optimizer=optimizer)
    # (0) 초기 희소 모델 생성(ER/ERK)
    masks = apply_prune(model, device, args)
    enforce_weight_mask(model, masks)
    print_prune(model, args)


    # 사이클 반복
    for c in range(args.num_cycles):
        print(f'[RigL+ADMM] Cycle {c+1}/{args.num_cycles}')

        # (1) 성장용 |grad| 수집 (업데이트 없이 평균)
        # grad_buf = collect_grad_for_growth(model, device, train_loader, steps=args.grow_grad_steps)

        # (2) RigL 성장: pruned 중 |grad| 상위 → 활성화(가중치 0으로 초기화)
        masks = rigl_grow_once_chore(model, masks, grow_frac=args.grow_frac, grad_buffers=grad_buf)
        grad_buf = {}
        # (3) ADMM 프루닝 스테이지: 짧게 학습 후 apply_prune로 목표 희소율 달성
        masks = admm_prune_stage(args, model, device, train_loader, test_loader, masks,
                                 optimizer, epochs=args.grow_interval, cycle=c + 1,
                                 grad_buffers=grad_buf)
        combined_scheduler.step()

    print('[RigL+ADMM] Final fixed-mask retraining...')

    # (4) 최종 재학습(마스크 고정, 성장/프루닝 없음)
    for epoch in range(args.num_re_epochs):
        print(f'[RigL+ADMM] Re epoch: {epoch+1}/{args.num_re_epochs}')
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if args.dataset == "imagenet":
                loss = F.cross_entropy(output, target)
            else:
                loss = F.nll_loss(output, target)
            loss.backward()
            mask_grads(model, masks)     # pruned 고정
            optimizer.step()
            enforce_weight_mask(model, masks)
        combined_scheduler.step()
        testAndSave(args, model, device, test_loader, "ADMM-Re-Training", epoch, optimizer=optimizer)

    print_prune(model, args)
    return masks



def _rigl_admm_cycle_train(args, model, device, train_loader, test_loader, base_optimizer_cls):
    # LR 스케줄러 설정
    optimizer = base_optimizer_cls(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)

    # 코사인 감쇠로 1차(Main-Pruning), 2차(Retraining) 스케줄러 설정
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_cycles)
    retrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_re_epochs)

    # 두 스케줄러를 사이클 단위로 순차적으로 적용하는 SequentialLR 설정
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[main_scheduler, retrain_scheduler], milestones=[args.num_cycles])

    # 사이클 단위로 저장할 Gradient Buffer 초기화
    grad_buf = {}
    testAndSave(args, model, device, test_loader, "Dense Model Test", optimizer=optimizer)

    # 초기 타겟 희소성과 남길 가중치 비율 계산
    initial_target_sparsities = args.percent.copy()
    target_retained_ratios = [1.0 - T for T in initial_target_sparsities] 
    
    # 각 사이클마다 적용할 알파와 베타 계산
    alphas = []
    betas = []
    for T_retained in target_retained_ratios:
        # T_retained: 최종 남길 가중치 비율
        alpha = find_alpha_for_t(T_retained, args.num_cycles, args.c)
        alphas.append(alpha)
        betas.append(alpha * args.c)

    # 현재 남길 가중치 비율 초기화
    current_retained_ratios = [1.0] * len(target_retained_ratios) 
    args.percent = [0.0] * len(initial_target_sparsities) 

    # Pruning 시작
    masks = apply_prune(model, device, args)

    for c in range(args.num_cycles):
        print(f'[RigL+ADMM] Cycle {c+1}/{args.num_cycles}')
        for i in range(len(current_retained_ratios)):
            current_retained_ratios[i] *= (1.0 - alphas[i])

        current_target_sparsities = [1.0 - ratio for ratio in current_retained_ratios]
        args.percent = current_target_sparsities 

        masks = _admm_prune_stage(args, model, device, train_loader, test_loader, masks,
                                 optimizer, epochs=args.grow_interval, cycle=c + 1,
                                 grad_buffers=grad_buf)
        if c < args.num_cycles - 1:
            for i in range(len(current_retained_ratios)):
                current_retained_ratios[i] *= (1.0 + betas[i])
            grow_frac_list = [beta for beta in betas]
            masks = rigl_grow_once_chore_layer_wise(model, masks, grow_fracs_list=grow_frac_list, grad_buffers=grad_buf)
        grad_buf = {}
        combined_scheduler.step()
    args.percent = initial_target_sparsities 
    print('[RigL+ADMM] Final fixed-mask retraining...')

    # (4) 최종 재학습(마스크 고정, 성장/프루닝 없음)
    for epoch in range(args.num_re_epochs):
        print(f'[RigL+ADMM] Re epoch: {epoch+1}/{args.num_re_epochs}')
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if args.dataset == "imagenet":
                loss = F.cross_entropy(output, target)
            else:
                loss = F.nll_loss(output, target)
            loss.backward()
            mask_grads(model, masks)     # pruned 고정
            optimizer.step()
            enforce_weight_mask(model, masks)
        combined_scheduler.step()
        testAndSave(args, model, device, test_loader, "ADMM-Re-Training", epoch, optimizer=optimizer)

    print_prune(model, args)
    return masks