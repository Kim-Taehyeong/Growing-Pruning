from utils.sparsity import get_pruning_sparsities_erk, get_pruning_sparsities_uniform
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import print_prune, apply_prune, testAndSave, \
find_alpha_for_t, apply_l1_prune, initialize_Z_and_U, admm_penalty_loss, \
online_accumulate_grad_ema, mask_grads, enforce_weight_mask, update_X, \
update_Z_l1, update_Z, update_U, print_convergence, rigl_grow_once_global, \
rebuild_masks_from_weights, apply_global_prune, update_Z_global
from utils.experiment import log_epoch_eta, save_checkpoint, save_final_model

def gpadmm(args, model, device, train_loader, test_loader, optimizer):
    # 사전 Sparsity 계산 방법 처리
    if args.sparsity_method == 'erk':
        args.percent = get_pruning_sparsities_erk(model, args)
    elif args.sparsity_method == 'er':
        args.percent = get_pruning_sparsities_erk(model, args, include_kernel=False)
    elif args.sparsity_method == 'uniform':
        args.percent = get_pruning_sparsities_uniform(model, args)
    else:
        # Sparsity 방법이 지정되지 않은 경우, Automation 모드로 설정
        args.percent = None
    _rigl_admm_cycle_train_global(args, model, device, train_loader, test_loader, optimizer)


def _rigl_admm_cycle_train_global(args, model, device, train_loader, test_loader, base_optimizer_cls):
    # LR 스케줄러 설정
    optimizer = base_optimizer_cls(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)

    # 코사인 감쇠로 1차(Main-Pruning), 2차(Retraining) 스케줄러 설정
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_cycles)
    retrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_re_epochs)

    # 두 스케줄러를 사이클 단위로 순차적으로 적용하는 SequentialLR 설정
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[main_scheduler, retrain_scheduler], milestones=[args.num_cycles])

    # 사이클 단위로 저장할 Gradient Buffer 초기화
    Z, _ = initialize_Z_and_U(model)
    grad_buf = {}
    total_epochs = args.num_cycles * args.grow_interval + args.num_re_epochs
    global_epoch = 0
    args._extra_metrics = {"global_epoch": global_epoch, "total_epochs": total_epochs, "stage": "dense-eval"}
    dense_metrics = testAndSave(args, model, device, test_loader, "Dense Model Test", optimizer=optimizer)
    save_checkpoint(args, model, optimizer, stage="gpadmm_dense", stage_epoch=0,
                    global_epoch=1, metrics=dense_metrics)

    # 초기 타겟 희소성과 남길 가중치 비율 계산
    T = 1.0 - args.sparsity
    N = args.num_cycles
    C = args.c
    
    current_retained_ratio = 1.0

    # 각 사이클마다 적용할 알파와 베타 계산
    alpha = find_alpha_for_t(T, N, C)
    beta = alpha * C

    # Pruning 시작
    masks = apply_global_prune(model, device, args)
 

    for c in range(args.num_cycles):
        print(f'[RigL+ADMM] Cycle {c+1}/{args.num_cycles}')

        current_retained_ratio *= (1.0 - alpha)
        args.global_target_sparsity = 1.0 - current_retained_ratio   

        masks = apply_global_prune(model, device, args)
        masks, Z = _admm_prune_stage(args, model, device, train_loader, test_loader, masks,
                                 optimizer, epochs=args.grow_interval, cycle=c + 1,
                                 grad_buffers=grad_buf, Z_input=Z,
                                 total_epochs=total_epochs, global_epoch=global_epoch)
        global_epoch += args.grow_interval
        if c < args.num_cycles - 1:
            current_retained_ratio *= (1.0 + beta)
            masks = rigl_grow_once_global(model, masks, global_grow_frac=beta, grad_buffers=grad_buf)
        grad_buf = {}
        combined_scheduler.step()
    
    args.global_target_sparsity = T
    print('[RigL+ADMM] Final fixed-mask retraining...')

    # (4) 최종 재학습(마스크 고정, 성장/프루닝 없음)
    for epoch in range(args.num_re_epochs):
        print(f'[RigL+ADMM] Re epoch: {epoch+1}/{args.num_re_epochs}')
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if args.dataset == "imagenet":
                loss = F.cross_entropy(output, target)
            else:
                loss = F.nll_loss(output, target)
            running_loss += loss.item()
            loss.backward()
            mask_grads(model, masks)     # pruned 고정
            optimizer.step()
            enforce_weight_mask(model, masks)
        global_epoch += 1
        elapsed, eta = log_epoch_eta(args.experiment["started_at"], global_epoch, total_epochs, "GPADMM-Re")
        args._extra_metrics = {
            "train_loss": running_loss / max(len(train_loader), 1),
            "global_epoch": global_epoch,
            "total_epochs": total_epochs,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "stage": "retraining",
        }
        combined_scheduler.step()
        metrics = testAndSave(args, model, device, test_loader, "ADMM-Re-Training", epoch, optimizer=optimizer)
        save_checkpoint(args, model, optimizer, stage="gpadmm_retrain", stage_epoch=epoch + 1,
                        global_epoch=global_epoch, metrics=metrics)

    print_prune(model, args)
    save_final_model(args, model, optimizer=optimizer, tag="gpadmm_final")
    return masks


def _admm_prune_stage(args, model, device, train_loader, test_loader, masks, optimizer, epochs, cycle,
                      grad_buffers, Z_input, total_epochs, global_epoch):
    Z = Z_input
    _, U = initialize_Z_and_U(model)

    for epoch in range(epochs):
        print(f'[RigL+ADMM] ADMM epoch {epoch+1}/{epochs}')
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if args.dataset == "imagenet":
                task_loss = F.cross_entropy(output, target)
            else:
                task_loss = F.nll_loss(output, target)
            running_loss += task_loss.item()
            task_loss.backward(retain_graph=True)
            online_accumulate_grad_ema(model, grad_buffers, beta=0.95)

            penalty_loss = admm_penalty_loss(args, device, model, Z, U)
            penalty_loss.backward()

            mask_grads(model, masks)
            optimizer.step()
            enforce_weight_mask(model, masks)

        # ADMM 업데이트(에폭 말)
        X = update_X(model)
        Z = update_Z_global(X, U, args)
        U = update_U(U, X, Z)
        print_convergence(model, X, Z)
        current_global_epoch = global_epoch + epoch + 1
        elapsed, eta = log_epoch_eta(args.experiment["started_at"], current_global_epoch, total_epochs,
                                     f"GPADMM-C{cycle}")
        args._extra_metrics = {
            "train_loss": running_loss / max(len(train_loader), 1),
            "global_epoch": current_global_epoch,
            "total_epochs": total_epochs,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "stage": f"cycle-{cycle}-admm",
        }
        metrics = testAndSave(args, model, device, test_loader, f"ADMM-Cycle {cycle}", epoch, optimizer=optimizer)
        save_checkpoint(args, model, optimizer, stage=f"gpadmm_cycle_{cycle}", stage_epoch=epoch + 1,
                        global_epoch=current_global_epoch, metrics=metrics)

    # 목표 희소도까지 프루닝
    masks = apply_global_prune(model, device, args)

    # 프루닝 후 모델에서 새 마스크 복원 (다음 사이클에 사용)
    new_masks = rebuild_masks_from_weights(model, device=device)
    print_prune(model, args)
    args._extra_metrics = {
        "global_epoch": global_epoch + epochs,
        "total_epochs": total_epochs,
        "stage": f"cycle-{cycle}-post-pruning",
    }
    post_metrics = testAndSave(args, model, device, test_loader, f"ADMM-Cycle {cycle} Post-Pruning", optimizer=optimizer)
    save_checkpoint(args, model, optimizer, stage=f"gpadmm_cycle_{cycle}_post", stage_epoch=epochs,
                    global_epoch=max(global_epoch + epochs, 1), metrics=post_metrics)
    return new_masks, Z