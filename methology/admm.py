import torch
import torch.nn.functional as F
import time

from utils.utils import admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune, testAndSave
from utils.experiment import save_checkpoint, save_final_model, update_live_eta


def admm(args, model, device, train_loader, test_loader, base_optimizer_cls):
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