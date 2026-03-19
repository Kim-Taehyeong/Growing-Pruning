import torch
import torch.nn.functional as F

from utils.utils import regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune, testAndSave
from utils.experiment import log_epoch_eta, save_checkpoint, save_final_model

from tqdm import tqdm


def admm(args, model, device, train_loader, test_loader, base_optimizer_cls):
    optimizer = base_optimizer_cls(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
    total_epochs = args.num_pre_epochs + args.num_epochs + args.num_re_epochs
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
            optimizer.prune_step(mask)
        global_epoch += 1
        elapsed, eta = log_epoch_eta(args.experiment["started_at"], global_epoch, total_epochs, "ADMM-Re")
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

    for epoch in range(args.num_pre_epochs):
        print('Pre epoch: {}'.format(epoch + 1))
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = regularized_nll_loss(args, model, output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        global_epoch += 1
        elapsed, eta = log_epoch_eta(args.experiment["started_at"], global_epoch, total_epochs, "ADMM-Pre")
        args._extra_metrics = {
            "train_loss": running_loss / max(len(train_loader), 1),
            "global_epoch": global_epoch,
            "total_epochs": total_epochs,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "stage": "pretraining",
        }
        metrics = testAndSave(args, model, device, test_loader, "ADMM-Pretraining", epoch, optimizer=optimizer)
        save_checkpoint(args, model, optimizer, stage="admm_pre", stage_epoch=epoch + 1,
                        global_epoch=global_epoch, metrics=metrics)

    Z, U = initialize_Z_and_U(model)
    for epoch in range(args.num_epochs):
        model.train()
        print('Epoch: {}'.format(epoch + 1))
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = admm_loss(args, device, model, Z, U, output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        X = update_X(model)
        Z = update_Z_l1(X, U, args) if args.l1 else update_Z(X, U, args)
        U = update_U(U, X, Z)
        print_convergence(model, X, Z)
        global_epoch += 1
        elapsed, eta = log_epoch_eta(args.experiment["started_at"], global_epoch, total_epochs, "ADMM-Main")
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