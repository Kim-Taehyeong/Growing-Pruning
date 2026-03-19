import time
import torch.nn.functional as F
from utils.utils import testAndSave
from utils.experiment import save_checkpoint, update_live_eta

def pretrain(args, kwargs, model, device, train_loader, test_loader, optimizer_cls, total_epochs):
    if args.dataset == "imagenet":
        print("Use Pretraining for ImageNet")
        return model

    optimizer = optimizer_cls(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
    global_epoch = 0

    for epoch in range(args.num_pre_epochs):
        print(f'[Pretrain] Epoch: {epoch + 1}/{args.num_pre_epochs}')
        model.train()
        running_loss = 0.0
        num_batches = max(len(train_loader), 1)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            completed = global_epoch + float(batch_idx + 1) / float(num_batches)
            update_live_eta(args, completed, total_epochs, stage="pretrain")
        global_epoch += 1
        elapsed = time.time() - args.experiment["started_at"]
        eta = 0.0
        if global_epoch > 0:
            eta = (elapsed / global_epoch) * max(total_epochs - global_epoch, 0)
        update_live_eta(args, global_epoch, total_epochs, stage="pretrain")
        args._extra_metrics = {
            "train_loss": running_loss / max(len(train_loader), 1),
            "global_epoch": global_epoch,
            "total_epochs": total_epochs,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "stage": "pretraining",
        }
        metrics = testAndSave(args, model, device, test_loader, "Pretraining", epoch, optimizer=optimizer)
        save_checkpoint(args, model, optimizer, stage="pretrain", stage_epoch=epoch + 1,
                        global_epoch=global_epoch, metrics=metrics)

    return model
