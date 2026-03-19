import torch
from torchvision import datasets, transforms

def load_CIFAR10(args, kwargs):
    args.percent = [0.8, 0.92, 0.93, 0.94, 0.95, 0.99, 0.99, 0.93]
    args.num_pre_epochs = 5
    args.num_epochs = 20
    args.num_re_epochs = 5
    args.num_cycles = 4
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                    (0.24703233, 0.24348505, 0.26158768))
                            ])), shuffle=True, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                    (0.24703233, 0.24348505, 0.26158768))
                            ])), shuffle=True, batch_size=args.test_batch_size, **kwargs)
    
    return train_loader, test_loader