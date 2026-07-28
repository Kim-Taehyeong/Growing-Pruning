import torch
from torchvision import datasets, transforms

def load_CIFAR10(args, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_dir, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                    (0.24703233, 0.24348505, 0.26158768))
                            ])), shuffle=True, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_dir, train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                    (0.24703233, 0.24348505, 0.26158768))
                            ])), shuffle=False, batch_size=args.test_batch_size, **kwargs)
    
    return train_loader, test_loader
