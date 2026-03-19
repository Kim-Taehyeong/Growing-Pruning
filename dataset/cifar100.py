
import torch
from torchvision import datasets, transforms


def load_CIFAR100(args, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            'data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),
        ),
        shuffle=True,
        batch_size=args.batch_size,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            'data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),
        ),
        shuffle=False,
        batch_size=args.test_batch_size,
        **kwargs,
    )

    return train_loader, test_loader