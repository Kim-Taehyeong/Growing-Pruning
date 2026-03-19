import torch
from torchvision import datasets, transforms

def load_imagenet(args, kwargs):
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),    # 랜덤 크롭 후 224x224로 리사이즈
            transforms.RandomHorizontalFlip(),    # 랜덤 좌우 반전
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],       # ImageNet 통계값 (mean, std)
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    test_transform = transforms.Compose([
                transforms.Resize(256),              # 짧은 쪽 256으로 resize
                transforms.CenterCrop(224),          # 가운데 224x224 crop
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],      # ImageNet mean
                    std=[0.229, 0.224, 0.225]        # ImageNet std
                ),
            ])
    
    # train & val dataset
    train_dataset = datasets.ImageFolder(root=f"{args.data_dir}/ILSVRC2012_img_train", transform=train_transform)
    test_dataset  = datasets.ImageFolder(root=f"{args.data_dir}/ILSVRC2012_img_val",   transform=test_transform)

    # DataLoader: main.py의 batch-size/worker/pin_memory 설정을 그대로 사용
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, test_loader