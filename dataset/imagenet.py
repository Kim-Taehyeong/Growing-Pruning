import torch
from datasets import load_dataset
from torchvision import transforms


def load_imagenet(args, kwargs):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Hugging Face ImageNet 데이터셋
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        cache_dir=args.data_dir,
    )

    def transform_train(batch):
        images = [
            train_transform(image.convert("RGB"))
            for image in batch["image"]
        ]

        return {
            "image": images,
            "label": batch["label"],
        }

    def transform_test(batch):
        images = [
            test_transform(image.convert("RGB"))
            for image in batch["image"]
        ]

        return {
            "image": images,
            "label": batch["label"],
        }

    train_dataset = dataset["train"].with_transform(transform_train)
    test_dataset = dataset["validation"].with_transform(transform_test)

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        )

        return images, labels

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **kwargs,
    )

    return train_loader, test_loader