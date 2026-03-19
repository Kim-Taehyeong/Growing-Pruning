import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.model import LeNet, AlexNet, ResNet20
from model.optimizer import PruneAdam


class LogSoftmaxWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return F.log_softmax(self.backbone(x), dim=1)


def _dataset_num_classes(dataset_name):
    if dataset_name == "mnist":
        return 10
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "imagenet":
        return 1000
    raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_dataset(args, kwargs):
    if args.dataset == "mnist":
        from dataset.mnist import load_mnist
        return load_mnist(args, kwargs)
    elif args.dataset == "cifar10":
        from dataset.cifar10 import load_CIFAR10
        return load_CIFAR10(args, kwargs)
    elif args.dataset == "cifar100":
        from dataset.cifar100 import load_CIFAR100
        return load_CIFAR100(args, kwargs)
    elif args.dataset == "imagenet":
        from dataset.imagenet import load_imagenet
        return load_imagenet(args, kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    

def load_model(args, kwargs):
    num_classes = _dataset_num_classes(args.dataset)
    args.torch_pretrained_loaded = False

    if args.model == "lenet":
        return LeNet()
    elif args.model == "alexnet":
        return AlexNet()
    elif args.model == "resnet20":
        return ResNet20(num_classes=num_classes)
    elif args.model == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        args.torch_pretrained_loaded = weights is not None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model if args.dataset == "imagenet" else LogSoftmaxWrapper(model)
    elif args.model == "vgg19":
        weights = models.VGG19_Weights.DEFAULT
        model = models.vgg19(weights=weights)
        args.torch_pretrained_loaded = weights is not None
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model if args.dataset == "imagenet" else LogSoftmaxWrapper(model)
    elif args.model == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        args.torch_pretrained_loaded = weights is not None
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model if args.dataset == "imagenet" else LogSoftmaxWrapper(model)
    elif args.model == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        args.torch_pretrained_loaded = weights is not None
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model if args.dataset == "imagenet" else LogSoftmaxWrapper(model)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
def load_optimizer(args, kwargs):
    # 현재는 PruneAdam만 지원하지만, 향후 다른 옵티마이저도 추가할 수 있도록 함수로 분리
    return PruneAdam