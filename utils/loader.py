from model.model import LeNet, AlexNet
import torchvision.models as models
from model.optimizer import PruneAdam

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
    if args.model == "lenet":
        return LeNet()
    elif args.model == "alexnet":
        return AlexNet()
    elif args.model == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
def load_optimizer(args, kwargs):
    # 현재는 PruneAdam만 지원하지만, 향후 다른 옵티마이저도 추가할 수 있도록 함수로 분리
    return PruneAdam