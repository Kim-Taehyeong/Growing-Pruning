import argparse
import torch

from utils.loader import load_dataset, load_model, load_optimizer
from methology.admm import admm
from methology.GPadmm import gpadmm
from utils.experiment import setup_experiment, redirect_output_to_log, finish_live_eta
from model.pretrain import pretrain


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR with ADMM or RigL+ADMM')

    # Dataset
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "cifar10", "imagenet"], metavar='D')
    parser.add_argument('--output-dir', type=str, default="", metavar='O', help='Directory to save metrics JSON files')
    parser.add_argument('--save-dir', type=str, default="./runs", help='Directory to save checkpoints and final models')
    parser.add_argument('--run-name', type=str, default="", help='Optional run name. If empty, generated automatically')
    parser.add_argument('--save-checkpoint-every', type=int, default=1,
                        help='Save checkpoint every N epochs (global epoch). Use 0 to disable')
    parser.add_argument('--eta-update-interval', type=float, default=1.0,
                        help='Terminal ETA refresh interval in seconds')

    # Model
    parser.add_argument('--model', type=str, default="lenet", choices=["lenet", "alexnet", "resnet50"], metavar='M')

    # Batches
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')

    # 기존 ADMM(풀런) 파라미터
    parser.add_argument('--percent', type=list, default=[0.8, 0.92, 0.991, 0.93], metavar='P',
                        help='(ADMM) per-layer pruning percentage list')
    parser.add_argument('--alpha', type=float, default=5e-4, metavar='L')
    parser.add_argument('--rho', type=float, default=1e-2, metavar='R')
    parser.add_argument('--l1', action='store_true', default=False,
                        help='use l1 ADMM regularization/pruning (instead of cardinality)')
    
    # ADMM Epoch 관련 파라미터
    parser.add_argument('--l2', action='store_true', default=False)
    parser.add_argument('--num_pre_epochs', type=int, default=3, metavar='P')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N')
    parser.add_argument('--num_re_epochs', type=int, default=3, metavar='R')

    # 공통 옵티마이저
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--save-model', action='store_true', default=False)

    # RigL + ADMM 파라미터
    parser.add_argument('--use-rigl-admm', action='store_true', default=False,
                        help='enable Grow (RigL) -> ADMM pruning cycles')
    parser.add_argument('--sparsity', type=float, default=0.99, help='global target sparsity (0~1)')
    parser.add_argument('--sparsity-method', type=str, default='uniform', choices=['erk', 'er', 'uniform'])
    parser.add_argument('--min-layer-keep-ratio', type=float, default=0.01,
                        help='minimum kept ratio per prunable layer during global pruning')

    # RigL + ADMM Epoch 설정 파라미터
    parser.add_argument('--num-cycles', type=int, default=3)
    parser.add_argument('--grow-interval', type=int, default=5)

    # Online 계산 사용시 무효화
    parser.add_argument('--grow-grad-steps', type=int, default=50)

    # 점진적 Cycle 학슴 (Pruning -> Growing)
    parser.add_argument('--c', type=float, default=0.5) 

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # 랜덤 시드 비활성화
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 데이터셋 로드
    train_loader, test_loader = load_dataset(args, kwargs)


    # 모델 초기화
    model = load_model(args, kwargs).to(device)

    # Optimizer 초기화
    base_optimizer_cls = load_optimizer(args, kwargs)

    # 실험 로그/체크포인트 저장 경로 초기화
    setup_experiment(args)
    redirect_output_to_log(args)

    # Pretrain Phase: 공통 사전학습(ADMM/GPADMM 모두 적용)
    if args.num_pre_epochs > 0:
        model = pretrain(
            args,
            kwargs,
            model,
            device,
            train_loader,
            test_loader,
            base_optimizer_cls,
            total_epochs=args.num_pre_epochs + args.num_epochs + args.num_re_epochs,
        )

    try:
        if args.use_rigl_admm:
            gpadmm(args, model, device, train_loader, test_loader, base_optimizer_cls)
        else: # 기존 ADMM 파이프 라인
            admm(args, model, device, train_loader, test_loader, base_optimizer_cls)
    finally:
        finish_live_eta(args)

if __name__ == "__main__":
    main()