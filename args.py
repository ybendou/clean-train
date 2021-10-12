import argparse
import os
import random

parser = argparse.ArgumentParser(description="""Optimized code for training usual datasets/model

Examples of use (to reach peak accuracy, not for fastest prototyping):
To train MNIST with 99.64% accuracy (5 minutes):
python main.py --epochs 30 --milestones "[10, 20]" --dataset MNIST --feature-maps 8
To train MNIST with 10% database and 99.31% accuracy (10 minutes):
python main.py --epochs 300 --milestones "[100, 200]" --dataset MNIST --dataset-size 6000 --model wideresnet --feature-maps 4 --skip-epochs 300
To train Fashion-MNIST with 96% accuracy (2 hours):
python main.py --dataset fashion --mixup
To train CIFAR10 with 95.90% accuracy (1 hour):
python main.py --mixup
To train CIFAR100 with 78.55% accuracy (93.54% top-5) (1hour):
python main.py --mixup --dataset cifar100
To train CIFAR100 with 80.12% accuracy (94.70% top-5) (4h):
python main.py --mixup --model wideresnet --feature-maps 16 --dataset CIFAR100
To train Omniglot (few-shot) with 99.85% accuracy (99.39% in 1-shot) (10minutes):
python main.py --dataset omniglotfs --dataset-device cpu --feature-maps 16 --milestones '[10,20]' --epochs 30
To train CUBFS (few-shot) with 85.24% accuracy (68.14% in 1-shot) (2h):
python main.py --dataset cubfs --mixup --rotations
To train CIFARFS (few-shot) with 84.87% accuracy (70.43% in 1-shot) (1h):
python main.py --dataset cifarfs --mixup --rotations --skip-epochs 300
To train CIFARFS (few-shot) with 86.83% accuracy (70.27% in 1-shot) (3h):
python main.py --dataset cifarfs --mixup --model wideresnet --feature-maps 16 --skip-epochs 300 --rotations
To train MiniImageNet (few-shot) with 80.43% accuracy (64.11% in 1-shot) (2h):
python main.py --dataset miniimagenet --model resnet12 --gamma 0.2 --milestones '[30,60,90]' --epochs 120 --batch-size 128 --preprocessing 'EME'
To train MiniImageNet (few-shot) with 83.18% accuracy (66.78% in 1-shot) (40h):
python main.py --device cuda:012 --dataset miniimagenet --model S2M2R --lr -0.001 --milestones '[]' --epochs 600 --feature-maps 16 --rotations --manifold-mixup 400 --skip-epochs 600
""", formatter_class=argparse.RawTextHelpFormatter)

### hyperparameters
parser.add_argument("--batch-size", type=int, default=64, help="batch size")
parser.add_argument("--feature-maps", type=int, default=64, help="number of feature maps")
parser.add_argument("--lr", type=float, default="0.1", help="initial learning rate (negative is for Adam, e.g. -0.001)")
parser.add_argument("--epochs", type=int, default=350, help="total number of epochs")
parser.add_argument("--milestones", type=str, default="[100,200,300]", help="milestones for lr scheduler")
parser.add_argument("--gamma", type=float, default=0.1, help="multiplier for lr at milestones")
parser.add_argument("--mixup", action="store_true", help="use of mixup since beginning")
parser.add_argument("--rotations", action="store_true", help="use of rotations self-supervision during training")
parser.add_argument("--model", type=str, default="ResNet18", help="model to train")
parser.add_argument("--preprocessing", type=str, default="RPEME", help="preprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")
parser.add_argument("--manifold-mixup", type=int, default="0", help="deploy manifold mixup as fine-tuning as in S2M2R for the given number of epochs, to be used only with S2M2R model")

### pytorch options
parser.add_argument("--device", type=str, default="cuda:0", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")
parser.add_argument("--dataset-path", type=str, default=os.environ.get("DATASETS"), help="dataset path")
parser.add_argument("--dataset-device", type=str, default="", help="use a different device for storing the datasets (use 'cpu' if you are lacking VRAM)")
parser.add_argument("--deterministic", action="store_true", help="use desterministic randomness for reproducibility")

### run options
parser.add_argument("--skip-epochs", type=int, default="0", help="number of epochs to skip before evaluating few-shot performance")
parser.add_argument("--runs", type=int, default=1, help="number of runs")
parser.add_argument("--quiet", action="store_true", help="prevent too much display of info")
parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset to use")
parser.add_argument("--dataset-size", type=int, default=-1, help="number of training samples (using a subset in that case) (only for classical classification)")
parser.add_argument("--output", type=str, default="", help="output file to write")
parser.add_argument("--save-features", type=str, default="", help="save features to file")
parser.add_argument("--save-model", type=str, default="", help="save model to file")
parser.add_argument("--test-features", type=str, default="", help="test features and exit")
parser.add_argument("--load-model", type=str, default="", help="load model from file")
parser.add_argument("--seed", type=int, default=-1, help="set random seed manually, and also use deterministic approach")

### few-shot parameters
parser.add_argument("--n-runs", type=int, default=10000, help="Number of few-shot runs")
parser.add_argument("--n-ways", type=int, default=5, help="Number of few-shot ways")
parser.add_argument("--n-queries", type=int, default=15, help="Number of few-shot queries")
parser.add_argument("--ncm-loss", action="store_true", help="Use ncm output instead of linear")

args = parser.parse_args()

if args.dataset_device == "":
    args.dataset_device = args.device
if args.dataset_path[-1] != '/':
    args.dataset_path += "/"
if args.device[:5] == "cuda:" and len(args.device) > 5:
    args.devices = []
    for i in range(len(args.device) - 5):
        args.devices.append(int(args.device[i+5]))
    args.device = args.device[:6]
else:
    args.devices = [args.device]

if args.seed == -1:
    args.seed = random.randint(0, 1000000000)
