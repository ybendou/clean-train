import torch
import numpy as np
import scipy.stats as st
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import time
import argparse
import random

print("Using pytorch version: " + torch.__version__)

parser = argparse.ArgumentParser(description='Cifar quick code')
parser.add_argument("--batch-size", type=int, default=64, help="batch_size")
parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
parser.add_argument("--feature-maps", type=int, default=64, help="number of feature maps")
parser.add_argument("--training", type=str, default="[(100, 0.1), (100, 0.01), (100, 0.001), (50, 0.0001)]", help="training routine")
parser.add_argument("--runs", type=int, default=100, help="number of runs")
parser.add_argument("--quiet", action="store_true", help="prevent too much display of info")
parser.add_argument("--dataset-path", type=str, default=os.environ.get("DATASETS"), help="dataset path")
parser.add_argument("--dataset-device", type=str, default="", help="Use a different dataset for storing the datasets")
parser.add_argument("--dataset-size", type=int, default=-1, help="Number of training samples")
parser.add_argument("--alpha", type=float, default = 0., help="multiplier inside gaussian")
parser.add_argument("--mixup", type=str, default = "standard", help="type of mixup")
args = parser.parse_args()

if args.dataset_device == "":
    args.dataset_device = args.device

class BatchGenerator():
    def __init__(self, data, targets, transforms = [], batch_size = args.batch_size, device = args.dataset_device):
        if args.dataset_size > 0:
            data = data[:args.dataset_size]
            targets = targets[:args.dataset_size]
        self.data = data.to(device)
        self.targets = targets.to(device)
        assert(data.shape[0] == targets.shape[0])
        self.length = data.shape[0]
        self.batch_size = batch_size
        self.transforms = transforms
    def __getitem__(self, idx):
        idx = torch.randint(self.length, [self.batch_size])
        return self.transforms(self.data[idx]), self.targets[idx]
    def generateBatch(self):
        return self.__getitem__(0)
    def __len__(self):
        return self.length

class Dataset():
    def __init__(self, data, targets, shuffle = False, transforms = [], batch_size = args.batch_size, device = args.dataset_device):
        self.data = data.to(device)
        self.targets = targets.to(device)
        assert(data.shape[0] == targets.shape[0])
        self.shuffle = shuffle
        self.length = data.shape[0]
        self.batch_size = batch_size
        self.transforms = transforms
    def __iter__(self):
        for i in range(self.length // self.batch_size):
            data, targets = self.data[i * self.batch_size : (i+1) * self.batch_size], self.targets[i * self.batch_size : (i+1) * self.batch_size]
            data = self.transforms(data)
            yield data, targets
        if self.length % self.batch_size != 0:
            data, targets = self.data[self.length - (self.length % self.batch_size):], self.targets[self.length - (self.length % self.batch_size):]
            data = self.transforms(data)
            yield data, targets
    def __len__(self):
        return self.length
    def batch_length(self):
        return self.length // self.batch_size + (0 if self.length % self.batch_size == 0 else 1)

    
def mnist(batch_size):

    transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True, transform=transform),
        batch_size = batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=True, transform=transform),
        batch_size = batch_size, shuffle=False
    )
    
    return (train_loader, train_loader, test_loader), [1, 28, 28], 10

def cifar10(data_augmentation = True):
    train_loader = datasets.CIFAR10(args.dataset_path, train = True, download = True)
    train_data = torch.stack(list(map(transforms.ToTensor(), train_loader.data)))
    train_targets = torch.LongTensor(train_loader.targets)
    test_loader = datasets.CIFAR10(args.dataset_path, train = False, download = True)
    test_data = torch.stack(list(map(transforms.ToTensor(), test_loader.data)))
    test_targets = torch.LongTensor(test_loader.targets)
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if data_augmentation:
        list_trans_train = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), norm)
    else:
        list_trans_train = norm
    train_loader = BatchGenerator(train_data, train_targets, transforms = list_trans_train)
    val_loader = Dataset(train_data, train_targets, transforms = norm)
    test_loader = Dataset(test_data, test_targets, transforms = norm)
    return (train_loader, val_loader, test_loader), [3, 32, 32], 10


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_maps, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = feature_maps
        self.length = len(num_blocks)
        self.conv1 = nn.Conv2d(3, feature_maps, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        layers = []
        for i, nb in enumerate(num_blocks):
            layers.append(self._make_layer(block, (2 ** i) * feature_maps, nb, stride = 1 if i == 0 else 2))            
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear((2 ** (len(num_blocks) - 1)) * feature_maps * block.expansion, num_classes)        
        self.depth = len(num_blocks)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            if i < len(strides) - 1:
                layers.append(nn.ReLU())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        return out, features


import random
last_update = 0

def train(model, train_loader, test_loader, optimizer, steps, test_interrupt = 60, mixup = "None"):
    model.train()
    if mixup == "local":
        criterion = torch.nn.CrossEntropyLoss(reduction = "none")
    global last_update
    last_test = time.time()
    losses = []
    for batch_idx in range(steps):
        data, target = train_loader.generateBatch()
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        
        if mixup == "standard" or mixup == "local":
            index_mixup = torch.randperm(data.shape[0])
        if mixup == "local":
            data_v = data.reshape(data.shape[0], -1)
            distances = torch.norm(data_v - data_v[index_mixup], p = 2, dim = 1)
            sim = torch.exp( -1 * args.alpha * distances * distances)
            #sim = (distances <= args.threshold).float()
        if mixup == "standard" or mixup == "local":
            lam = random.random()
            data_mixed = lam * data + (1 - lam) * data[index_mixup]
            output, _ = model(data_mixed)
        if mixup == "standard":
            loss = lam * criterion(output, target) + (1 - lam) * criterion(output, target[index_mixup])            
        if mixup == "local":
            loss = (sim * (lam * criterion(output, target) + (1 - lam) * criterion(output, target[index_mixup]))).sum() / sim.sum()
        else:
            output, _ = model(data)
            loss = criterion(output, target)
            
        loss.backward()
        losses.append(loss.item())
        losses = losses[-1000:]
        optimizer.step()
        if time.time() - last_update > 0.1 and not args.quiet:
            print("\r{:9d}/{:9d} loss: {:.5f} time: {:s} ".format(batch_idx + 1, steps, np.mean(losses), format_time(time.time() - start_time)), end = "")
            last_update = time.time()
        if time.time() - last_test > test_interrupt:
            test_scores = test(model, test_loader)
            if args.quiet:
                print("{:9d}/{:9d} ".format(batch_idx + 1, steps), end = '')
            print("Test loss: {:.5f}, accuracy: {:.2f}%".format(test_scores["test_loss"], 100 * test_scores["test_acc"]))
            last_test = time.time()
    if not args.quiet:
        print()
    return { "train_loss" : np.mean(losses)}

def test(model, test_loader):
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output, _ = model(data)
            test_loss += criterion(output, target).item() * data.shape[0]
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
    model.train()
    return { "test_loss" : test_loss / len(test_loader), "test_acc" : accuracy / len(test_loader) }


def save_features(loader, filename, pre = False):
    all_features = []
    all_targets = []
    elements_per_class = torch.zeros(10, dtype=torch.long)
    for batch_idx, (data, target) in enumerate(loader):        
        with torch.no_grad():
            data, target = data.to(args.device), target.to(args.device)
            output, features = model(data)
            dim = features.shape[1]
            all_features.append(features.cpu())
            all_targets.append(target.cpu())
            for i in range(target.shape[0]):
                elements_per_class[target[i]] += 1

    max_elements_per_class = torch.max(elements_per_class).item()
    features = torch.zeros((10, max_elements_per_class, dim))
    elements_per_class[:] = 0
    for i in range(len(all_features)):
        for j in range(all_features[i].shape[0]):
            features[all_targets[i][j], elements_per_class[all_targets[i][j]]] = all_features[i][j]
            elements_per_class[all_targets[i][j]] += 1
    torch.save(features, filename)


def format_time(duration):
    duration = int(duration)
    s = duration % 60
    m = (duration // 60) % 60
    h = (duration // 3600)
    return "{:4d}h{:02d}m{:02d}s".format(h,m,s)

def train_complete(model, training, loaders, mixup = False):
    global start_time
    start_time = time.time()
    train_loader, val_loader, test_loader = loaders
    for era, (steps, lr) in enumerate(training):
        if lr < 0:
            optimizer = torch.optim.Adam(model.parameters(), lr = -1 * lr, weight_decay = 5e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)
        train_stats = train(model, train_loader, test_loader, optimizer, steps, mixup = mixup)
        test_stats = test(model, test_loader)
        print("Era: {:3d}, test_acc: {:.2f}%, train_loss: {:.5f} time: {:s}".format(era, 100 * test_stats["test_acc"], train_stats["train_loss"], format_time(time.time() - start_time)))
    return test_stats["test_acc"]


training = eval(args.training)
criterion = torch.nn.CrossEntropyLoss()
loaders, _, _ = cifar10(data_augmentation = True)
train_loader, _ , _ = loaders
training = list(map(lambda x: (x[0] * len(train_loader) // args.batch_size, x[1]), training))

scores = []

for i in range(args.runs):
    print(args)
    model = ResNet(BasicBlock, [2, 2, 2, 2], args.feature_maps).to(args.device)
    scores.append(100 * train_complete(model, training, loaders, mixup = args.mixup))
    if i == 0:
        bel, up = float('nan'), float('nan')
    elif i < 30:
        bel, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        bel, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    print("run {:4d}/{:5d}, {:.5f}%, ({:.6f}%,{:.6f}%)".format(i, args.runs, np.mean(scores), bel, up))
