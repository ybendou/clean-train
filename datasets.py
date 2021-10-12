from torchvision import transforms, datasets
from args import args
import numpy as np
import torch
import json
import os

class CPUDataset():
    def __init__(self, data, targets, transforms = [], batch_size = args.batch_size, use_hd = False):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.targets = targets
        assert(self.length == targets.shape[0])
        self.batch_size = batch_size
        self.transforms = transforms
        self.n_batches = self.length // self.batch_size + (0 if self.length % self.batch_size == 0 else 1)
        self.use_hd = use_hd
    def __getitem__(self, idx):
        if self.use_hd:
            elt = transforms.ToTensor()(np.array(Image.open(self.data[idx]).convert('RGB')))
        else:
            elt = self.data[idx]
        return self.transforms(elt), self.targets[idx]
    def __len__(self):
        return self.length

class Dataset():
    def __init__(self, data, targets, transforms = [], batch_size = args.batch_size, shuffle = True, device = args.dataset_device):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(device)
        else:
            self.length = len(self.data)
        self.targets = targets.to(device)
        assert(self.length == targets.shape[0])
        self.batch_size = batch_size
        self.transforms = transforms
        self.permutation = torch.arange(self.length)
        self.n_batches = self.length // self.batch_size + (0 if self.length % self.batch_size == 0 else 1)
        self.shuffle = shuffle
    def __iter__(self):
        if self.shuffle:
            self.permutation = torch.randperm(self.length)
        for i in range(self.length // self.batch_size + (1 if self.length % self.batch_size != 0 else 0)):
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]), self.targets[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]
            else:
                yield torch.stack([self.transforms(self.data[x]) for x in self.permutation[i * self.batch_size : (i+1) * self.batch_size]]), self.targets[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]
    def __len__(self):
        return self.n_batches

class EpisodicDataset():
    def __init__(self, data, num_classes, transforms = [], episod_size = args.batch_size, device = args.dataset_device, use_hd = False):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(device)
        else:
            self.data = data
            self.length = len(self.data)
        self.episod_size = episod_size
        self.transforms = transforms
        self.num_classes = num_classes
        self.n_batches = self.length // self.episod_size
        self.use_hd = use_hd
        self.device = device
    def __iter__(self):
        for i in range(self.n_batches):
            classes = np.random.permutation(np.arange(self.num_classes))[:args.n_ways]
            indices = []
            for c in range(args.n_ways):
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[:self.episod_size // 5]
                indices += list(class_indices + classes[c] * (self.length // self.num_classes))
            targets = torch.repeat_interleave(torch.tensor(classes), self.episod_size // 5).to(self.device)
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[indices]), targets
            else:
                if self.use_hd:
                    yield torch.stack([self.transforms(transforms.ToTensor()(np.array(Image.open(self.data[x]).convert('RGB')))) for x in indices]), targets
                else:
                    yield torch.stack([self.transforms(self.data[x]) for x in indices]), targets
    def __len__(self):
        return self.n_batches

def iterator(data, target, transforms, forcecpu = False, shuffle = True, use_hd = False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = CPUDataset(data, target, transforms, use_hd = use_hd)
        return torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(8, os.cpu_count()))
    else:
        return Dataset(data, target, transforms, shuffle = shuffle)

def create_dataset(train_data, test_data, train_targets, test_targets, train_transforms, test_transforms):
    train_loader = iterator(train_data[:args.dataset_size], train_targets[:args.dataset_size], transforms = train_transforms)
    val_loader = iterator(train_data, train_targets, transforms = test_transforms)
    test_loader = iterator(test_data, test_targets, transforms = test_transforms)
    return train_loader, val_loader, test_loader

import random
def mnist():
    train_loader = datasets.MNIST(args.dataset_path, train = True, download = True)
    train_data = (train_loader.data.float() / 256).unsqueeze(1)
    train_targets = torch.LongTensor(train_loader.targets.clone())
    if args.dataset_size >= 0:
        data_per_class = []
        test = []
        for i in range(10):
            data_per_class.append(train_data[torch.where(train_targets == i)[0]][:args.dataset_size // 10])
            test.append(torch.zeros(args.dataset_size // 10) + i)
        train_data = torch.stack(data_per_class, dim = 1).view(args.dataset_size, 1, 28, 28)
        train_targets = torch.arange(10).repeat(args.dataset_size // 10)
    test_loader = datasets.MNIST(args.dataset_path, train = False, download = True)
    test_data = (test_loader.data.float() / 256).unsqueeze(1)
    test_targets = torch.LongTensor(test_loader.targets.clone())
    all_transforms = transforms.Normalize((0.1302,), (0.3069,))
    loaders = create_dataset(train_data, test_data, train_targets, test_targets, all_transforms, all_transforms)
    return loaders, train_data.shape[1:], torch.max(train_targets).item() + 1, False, False

def fashion_mnist(data_augmentation = True):
    train_loader = datasets.FashionMNIST(args.dataset_path, train = True, download = True)
    train_data = (train_loader.data.float() / 256).unsqueeze(1)
    train_targets = torch.LongTensor(train_loader.targets)
    if args.dataset_size >= 0:
        data_per_class = []
        test = []
        for i in range(10):
            data_per_class.append(train_data[torch.where(train_targets == i)[0]][:args.dataset_size // 10])
            test.append(torch.zeros(args.dataset_size // 10) + i)
        train_data = torch.stack(data_per_class, dim = 1).view(args.dataset_size, 1, 28, 28)
        train_targets = torch.arange(10).repeat(args.dataset_size // 10)
    test_loader = datasets.FashionMNIST(args.dataset_path, train = False, download = True)
    test_data = (test_loader.data.float() / 256).unsqueeze(1)
    test_targets = torch.LongTensor(test_loader.targets)
    norm = transforms.Normalize((0.2849,), (0.3516,))
    if data_augmentation:
        list_trans_train = torch.nn.Sequential(transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), norm)
    all_transforms = norm
    loaders = create_dataset(train_data, test_data, train_targets, test_targets, list_trans_train, all_transforms)
    return loaders, train_data.shape[1:], torch.max(train_targets).item() + 1, False, False

def cifar10(data_augmentation = True):
    train_loader = datasets.CIFAR10(args.dataset_path, train = True, download = True)
    train_data = torch.stack(list(map(transforms.ToTensor(), train_loader.data)))
    train_targets = torch.LongTensor(train_loader.targets)
    if args.dataset_size >= 0:
        data_per_class = []
        test = []
        for i in range(10):
            data_per_class.append(train_data[torch.where(train_targets == i)[0]][:args.dataset_size // 10])
            test.append(torch.zeros(args.dataset_size // 10) + i)
        train_data = torch.stack(data_per_class, dim = 1).view(args.dataset_size, 3, 32, 32)
        train_targets = torch.arange(10).repeat(args.dataset_size // 10)
    test_loader = datasets.CIFAR10(args.dataset_path, train = False, download = True)
    test_data = torch.stack(list(map(transforms.ToTensor(), test_loader.data)))
    test_targets = torch.LongTensor(test_loader.targets)
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if data_augmentation:
        list_trans_train = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), norm)
    else:
        list_trans_train = norm
    loaders = create_dataset(train_data, test_data, train_targets, test_targets, list_trans_train, norm)
    return loaders, train_data.shape[1:], torch.max(train_targets).item() + 1, False, False

def cifar100(data_augmentation = True):
    train_loader = datasets.CIFAR100(args.dataset_path, train = True, download = True)
    train_data = torch.stack(list(map(transforms.ToTensor(), train_loader.data)))
    train_targets = torch.LongTensor(train_loader.targets)
    if args.dataset_size >= 0:
        data_per_class = []
        test = []
        for i in range(10):
            data_per_class.append(train_data[torch.where(train_targets == i)[0]][:args.dataset_size // 10])
            test.append(torch.zeros(args.dataset_size // 10) + i)
        train_data = torch.stack(data_per_class, dim = 1).view(args.dataset_size, 3, 32, 32)
        train_targets = torch.arange(10).repeat(args.dataset_size // 10)
    test_loader = datasets.CIFAR100(args.dataset_path, train = False, download = True)
    test_data = torch.stack(list(map(transforms.ToTensor(), test_loader.data)))
    test_targets = torch.LongTensor(test_loader.targets)
    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    if data_augmentation:
        list_trans_train = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), norm)
    else:
        list_trans_train = norm
    loaders = create_dataset(train_data, test_data, train_targets, test_targets, list_trans_train, norm)
    return loaders, train_data.shape[1:], torch.max(train_targets).item() + 1, False, True

def cifarfs(data_augmentation = True):
    novel_labels = ["baby","bed","bicycle","chimpanzee","fox","leopard","man","pickup_truck","plain","poppy","rocket","rose","snail","sweet_pepper","table","telephone","wardrobe","whale","woman","worm"]
    val_labels = ["otter","motorcycle","television","lamp","crocodile","shark","butterfly","beaver","beetle","tractor","flatfish","maple_tree","camel","crab","sea","cattle"]
    data_train = datasets.CIFAR100(args.dataset_path, train=True, download=True)
    data_val = datasets.CIFAR100(args.dataset_path, train=False, download=True)
    all_data = torch.cat([torch.stack(list(map(transforms.ToTensor(), data_train.data))), torch.stack(list(map(transforms.ToTensor(), data_val.data)))], dim = 0)
    all_labels = torch.cat([torch.LongTensor(data_train.targets), torch.LongTensor(data_val.targets)], dim = 0)
    novel_targets = [data_train.class_to_idx[label] for label in novel_labels]    
    val_targets = [data_train.class_to_idx[label] for label in val_labels]
    train_targets = [x for x in np.arange(100) if x not in novel_targets and x not in val_targets]
    new_labels = torch.zeros(all_labels.shape, dtype=torch.long)
    index = 0
    for dataset in [train_targets, val_targets, novel_targets]:
        for i in dataset:
            new_labels[np.where(all_labels == i)] = index
            index += 1
    all_labels = new_labels
    novel_targets = np.arange(80, 100)
    val_targets = np.arange(64, 80)
    train_targets = np.arange(0, 64)
    train_data = all_data[torch.where(all_labels < 64)[0]]
    train_targets = all_labels[torch.where(all_labels < 64)[0]]
    remaining_data = all_data[torch.where(all_labels >= 64)[0]]
    remaining_labels = all_labels[torch.where(all_labels >= 64)[0]]
    val_data = remaining_data[torch.where(remaining_labels < 80)[0]]
    val_targets = remaining_labels[torch.where(remaining_labels < 80)[0]]
    test_data = all_data[torch.where(all_labels >= 80)[0]]
    test_targets = all_labels[torch.where(all_labels >= 80)[0]]

    train_data = torch.cat([train_data[torch.where(train_targets == i)] for i in range(64)], dim = 0)
    val_data = torch.cat([val_data[torch.where(val_targets == i)] for i in range(64, 80)], dim = 0)
    test_data = torch.cat([test_data[torch.where(test_targets == i)] for i in range(80, 100)], dim = 0)
    train_targets = torch.repeat_interleave(torch.arange(64), 600)
    val_targets = torch.repeat_interleave(torch.arange(64, 80), 600)
    test_targets = torch.repeat_interleave(torch.arange(80, 100), 600)
    
    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    if data_augmentation:
        list_trans_train = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), norm)
    else:
        list_trans_train = norm
    if args.episodic:
        train_loader = EpisodicDataset(train_data, 64, transforms = list_trans_train)
    else:
        train_loader = iterator(train_data, train_targets, transforms = list_trans_train)
    val_loader = iterator(val_data, val_targets, transforms = norm, shuffle = False)
    test_loader = iterator(test_data, test_targets, transforms = norm, shuffle = False)
    return (train_loader, val_loader, test_loader), [3, 32, 32], (64, 16, 20, 600), True, False

from PIL import Image

def miniImageNet(use_hd = True):
    datasets = {}
    classes = []
    total = 60000
    count = 0
    for subset in ["train", "validation", "test"]:
        data = []
        target = []
        with open(args.dataset_path + "miniimagenetimages/" + subset + ".csv", "r") as f:
            start = 0
            for line in f:
                if start == 0:
                    start += 1
                else:
                    splits = line.split(",")
                    fn, c = splits[0], splits[1]
                    if c not in classes:
                        classes.append(c)
                    count += 1
                    target.append(len(classes) - 1)
                    path = args.dataset_path + "miniimagenetimages/" + "images/" + fn
                    if not use_hd:
                        image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                        data.append(image)
                    else:
                        data.append(path)
        datasets[subset] = [data, torch.LongTensor(target)]
    print()
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm)
    if args.episodic:
        train_loader = EpisodicDataset(datasets["train"][0], 64, transforms = train_transforms, device = "cpu", use_hd = True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms = train_transforms, forcecpu = True, use_hd = use_hd)
    val_loader = iterator(datasets["validation"][0], datasets["validation"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    return (train_loader, val_loader, test_loader), [3, 84, 84], (64, 16, 20, 600), True, False

import pickle

def CUBfs():
    with open(args.dataset_path + "CUB/base.pkl", "rb") as f:
        train_file = pickle.load(f)
    train, train_targets = [(x.float() / 256) for x in train_file['data']], torch.LongTensor(train_file['labels'])
    for i in range(len(train)):
        if train[i].shape[0] != 3:
            train[i] = train[i].repeat(3,1,1)
    if args.episodic:
        new_train = []
        num_elements_train = []
        for i in range(100):
            indices = torch.where(train_targets == i)[0]
            num_elements_train.append(len(indices))
            for x in indices:
                new_train.append(train[x])
            size = len(indices)
            if size < 60:
                for i in range(60 - size):
                    new_train.append(train[indices[i]])
        train, train_targets = [new_train, torch.arange(100).repeat_interleave(60)]
    with open(args.dataset_path + "CUB/val.pkl", "rb") as f:
        train_file = pickle.load(f)
    validation, validation_targets = [(x.float() / 256) for x in train_file['data']], torch.LongTensor(train_file['labels'])
    new_val = []
    num_elements_val = []
    for i in range(len(validation)):
        if validation[i].shape[0] != 3:
            validation[i] = validation[i].repeat(3, 1, 1)
    for i in range(100,150):
        indices = torch.where(validation_targets == i)[0]
        num_elements_val.append(len(indices))
        for x in indices:
            new_val.append(validation[x])
        size = len(indices)
        if size < 60:
            for i in range(60 - size):
                new_val.append(validation[indices[i]])
    validation, validation_targets = [new_val, torch.arange(100, 150).repeat_interleave(60)]
    with open(args.dataset_path + "CUB/novel.pkl", "rb") as f:
        train_file = pickle.load(f)
    novel, novel_targets = [(x.float() / 256) for x in train_file['data']], torch.LongTensor(train_file['labels'])
    for i in range(len(novel)):
        if novel[i].shape[0] != 3:
            novel[i] = novel[i].repeat(3, 1, 1)
    new_novel = []
    num_elements_novel = []
    for i in range(150,200):
        indices = torch.where(novel_targets == i)[0]
        for x in indices:
            new_novel.append(novel[x])
        num_elements_novel.append(len(indices))
        size = len(indices)
        if size < 60:
            for i in range(60 - size):
                new_novel.append(novel[indices[i]])
    novel, novel_targets = [new_novel, torch.arange(150, 200).repeat_interleave(60)]
    train_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), transforms.RandomHorizontalFlip(), transforms.Normalize((0.4770, 0.4921, 0.4186) ,(0.1805, 0.1792, 0.1898)))
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), transforms.Normalize((0.4770, 0.4921, 0.4186), (0.1805, 0.1792, 0.1898)))
    if args.episodic:
        train_loader = EpisodicDataset(train, 100, transforms = train_transforms, device = "cpu", use_hd = True)
    else:
        train_loader = iterator(train, train_targets, transforms = train_transforms, forcecpu = True)
    val_loader = iterator(validation, validation_targets, transforms = all_transforms, forcecpu = True, shuffle = False)
    test_loader = iterator(novel, novel_targets, transforms = all_transforms, forcecpu = True, shuffle = False)
    return (train_loader, val_loader, test_loader), [3, 84, 84], (100, 50, 50, (num_elements_val, num_elements_novel)), True, False

def omniglotfs():
    base = torch.load(args.dataset_path + "omniglot/base.pt")
    base_data = base.reshape(-1, base.shape[2], base.shape[3], base.shape[4]).float()
    base_targets = torch.arange(base.shape[0]).unsqueeze(1).repeat(1, base.shape[1]).reshape(-1)
    val = torch.load(args.dataset_path + "omniglot/val.pt")
    val_data = val.reshape(-1, val.shape[2], val.shape[3], val.shape[4]).float()
    val_targets = torch.arange(val.shape[0]).unsqueeze(1).repeat(1, val.shape[1]).reshape(-1)
    novel = torch.load(args.dataset_path + "omniglot/novel.pt")
    novel_data = novel.reshape(-1, novel.shape[2], novel.shape[3], novel.shape[4]).float()
    novel_targets = torch.arange(novel.shape[0]).unsqueeze(1).repeat(1, novel.shape[1]).reshape(-1)
    train_transforms = torch.nn.Sequential(transforms.RandomCrop(100, padding = 4), transforms.Normalize((0.0782) ,(0.2685)))
    all_transforms = torch.nn.Sequential(transforms.CenterCrop(100), transforms.Normalize((0.0782), (0.2685)))
    if args.episodic:
        train_loader = EpisodicDataset(base_data, base.shape[0], transforms = train_transforms)
    else:
        train_loader = iterator(base_data, base_targets, transforms = train_transforms)
    val_loader = iterator(val_data, val_targets, transforms = all_transforms, shuffle = False)
    test_loader = iterator(novel_data, novel_targets, transforms = all_transforms, shuffle = False)
    return (train_loader, val_loader, test_loader), [1, 100, 100], (base.shape[0], val.shape[0], novel.shape[0], novel.shape[1]), True, False

def miniImageNet84():
    with open(args.dataset_path + "miniimagenet/train.pkl", 'rb') as f:
        train_file = pickle.load(f)
    train, train_targets = [transforms.ToTensor()(x) for x in train_file["data"]], train_file["labels"]
    with open(args.dataset_path + "miniimagenet/test.pkl", 'rb') as f:
        test_file = pickle.load(f)
    test, test_targets = [transforms.ToTensor()(x) for x in test_file["data"]], test_file["labels"]
    with open(args.dataset_path + "miniimagenet/validation.pkl", 'rb') as f:
        validation_file = pickle.load(f)
    validation, validation_targets = [transforms.ToTensor()(x) for x in validation_file["data"]], validation_file["labels"]
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm)
    if args.episodic:
        train_loader = EpisodicDataset(train, 64, transforms = train_transforms, device = "cpu")
    else:
        train_loader = iterator(train, train_targets, transforms = train_transforms, forcecpu = True)
    val_loader = iterator(validation, validation_targets, transforms = all_transforms, forcecpu = True, shuffle = False)
    test_loader = iterator(test, test_targets, transforms = all_transforms, forcecpu = True, shuffle = False)
    return (train_loader, val_loader, test_loader), [3, 84, 84], (64, 16, 20, 600), True, False

def get_dataset(dataset_name):
    if args.dataset.lower() == "cifar10":
        return cifar10(data_augmentation = True)
    elif args.dataset.lower() == "cifar100":
        return cifar100(data_augmentation = True)
    elif args.dataset.lower() == "cifarfs":
        return cifarfs(data_augmentation = True)
    elif args.dataset.lower() == "mnist":
        return mnist()
    elif args.dataset.lower() == "fashion":
        return fashion_mnist()
    elif args.dataset.lower() == "miniimagenet":
        return miniImageNet()
    elif args.dataset.lower() == "miniimagenet84":
        return miniImageNet84()
    elif args.dataset.lower() == "cubfs":
        return CUBfs()
    elif args.dataset.lower() == "omniglotfs":
        return omniglotfs()
    else:
        print("Unknown dataset!")
