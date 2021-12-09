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
        self.use_hd = use_hd
    def __getitem__(self, idx):
        if self.use_hd:
            elt = transforms.ToTensor()(np.array(Image.open(self.data[idx]).convert('RGB')))
        else:
            elt = self.data[idx]
        return self.transforms(elt), self.targets[idx]
    def __len__(self):
        return self.length
        
class EpisodicCPUDataset():
    def __init__(self, data, num_classes, transforms = [], episode_size = args.batch_size, use_hd = False):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.episode_size = (episode_size // args.n_ways) * args.n_ways
        self.transforms = transforms
        self.use_hd = use_hd
        self.num_classes = num_classes
        self.targets = []
        self.indices = []
        self.corrected_length = args.episodes_per_epoch * self.episode_size
        episodes = args.episodes_per_epoch
        for i in range(episodes):
            classes = np.random.permutation(np.arange(self.num_classes))[:args.n_ways]
            for c in range(args.n_ways):
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[:self.episode_size // args.n_ways]
                self.indices += list(class_indices + classes[c] * (self.length // self.num_classes))
                self.targets += [c] * (self.episode_size // args.n_ways)
        self.indices = np.array(self.indices)
        self.targets = np.array(self.targets)

    def generate_next_episode(self, idx):
        if idx >= args.episodes_per_epoch:
            idx = 0
        classes = np.random.permutation(np.arange(self.num_classes))[:args.n_ways]
        n_samples = (self.episode_size // args.n_ways)
        for c in range(args.n_ways):
            class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[:self.episode_size // args.n_ways]
            self.indices[idx * self.episode_size + c * n_samples: idx * self.episode_size + (c+1) * n_samples] = (class_indices + classes[c] * (self.length // self.num_classes))

    def __getitem__(self, idx):
        if idx % self.episode_size == 0:
            self.generate_next_episode((idx // self.episode_size) + 1)
        if self.use_hd:
            elt = transforms.ToTensor()(np.array(Image.open(self.data[self.indices[idx]]).convert('RGB')))
        else:
            elt = self.data[self.indices[idx]]
        return self.transforms(elt), self.targets[idx]

    def __len__(self):
        return self.corrected_length

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
        for i in range(self.n_batches):
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]), self.targets[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]
            else:
                yield torch.stack([self.transforms(self.data[x]) for x in self.permutation[i * self.batch_size : (i+1) * self.batch_size]]), self.targets[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]
    def __len__(self):
        return self.n_batches

class EpisodicDataset():
    def __init__(self, data, num_classes, transforms = [], episode_size = args.batch_size, device = args.dataset_device, use_hd = False):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(device)
        else:
            self.data = data
            self.length = len(self.data)
        self.episode_size = episode_size
        self.transforms = transforms
        self.num_classes = num_classes
        self.n_batches = args.episodes_per_epoch
        self.use_hd = use_hd
        self.device = device
    def __iter__(self):
        for i in range(self.n_batches):
            classes = np.random.permutation(np.arange(self.num_classes))[:args.n_ways]
            indices = []
            for c in range(args.n_ways):
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[:self.episode_size // args.n_ways]
                indices += list(class_indices + classes[c] * (self.length // self.num_classes))
            targets = torch.repeat_interleave(torch.arange(args.n_ways), self.episode_size // args.n_ways).to(self.device)
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[indices]), targets
            else:
                if self.use_hd:
                    yield torch.stack([self.transforms(transforms.ToTensor()(np.array(Image.open(self.data[x]).convert('RGB'))).to(self.device)) for x in indices]), targets
                else:
                    yield torch.stack([self.transforms(self.data[x].to(self.device)) for x in indices]), targets
    def __len__(self):
        return self.n_batches

def iterator(data, target, transforms, forcecpu = False, shuffle = True, use_hd = False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = CPUDataset(data, target, transforms, use_hd = use_hd)
        return torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(8, os.cpu_count()))
    else:
        return Dataset(data, target, transforms, shuffle = shuffle)

def episodic_iterator(data, num_classes, transforms, forcecpu = False, use_hd = False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = EpisodicCPUDataset(data, num_classes, transforms, use_hd = use_hd)
        return torch.utils.data.DataLoader(dataset, batch_size = (args.batch_size // args.n_ways) * args.n_ways, shuffle = False, num_workers = min(8, os.cpu_count()))
    else:
        return EpisodicDataset(data, num_classes, transforms, use_hd = use_hd)

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

from PIL import Image

def cifarfs(use_hd=True, data_augmentation=True):
    """
    CIFAR FS dataset
    Number of classes : 
    - train: 64
    - val  : 16
    - novel: 20
    Number of samples per class: exactly 600
    Total number of images: 60000
    Images size : 32x32
    """
    datasets = {}
    classes = []
    total = 60000
    buffer = {'train':0, 'val':64, 'test':64+16}
    for metaSub in ["meta-train", "meta-val", "meta-test"]:
        subset = metaSub.split('-')[-1]
        data = []
        target = []
        subset_path = os.path.join(args.dataset_path, 'cifar_fs', metaSub)
        classe_files = os.listdir(subset_path)
        
        for c, classe in enumerate(classe_files):
            files = os.listdir(os.path.join(subset_path, classe))
            count = 0
            for file in files:
                count += 1
                target.append(c+buffer[subset])
             
                path = os.path.join(subset_path, classe, file)
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    data.append(image)
                else:
                    data.append(path)
                  
        datasets[subset] = [data, torch.LongTensor(target)]
            
    assert (len(datasets['train'][0])+len(datasets['val'][0])+len(datasets['test'][0])==total), 'Total number of sample per class is not 600'
    print()
    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    if data_augmentation:
        train_transforms = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), norm)
    else:
        train_transforms = norm
    if args.episodic:
        train_loader = episodic_iterator(datasets['train'][0], 64, transforms = train_transforms, forcecpu=True, use_hd=True)
    else:
        train_loader = iterator(datasets['train'][0], datasets['train'][1], transforms = train_transforms, forcecpu=True, use_hd = use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms = norm, forcecpu = True, shuffle = False, use_hd = use_hd)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms = norm, forcecpu = True, shuffle = False, use_hd = use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms = norm, forcecpu = True, shuffle = False, use_hd = use_hd)
    return (train_loader, train_clean, val_loader, test_loader), [3, 32, 32], (64, 16, 20, 600), True, False

import os.path as osp

class MiniImageNet(Dataset):

    def __init__(self, setname_x, args):
        IMAGE_PATH = os.path.join(args.dataset_path, 'miniimagenet2/miniimagenet/images')
        SPLIT_PATH = os.path.join(args.dataset_path, 'miniimagenet2/miniimagenet/split')
        if setname_x == 'train_clean':
            setname = 'train'
        else:
            setname = setname_x 
        
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
<<<<<<< HEAD
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))

        if setname_x == 'val' or setname_x == 'test' or setname_x=='train_clean':
            image_size = 84
            size_transform = transforms.Compose([transforms.Resize(92),
                                            transforms.CenterCrop(image_size)])
            self.transform = transforms.Compose([
                size_transform,
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            
        elif setname_x == 'train':
            image_size = 84
            size_transform = transforms.RandomResizedCrop(image_size)
            
            self.transform = transforms.Compose([
                size_transform,
                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            
        else:
            raise ValueError('no such set')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


def miniImageNet(use_hd = True):
    
    train_loader = torch.utils.data.DataLoader(MiniImageNet('train', args), batch_size=args.batch_size, shuffle=True, num_workers = min(8, os.cpu_count()))
    train_clean  = torch.utils.data.DataLoader(MiniImageNet('train_clean', args), batch_size=args.batch_size, shuffle=False, num_workers = min(8, os.cpu_count()))
    val_loader   = torch.utils.data.DataLoader(MiniImageNet('val', args), batch_size=args.batch_size, shuffle=False, num_workers = min(8, os.cpu_count()))
    test_loader  = torch.utils.data.DataLoader(MiniImageNet('test', args), batch_size=args.batch_size, shuffle=False, num_workers = min(8, os.cpu_count()))
    
=======
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
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomResizedCrop(84, scale=(0.14,1)), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets["train"][0], 64, transforms = train_transforms, forcecpu = True, use_hd = True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms = train_transforms, forcecpu = True, use_hd = use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    val_loader = iterator(datasets["validation"][0], datasets["validation"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
>>>>>>> b3dc8974b2ac5ac0425bbfadd6fba26df3574bd6
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (64, 16, 20, 600), True, False


def tieredImageNet(use_hd=True):
    """
    tiredImagenet dataset
    Number of classes : 
    - train: 351
    - val  : 97
    - novel: 160
    Number of samples per class: at most 1300
    Total number of images: 790400
    Images size : 84x84
    """
    datasets = {}
    total = 790400
    num_elements = {}
    buffer = {'train':0, 'val':351, 'test':351+97}
    for subset in ['train', 'val', 'test']:
        data = []
        target = []
        num_elements[subset]=[]
        if subset=='train':
            data_train = []
            target_train = []
        subset_path = os.path.join(args.dataset_path, 'tieredimagenet', subset)
        classe_files = os.listdir(subset_path)
        
        for c, classe in enumerate(classe_files):
            files = os.listdir(os.path.join(subset_path, classe))
            count = 0
            for file in files:
                count += 1
                target.append(c+buffer[subset])
                if subset=='train':
                    target_train.append(c)
                path = os.path.join(subset_path, classe, file)
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    data.append(image)
                    if subset=='train':
                        data_train.append(image)
                else:
                    data.append(path)
                    if subset=='train':
                        data_train.append(path)
            num_elements[subset].append(count)
            if count<1300:
                for i in range(1300-count):
                    target.append(c+buffer[subset]) 
                    if not use_hd: # add the same element
                        image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                        data.append(image)
                    else:
                        data.append(path) 
                        
        datasets[subset] = [data, torch.LongTensor(target)]

    datasets['train_base']=[data_train, torch.LongTensor(target_train)] # clean train without duplicates
            
    assert (len(datasets['train'][0])+len(datasets['val'][0])+len(datasets['test'][0])==total), 'Total number of sample per class is not 1300'
    print()
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomResizedCrop(84, scale=(0.14,1)), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets["train_base"][0], 351, transforms = train_transforms, forcecpu = True, use_hd = True)
    else:
        train_loader = iterator(datasets["train_base"][0], datasets["train_base"][1], transforms = train_transforms, forcecpu = True, use_hd = use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (351, 97, 160, (num_elements['train'], num_elements['val'], num_elements['test'])), True, False

import pickle

def CUBfs(use_hd=False):
    """
    CUB FS dataset
    Number of classes : 
    - train: 100
    - val  : 50
    - novel: 50
    Number of samples per class: at most 60
    Images size : 84x84
    """
    classes      = []
    datasets     = {}
    num_elements = {}
    path         = os.path.join(args.dataset_path, 'cub')
    list_files = os.listdir(path)

    for subset in ['train', 'val', 'test']:
        data   = []
        class_counter = {}
        target = []
        csv_path = os.path.join(path, 'split', f'{subset}.csv')
        with open(csv_path, "r") as f:
            start = 0
            for line in f:
                if start == 0:
                    start += 1
                else:
                    splits = line.split(",")
                    fn, c = splits[0], splits[1]
                    if fn in list_files:
                        c = int(c.split('.')[0])
                        if c not in classes:
                            classes.append(c)
                            class_counter[len(classes)-1]=1
                        else:
                            class_counter[len(classes)-1]+=1
                        target.append(len(classes)-1)
                        file_path = os.path.join(args.dataset_path, 'cub', fn)
                        if not use_hd:
                            image = transforms.ToTensor()(np.array(Image.open(file_path).convert('RGB')))
                            data.append(image)
                        else:
                            data.append(file_path)
        if subset == 'train':
            datasets['train_base'] = [data.copy(), torch.LongTensor(target)]
        
        for c, count in class_counter.items():
            if count < 60:
                for _ in range(60-count):
                    if not use_hd:
                        data.append(image)
                    else:
                        data.append(file_path)
                    target.append(c)
        datasets[subset] = [data, torch.LongTensor(target)]
        num_elements[subset] = list(class_counter.values())
    print()
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84), transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomResizedCrop(84, scale=(0.14,1)), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets['train_base'][0], 100, transforms = train_transforms, forcecpu = True, use_hd = True)
    else:
        train_loader = iterator(datasets['train_base'][0], datasets['train_base'][1], transforms = train_transforms, forcecpu = True, use_hd = use_hd)
    train_clean = iterator(datasets['train'][0], datasets['train'][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    val_loader = iterator(datasets['val'][0], datasets['val'][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    test_loader = iterator(datasets['test'][0], datasets['test'][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (100, 50, 50, (num_elements['train'], num_elements['val'], num_elements['test'])), True, False

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
    all_transforms = torch.nn.Sequential(transforms.CenterCrop(100), transforms.Normalize((0.0782), (0.2685))) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomCrop(100, padding = 4), transforms.Normalize((0.0782) ,(0.2685)))
    if args.episodic:
        train_loader = episodic_iterator(base_data, base.shape[0], transforms = train_transforms)
    else:
        train_loader = iterator(base_data, base_targets, transforms = train_transforms)
    train_clean = iterator(base_data, base_targets, transforms = all_transforms, shuffle = False)
    val_loader = iterator(val_data, val_targets, transforms = all_transforms, shuffle = False)
    test_loader = iterator(novel_data, novel_targets, transforms = all_transforms, shuffle = False)
    return (train_loader, train_clean, val_loader, test_loader), [1, 100, 100], (base.shape[0], val.shape[0], novel.shape[0], novel.shape[1]), True, False

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
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomResizedCrop(84), norm)
    if args.episodic:
        train_loader = episodic_iterator(train, 64, transforms = train_transforms, forcecpu = True)
    else:
        train_loader = iterator(train, train_targets, transforms = train_transforms, forcecpu = True)
    train_clean = iterator(train, train_targets, transforms = all_transforms, forcecpu = True, shuffle = False)
    val_loader = iterator(validation, validation_targets, transforms = all_transforms, forcecpu = True, shuffle = False)
    test_loader = iterator(test, test_targets, transforms = all_transforms, forcecpu = True, shuffle = False)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (64, 16, 20, 600), True, False

def get_dataset(dataset_name):
    if dataset_name.lower() == "cifar10":
        return cifar10(data_augmentation = True)
    elif dataset_name.lower() == "cifar100":
        return cifar100(data_augmentation = True)
    elif dataset_name.lower() == "cifarfs":
        return cifarfs(data_augmentation = True)
    elif dataset_name.lower() == "mnist":
        return mnist()
    elif dataset_name.lower() == "fashion":
        return fashion_mnist()
    elif dataset_name.lower() == "miniimagenet":
        return miniImageNet()
    elif dataset_name.lower() == "miniimagenet84":
        return miniImageNet84()
    elif dataset_name.lower() == "cubfs":
        return CUBfs()
    elif dataset_name.lower() == "omniglotfs":
        return omniglotfs()
    elif dataset_name.lower() == "tieredimagenet":
        return tieredImageNet()
    else:
        print("Unknown dataset!")

print("datasets, ", end='')
