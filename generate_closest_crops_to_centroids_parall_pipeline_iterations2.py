import os
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
from torchvision import transforms, datasets
import torch
from PIL import Image
from resnet12 import ResNet12
from args import args


def normalized_bb_intersection_over_union(boxAA, boxBB):
    """
    
    """
    boxA = [boxAA[0], boxAA[1], boxAA[2], boxAA[3]]
    boxB = [boxBB[0], boxBB[1], boxBB[2], boxBB[3]]
    boxA[2] = boxA[0]+boxA[2]
    boxA[3] = boxA[1]+boxA[3] 
    boxB[2] = boxB[0]+boxB[2]
    boxB[3] = boxB[1]+boxB[3]
    
    # (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if args.niou_asymmetric:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(min(boxAArea, boxBArea))
    # return the intersection over union value
    return iou

def select_crop(elt, transformations, closest_crop):
    h, w = elt.shape[-2],  elt.shape[-1]
    crop = transformations[0]
    params = crop.get_params(elt, scale=(0.08,1), ratio=(0.75, 1.333333)) # sample some parameter
    NIOU = normalized_bb_intersection_over_union(closest_crop, params)
    while NIOU < args.niou_treshold:
        params = crop.get_params(elt, scale=(0.08,1), ratio=(0.75, 1.333333)) # sample some parameter
        NIOU = normalized_bb_intersection_over_union(closest_crop, params)
    elt = transforms.functional.crop(elt, *params)
    elt = torch.nn.Sequential(*transformations[1:])(elt)
    return elt


def fix_seed(seed, deterministic=False):
    ### generate random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def crop_resize_rescale_image(elt, transformations, K_resize):
    h, w = elt.shape[-2],  elt.shape[-1]
    crop = transformations[0]
    params = crop.get_params(elt, scale=(0.14,1), ratio=(0.75, 1.333333)) # sample some parameter
    elt = transforms.functional.crop(elt, *params)
    elt = torch.nn.Sequential(*transformations[1:])(elt)
    return elt, torch.Tensor([params[0], params[1], params[2], params[3], h, w, K_resize])

class CPUDataset():
    def __init__(self, data, targets, transforms = [], batch_size = args.batch_size, use_hd = False, closest_crops=None, K_resize=0, centroids=None):
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
        self.closest_crops = closest_crops
        self.K_resize = K_resize
        self.centroids = centroids

    def __getitem__(self, idx):
        if self.use_hd:
            elt = transforms.ToTensor()(np.array(Image.open(self.data[idx]).convert('RGB')))
        else:
            elt = self.data[idx]

        if self.centroids != None:
            elt, params = crop_resize_rescale_image(elt, self.transforms, self.K_resize)
            return elt, params, self.centroids[self.targets[idx]]
        else:
            if self.closest_crops != None:
                elt = select_crop(elt, self.transforms, self.closest_crops[idx])
            else:
                elt = self.transforms(elt)
            return elt, self.targets[idx]

    def __len__(self):
        return self.length

def iterator(data, target, transforms, forcecpu = False, shuffle = True, use_hd = False, closest_crops=None, centroids=None, K_resize=0):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = CPUDataset(data, target, transforms, use_hd = use_hd, closest_crops=closest_crops, centroids=centroids, K_resize=K_resize)
        return torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(8, os.cpu_count()))
    else:
        return Dataset(data, target, transforms, shuffle = shuffle)

def miniImageNet_standardTraining(closest_crops=None, use_hd = True, K_resize=84, centroids=None):
    datasets = {}
    classes = []
    count = 0
    target = []
    data = []
    nb_element_per_class = 600
    # Retrieve images and their classes
    for subset in ["train", "validation", "test"]:
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
    # Split each class to train and test 
    train = [[], []]
    for c in range(len(classes)):
        train[0] = train[0]+data[nb_element_per_class*c:nb_element_per_class*c+500]
        train[1] = train[1]+target[nb_element_per_class*c:nb_element_per_class*c+500]
        
    datasets['train'] = [train[0], torch.LongTensor(train[1])]

    print()
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    if closest_crops == None and centroids==None:
        train_transforms = transforms.Compose([transforms.RandomResizedCrop(K_resize), norm])
    else:
        train_transforms = [transforms.RandomResizedCrop(K_resize), transforms.Resize([K_resize, K_resize]), norm]
    if closest_crops != None:
        closest_crops = closest_crops.reshape(100*500, -1) 
    train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms = train_transforms, forcecpu = True, use_hd = use_hd, shuffle=False, closest_crops=closest_crops, centroids=centroids, K_resize=K_resize) # IMPORTANT TO NOT SHUFFLE
    return datasets['train'], train_loader

def freeze(model):
        for p in model.parameters():
            p.requires_grad = False

def get_features(model, loader, n_aug = args.sample_aug):
    model.eval()
    for augs in tqdm(range(n_aug)):
        all_features, offset, max_offset = [], 1000000, 0
        for batch_idx, (data, target) in enumerate(loader):        
            with torch.no_grad():
                data, target = data.to(args.device), target.to(args.device)
                _, features = model(data)
                all_features.append(features)
                offset = min(min(target), offset)
                max_offset = max(max(target), max_offset)
        num_classes = max_offset - offset + 1
        print(".", end='')
        if augs == 0:
            features_total = torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
        else:
            features_total += torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
    return features_total / n_aug

def generate_closest_crop_to_centroid(model, centroids, dataset):
    # Generate K_resize crops of the centroid
    min_distances = torch.Tensor([10e4]*len(dataset[0])).to(args.device)
    best_params = torch.zeros(len(dataset[0]), 7).to(args.device)
    
    for K_resize in [84, 92, 100, 110, 128, 164, 184]:
        _, loader = miniImageNet_standardTraining(closest_crops=None, K_resize=K_resize, centroids=centroids)
        for _ in tqdm(range(args.sample_aug)):
            distances = torch.Tensor().to(args.device)
            all_params = torch.Tensor().to(args.device)
            for batch_idx, (elt, params, class_centroids) in enumerate(loader):
                with torch.no_grad():
                    elt = elt.to(args.device)
                    class_centroids = class_centroids.to(args.device)
                    params = params.to(args.device)
                    _, features = model(elt)
                    # compute distance of feature to class centroid
                    distance = torch.sqrt((features - class_centroids).pow(2).sum(1)) 
                    distances = torch.cat([distances, distance], dim = 0)
                    all_params = torch.cat([all_params, params], dim = 0)
            mask = (min_distances-distances>0)
            best_params[mask] = all_params[mask]
            min_distances[mask] = distances[mask]
    return best_params

if __name__ == '__main__':
    fix_seed(args.seed)
    print('seed:', args.seed)

    datasets, _ = miniImageNet_standardTraining(closest_crops=None)

    features = torch.load(args.save_features, map_location='cpu')[:, :500]
    centroids = features.mean(dim=1).to(args.device)

    # Get the model 
    model = ResNet12(args.feature_maps, [3, 84, 84], 100, True, args.rotations).to(args.device)
    model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
    model.to(args.device)
    if len(args.devices) > 1:
        model = torch.nn.DataParallel(model, device_ids = args.devices)
    model.eval()
    freeze(model)
    print()

    diff = 10e4
    previous_diff = 10
    for epoch in range(args.epochs):
        # Generate closest_crops
        closest_crops = generate_closest_crop_to_centroid(model, centroids.cpu(), datasets)
        torch.save(closest_crops, args.closest_crops+'_iter_'+str(epoch))
        # Generate new centroids
        _, loader = miniImageNet_standardTraining(closest_crops=closest_crops.cpu(), K_resize=84, centroids=None)
        new_centroids = get_features(model, loader, n_aug = 50).mean(dim=1) # generate 50 crops per image for AS
        previous_diff = diff*1
        diff = (new_centroids - centroids).pow(2).sum(1).mean()
        print('diff:', diff.item(), 'previous diff:', previous_diff, 'ratio:', (previous_diff/diff).item())
        centroids = new_centroids

    torch.save(closest_crops, args.closest_crops)

    
