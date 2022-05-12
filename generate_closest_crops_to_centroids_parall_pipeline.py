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
    return elt, torch.Tensor([params[0], params[1], params[2], params[3], h, w, K_resize]).unsqueeze(0)

def miniImageNet_standardTraining(use_hd = True):
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
    test = [[], []]
    for c in range(len(classes)):
        train[0] = train[0]+data[nb_element_per_class*c:nb_element_per_class*c+500]
        train[1] = train[1]+target[nb_element_per_class*c:nb_element_per_class*c+500]
        test[0] = test[0]+data[nb_element_per_class*c+500:nb_element_per_class*(c+1)]
        test[1] = test[1]+target[nb_element_per_class*c+500:nb_element_per_class*(c+1)]
        
    datasets['train'] = [train[0], torch.LongTensor(train[1])]
    datasets['test'] = [test[0], torch.LongTensor(test[1])]
    print()
    
    return datasets['train'] #[data, torch.LongTensor(target)]

def freeze(model):
        for p in model.parameters():
            p.requires_grad = False

if __name__ == '__main__':

    from args import args

    fix_seed(args.seed)
    print('seed:', args.seed)

    datasets = miniImageNet_standardTraining()

    features = torch.load(args.save_features, map_location='cpu')[:, :500]
    centroids = features.mean(dim=1)

    # Get the model 
    model = ResNet12(args.feature_maps, [3, 84, 84], 100, True, args.rotations).to(args.device)
    model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
    model.to(args.device)
    model.eval()
    print()
    
            
    freeze(model)
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))

    closest_crops = []
    with torch.no_grad():
        for i in tqdm(range(len(datasets[0])//args.batch_size)):
            img_path, classe = datasets[0][i*args.batch_size:(i+1)*args.batch_size], datasets[1][i*args.batch_size:(i+1)*args.batch_size] 
            images = [transforms.ToTensor()(np.array(Image.open(path).convert('RGB'))) for path in img_path]
            class_centroids = torch.stack([centroids[c] for c in classe]).to(args.device)
            min_distances = torch.Tensor([10e4]*args.batch_size).cuda()
            best_params = torch.zeros(args.batch_size, 7)
            for K_resize in [84, 110, 128, 164]:
                augmentations = [transforms.RandomResizedCrop(84), transforms.Resize([K_resize, K_resize]), norm]
                for _ in range(args.sample_aug):
                    elt_params = [crop_resize_rescale_image(img, augmentations, K_resize) for img in images] 
                    elt, params = list(zip(*elt_params))
                    elt = torch.stack(elt)
                    params = torch.cat(params)
                    elt = elt.to(args.device)
                    _, features = model(elt)
                    # compute distance of feature to class centroid
                    distance = torch.sqrt((features - class_centroids).pow(2).sum(1)) #torch.norm(features-class_centroids, p=2, dim=1)
                    mask = (min_distances-distance>0)
                    best_params[mask] = params[mask]
                    min_distances[mask] = distance[mask]
            closest_crops.append(best_params)
    closest_crops = torch.cat(closest_crops)

    torch.save(closest_crops, args.closest_crops)
