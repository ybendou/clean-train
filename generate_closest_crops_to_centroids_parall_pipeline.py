import os
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
import torch
import torch.nn as nn

def fix_seed(seed, deterministic=False):
    ### generate random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if args.dropout > 0:
            out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        return out
    
class ResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(ResNet12, self).__init__()
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps))
        self.layers = nn.Sequential(*layers)
        self.linear = linear(10 * feature_maps, num_classes)
        self.rotations = rotations
        self.linear_rot = linear(10 * feature_maps, 4)
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, index_mixup = None, lam = -1):
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = self.mp(F.leaky_relu(out, negative_slope = 0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features
        return out, features

from torchvision import transforms, datasets
import torch
import numpy as np
from PIL import Image

def crop_resize_rescale_image(elt, transformations, K_resize):
    h, w = elt.shape[-2],  elt.shape[-1]
    crop = transformations[0]
    params = crop.get_params(elt, scale=(0.14,1), ratio=(0.75, 1.333333)) # sample some parameter
    elt = transforms.functional.crop(elt, *params)
    elt = torch.nn.Sequential(*transformations[1:])(elt)
    return elt, torch.Tensor([params[0], params[1], params[2], params[3], h, w, K_resize]).unsqueeze(0)

def miniImageNet(use_hd = True):
    """
        Get the features of the miniImageNet dataset
    """
    datasets = {}
    count = 0
    subset = "train"
    data = []
    target = []
    classes = []
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
                # Get image for each summit
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    data.append(image)
                else:
                    data.append(path)
                count += 1
    datasets[subset] = [data, torch.LongTensor(target)]
    print('stats:',len(data), datasets[subset][1].shape)
    return datasets

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
    
    return [data, torch.LongTensor(target)]


def linear(indim, outdim):
    return nn.Linear(indim, outdim)
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
    for i in tqdm(range(len(datasets[0])//args.batch_size)):
        img_path, classe = datasets[0][i*args.batch_size:(i+1)*args.batch_size], datasets[1][i*args.batch_size:(i+1)*args.batch_size] 
        images = [transforms.ToTensor()(np.array(Image.open(path).convert('RGB'))) for path in img_path]
        class_centroids = torch.stack([centroids[c] for c in classe]).to(args.device)
        min_distances = [10e8]*args.batch_size
        best_params = torch.zeros(args.batch_size, 7)
        for K_resize in [84, 92, 100, 110, 128, 164, 184]:
            for _ in range(args.sample_aug):
                augmentations = [transforms.RandomResizedCrop(84), transforms.Resize([K_resize, K_resize]), norm]
                elt_params = [crop_resize_rescale_image(img, augmentations, K_resize) for img in images] 
                elt, params = list(zip(*elt_params))
                elt = torch.stack(elt)
                params = torch.cat(params)
                
                with torch.no_grad():
                    elt = elt.to(args.device)
                    _, features = model(elt)
                # compute distance of feature to class centroid
                distance = torch.norm(features-class_centroids, p=2, dim=1)
                for b in range(args.batch_size):
                    if distance[b].item()<min_distances[b]:
                        min_distances[b] = distance[b]
                        best_params[b] = params[b]
        closest_crops.append(best_params)
    closest_crops = torch.cat(closest_crops)

    torch.save(closest_crops, args.closest_crops)
