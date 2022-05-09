import os
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
from fstools.utils import fix_seed, load_features, stats
from fstools.args import process_arguments


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

def sample_crop(elt, params, device='cpu'):
    cw, ch, dw, dh, size = params
    h, w = elt.shape[-2:]
    lineh = torch.linspace(-1, 1, h).to(device)
    linew = torch.linspace(-1, 1, w).to(device)
    meshx, meshy = torch.meshgrid((lineh, linew))
    meshx = (meshx) * dh +ch
    meshy = (meshy) * dw +cw
    grid = torch.stack((meshy, meshx), 2)
    grid = grid.unsqueeze(0)
    warped = F.grid_sample(elt, grid, mode='bilinear', align_corners=False)
    resizer = nn.FractionalMaxPool2d(3, output_size=(size.int(), size.int()))    
    return resizer(warped)
# Convert params to crops parameters
def convert_grid_params_to_crops_params(params, shape):
    maxh, maxw = shape
    cw, ch, dw, dh, size = params
    cw_coords = int((cw+1)*maxw/2)
    ch_coords = int((ch+1)*maxh/2)
    h = int((ch-dh+1)*maxh/2)
    w = int((cw-dw+1)*maxw/2)
    dh_coords = int((ch+dh+1)*maxh/2)
    dw_coords = int((cw+dw+1)*maxw/2)
    return torch.Tensor([h, w, dh_coords, dw_coords, maxh, maxw, size]).unsqueeze(0)
def clamp(params):
    if params.data[2] + abs(params.data[0])>1:
        params.data[2] = 1 - abs(params.data[0])
    if params.data[3] + abs(params.data[1])>1:
        params.data[3] = 1 - abs(params.data[1])
def train(X, centroids, model, device='cuda:0', trainCfg={'epochs':100, 'lr':0.01, 'mmt':0.9, 'loss_amp':1}, limit_borders=False, verbose=False):
    """
        Train adversarial masks
    """
    X = X.to(device)
    centroids = centroids.to(device)
    cw, ch, dw, dh, size = 0., 0., 1, 1, 110.
    params = nn.Parameter(torch.tensor([cw, ch, dw, dh, size]).to(device))
    if verbose:
        print('Init Params from center of image:', params)
    optimizer = torch.optim.Adam([params], lr=trainCfg['lr'])
    best_epoch = {'epoch': 0, 'loss': 1e10, 'params': params.detach().cpu().numpy(), 'crop':None}
    L2 = nn.MSELoss()
    for epoch in range(trainCfg['epochs']):
        optimizer.zero_grad()
        # clip M between 0 and 1
        if limit_borders:
            with torch.no_grad():
                clamp(params)
        crop = sample_crop(X, params, device=device)
        _, output = model(crop)
        loss = L2(output, centroids)*trainCfg['loss_amp']
        # if loss is smaller than the best loss, then save the model
        if loss.item() < best_epoch['loss']:
            best_epoch['epoch'] = epoch
            best_epoch['loss'] = loss.item()
            best_epoch['params'] = params.detach().cpu()
            best_epoch['crop'] = crop.detach().cpu()
        if epoch % 2 == 0 and verbose:
            print(f'Epoch: {epoch}/{trainCfg["epochs"]} Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()
    # if loss is smaller than the best loss, then save the model
    crop = sample_crop(X, params, device=device)
    _, output = model(crop)
    if limit_borders:
        with torch.no_grad():
            clamp(params)
    loss = L2(output, centroids)*trainCfg['loss_amp']
    if loss.item() < best_epoch['loss']:
        best_epoch['epoch'] = epoch
        best_epoch['loss'] = loss.item()
        best_epoch['params'] = params.detach().cpu()
        best_epoch['crop'] = crop.detach().cpu()
    return best_epoch
def freeze(model):
    for p in model.parameters():
        p.requires_grad = False

if __name__ == '__main__':

    from args import args

    fix_seed(args.seed)
    print('seed:', args.seed)
    if args.wandb:
        import wandb
        tag = (args.dataset != '')*[args.dataset] + (args.dataset == '')*['cross-domain']
        wandb.init(project=args.wandbProjectName, 
            entity=args.wandb, 
            tags=tag, 
            config=vars(args)
            )
    datasets = miniImageNet_standardTraining()
    features = torch.load(args.save_features, map_location='cpu')[:, :500]
    centroids = features.mean(dim=1)
    print('data loaded')
    # Get the model 
    model = ResNet12(args.feature_maps, [3, 84, 84], 100, True, args.rotations).to(args.device)
    model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
    model.to(args.device)
    model.eval()
    freeze(model)
    print('model loaded')
    
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))

    print('Start crop generation')


    if args.end_generation_idx == -1:
        args.end_generation_idx = len(datasets)

    closest_crops = []
    for i in tqdm(range(args.start_generation_idx, args.end_generation_idx)): 
        img_path, classe = datasets[0][i], datasets[1][i] 
        img = norm(transforms.ToTensor()(np.array(Image.open(img_path).convert('RGB')))).unsqueeze(0)
        class_centroid = centroids[classe].to(args.device)
        M = train(img, class_centroid.unsqueeze(0), model, device=args.device, trainCfg={'epochs':1000, 'lr':0.01, 'mmt':0.8, 'loss_amp':1}, limit_borders=True)
        closest_crops.append(M['params'])
    closest_crops = torch.stack(closest_crops)
    print('Close crop generation Done:', closest_crops.shape)
    if args.end_generation_idx == len(datasets):
        torch.save(closest_crops, args.closest_crops)
    else:
        torch.save(closest_crops, args.closest_crops+args.end_generation_idx)