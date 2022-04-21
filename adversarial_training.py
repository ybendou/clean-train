#############################################################################################################
# Generate new crops around the bounding box of the closest crops to each summit of the simplex
#############################################################################################################

from torchvision import transforms, datasets
import torch
import numpy as np
from args import *
import resnet12
import pickle
from PIL import Image
import wandb
from fstools.cropping_utils import sample_new_crop
from fstools.utils import fastpickledump

def miniImageNet(use_hd = True):
    datasets = {}
    classes = []
    total = 60000
    count = 0
    for subset in ["train"]:
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
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomResizedCrop(84), norm)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    return train_clean

def iterator(data, target, transforms, forcecpu = False, shuffle = True, use_hd = False):
    """
    Get the iterator for the dataset
    Arguments:
        data: the data
        target : the target
        transforms: the transforms
        forcecpu: if True, use cpu
        shuffle: if True, shuffle the data
        use_hd: if True, use load images on hard disk
    Returns:
        the iterator
    """
    dataset = CPUDataset(data, target, transforms, use_hd = use_hd)
    return torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(8, os.cpu_count()))

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

def train(X, target, model, device='cuda:0', trainCfg={'epochs':100, 'lr':0.01, 'mmt':0.9}):
    """
        Train adversarial masks
    """
    batch_size = X.shape[0]
    X = X.to(args.device)
    target = target.to(args.device)
    M = nn.Parameter(torch.randn(batch_size, 1, X.shape[2], X.shape[3]).to(device))
    optimizer = torch.optim.SGD([M], lr=trainCfg['lr'], momentum=trainCfg['mmt'])
    best_epoch = {'epoch': 0, 'loss': 1e10, 'M': M.detach().cpu().numpy()}
    for epoch in range(trainCfg['epochs']):
        optimizer.zero_grad()
        # clip M between 0 and 1 
        M.data.clamp_(0, 1)
        output = model(X*M)
        loss = F.mse_loss(output, target)
        # if loss is smaller than the best loss, then save the model
        if loss.item() < best_epoch['loss']:
            best_epoch['epoch'] = epoch
            best_epoch['loss'] = loss.item()
            best_epoch['M'] = M.detach().cpu().numpy()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}/{trainCfg["epochs"]} Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()
    # if loss is smaller than the best loss, then save the model
    M.data.clamp_(0, 1)
    output = model(X*M)
    loss = F.mse_loss(output, target)
    if loss.item() < best_epoch['loss']:
        best_epoch['epoch'] = epoch
        best_epoch['loss'] = loss.item()
        best_epoch['M'] = M.detach().cpu().numpy()
    return best_epoch

if __name__=='__main__':
    # Initiate wandb project
    if args.wandb:
        wandb.init(project="adversarial_training", name="Adversarial Training", 
                    config=args, entity=args.wandb,
                    notes='Generate masks using adversarial training')

    # Get the loaders
    novel_loader = miniImageNet()
    # Get the model 
    model = resnet12.ResNet12(args.feature_maps, [3, 84, 84], 64, True, args.rotations).to(args.device)
    model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
    model.to(args.device)
    model.eval()

    # Load features of the base dataset
    print('Loading features of the base dataset')
    args.load_features = '/hdd/data/features/easyfeatures/minifeatures1.pt11'
    features = torch.load(args.load_features, map_location='cpu')[:64]
    centroids = features.mean(dim=1)

    for c in range(centroids.shape[0]):
        for batch_idx, (data, target) in enumerate(loader):        
            M = train(data, target, model)
            

    
    print('The end')