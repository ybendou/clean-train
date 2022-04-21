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
def miniImageNet(bounding_box_summits, use_hd = True, sample_outer_bb=True):
    """
        Get the features of the miniImageNet dataset
    """
    datasets = {}
    count = 0
    subset = "test"
    data = []
    bounding_boxes = []
    with open(args.dataset_path + "miniimagenetimages/" + subset + ".csv", "r") as f:
        start = 0
        for line in f:
            if start == 0:
                start += 1
            else:
                splits = line.split(",")
                fn, _ = splits[0], splits[1]
                path = args.dataset_path + "miniimagenetimages/" + "images/" + fn
                # Get bounding box summits
                summits = bounding_box_summits[count]
                # Get image for each summit
                for summit in summits:
                    if not use_hd:
                        image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                        data.append(image)
                    else:
                        data.append(path)
                    bounding_boxes.append(summit)
                count += 1
    datasets[subset] = [data, torch.stack(bounding_boxes)]
    print('stats:',len(data), datasets[subset][1].shape)
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))

    augmentations = []
    for augment in args.augmentations:
        if augment=='crop':
            augmentations.append(transforms.RandomResizedCrop(84))
            augmentations.append(transforms.Resize([84, 84]))
            print('Random Cropping')
            
    augmentations.append(norm)
    test_loader_aug = iterator(datasets["test"][0], datasets["test"][1], transforms = augmentations, forcecpu = True, shuffle = False, use_hd = use_hd, sample_outer_bb=sample_outer_bb)

    return test_loader_aug

def iterator(data, bouding_boxes, transforms, forcecpu = False, shuffle = True, use_hd = False, sample_outer_bb=True):
    """
    Get the iterator for the dataset
    Arguments:
        data: the data
        bouding_boxes: the bounding box summits
        transforms: the transforms
        forcecpu: if True, use cpu
        shuffle: if True, shuffle the data
        use_hd: if True, use load images on hard disk
        sample_outer_bb: if True, sample outside the bounding box
    Returns:
        the iterator
    """
    dataset = CPUDataset(data, bouding_boxes, transforms, use_hd = use_hd, sample_outer_bb=sample_outer_bb)
    return torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(8, os.cpu_count()))

class CPUDataset():
    """
    Dataset class
    """
    def __init__(self, data, bounding_boxes, transforms = [], batch_size = args.batch_size, use_hd = False, sample_outer_bb=True):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.bounding_boxes = bounding_boxes
        assert(self.length == bounding_boxes.shape[0])
        self.batch_size = batch_size
        self.transforms = transforms
        self.use_hd = use_hd
        self.sample_outer_bb = sample_outer_bb
    def __getitem__(self, idx):
        if self.use_hd:
            elt = transforms.ToTensor()(np.array(Image.open(self.data[idx]).convert('RGB')))
        else:
            elt = self.data[idx]
        
        # Get the cropping object
        crop = self.transforms[0]

        # Get the bounding box parameters of each summit
        bb_params = self.bounding_boxes[idx]

        # Sample a random bounding box around each summit, if sample_outer_bb is False, then the bounding box is a square around the summit
        new_sampled_bb_params = sample_new_crop(*bb_params.tolist(), scale=1.5, deterministic=self.sample_outer_bb)

        # Crop the original image with the new bounding box
        elt = transforms.functional.crop(elt, *new_sampled_bb_params[:4])
        h, w, dh, dw, maxh, maxw = new_sampled_bb_params

        # If the sampling is not only outside the crop, we need to crop the crop again
        if not self.sample_outer_bb:
            re_new_sampled_bb_params = crop.get_params(elt, scale=(0.14,1), ratio=(0.75, 1.333333)) # sample some parameter
            elt = transforms.functional.crop(elt, *re_new_sampled_bb_params)
            new_h, new_w, dh, dw = re_new_sampled_bb_params
            h = new_h + h
            w = new_w + w
            
        # Resize the image to 84x84
        elt = torch.nn.Sequential(*self.transforms[1:])(elt)
        
        return elt, torch.Tensor([h, w, dh, dw, maxh, maxw]).unsqueeze(0)

    def __len__(self):
        return self.length

def get_features(model, loader, dataset='novel'):
    """
        Get all features from the novel dataset with augmentations
    """
    feats = []
    crops_params = []

    for n in range(args.n_augmentation):
        print(f'----- Augmentation number:{n}')
        features, params = get_features_(model, loader, augmentation_num=n, dataset=dataset)
        feats.append(features)
        crops_params.append(params)

    augmented_features = torch.stack(feats, dim=1)
    crops_params = torch.stack(crops_params, dim=1)
    return augmented_features, crops_params
def get_features_(model, loader, augmentation_num=None, dataset='train'):
    """
        Get features of one augmentation from the novel dataset
    """
    model.eval()
    all_features, all_params = [], []
    for batch_idx, (data, params) in enumerate(loader):        
        with torch.no_grad():
            all_params.append(params)
            data = data.to(args.device)
            _, features = model(data)
            all_features.append(features)

    print(".", end='')
    return torch.cat(all_features, dim = 0).reshape(-1, all_features[0].shape[-1]), torch.cat(all_params, dim=0).reshape(-1, all_params[0].shape[-1])

def train(X, target, model):
    """
        Train adversarial masks
    """
    batch_size = X.shape[0]
    X = X.to(args.device)
    target = target.to(args.device)
    M = nn.Parameter(torch.randn(batch_size, X.shape[1], X.shape[2]).to(device))
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
    # Load bounding box parameters
    with open(args.bounding_box_file, 'rb') as pickle_file:
        bounding_box_summits = pickle.load(pickle_file)
   

    # Initiate wandb project
    if args.wandb:
        wandb.init(project="adversarial_training", name="Adversarial Training", 
                    config=args, entity=args.wandb,
                    notes='Generate masks using adversarial training')

    # Get the loaders
    novel_loader = miniImageNet(sample_outer_bb=False)

    # Get the model 
    model = resnet12.ResNet12(args.feature_maps, [3, 84, 84], 64, True, args.rotations).to(args.device)
    model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
    model.to(args.device)
    
    # Do a first forward pass to get the features of the base classes



    
    # Get features 
    features, features_params = get_features(model, novel_loader, dataset='novel')

    # Save the features
    if args.save_augmented_features != "":
        torch.save(features, args.save_augmented_features)
        torch.save(features_params, args.save_augmented_features+'params')

    # Compute mean over all crops
    AS_simplex_features = features.mean(dim=1)
    simplex_list, params_list = convert_array_simplex_features_to_list(AS_simplex_features, features_params, bounding_box_summits)  

    # Save the simplex features
    save_summits_list = '/ssd2/data/AugmentedSamples/features/miniImagenet/boundingboxSimplex/AS1000_0123_noPrep_Simplex0.05/simplex_listInnerbb'

    # print('Saving features')
    # fastpickledump(simplex_list, save_summits_list+'.pickle')
    
    # print('Saving params')
    # fastpickledump(params_list, save_summits_list+'params.pickle')
   
    print('The end')