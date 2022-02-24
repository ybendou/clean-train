import torch
import numpy as np
from args import *
from utils import *

n_runs = args.n_runs
batch_few_shot_runs = 100
assert(n_runs % batch_few_shot_runs == 0)

def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class):
    shuffle_classes = torch.LongTensor(np.arange(num_classes))
    run_classes = torch.LongTensor(n_runs, n_ways).to(args.device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(args.device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices

def generate_runs(data, run_classes, run_indices, batch_idx):
    n_runs, n_ways, n_samples = run_classes.shape[0], run_classes.shape[1], run_indices.shape[2]
    run_classes = run_classes[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_indices = run_indices[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_classes = run_classes.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2])
    run_indices = run_indices.unsqueeze(3).repeat(1, 1, 1, data.shape[2])
    datas = data.unsqueeze(0).repeat(batch_few_shot_runs, 1, 1, 1)
    cclasses = torch.gather(datas, 1, run_classes)
    res = torch.gather(cclasses, 2, run_indices)
    return res

def ncm(train_features, features, run_classes, run_indices, n_shots):
    with torch.no_grad():
        dim = features.shape[2] if args.n_augmentation == 0 else features['normal'].shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            if args.n_augmentation == 0:
                runs = generate_runs(features, run_classes, run_indices, batch_idx)
                means = torch.mean(runs[:,:,:n_shots], dim = 2)
                distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)
            else:
                all_runs = {k:generate_runs(feats, run_classes, run_indices, batch_idx)[:,:,:n_shots] for k,feats in features.items() if k!='normal'}
                all_runs['normal'] = generate_runs(features['normal'], run_classes, run_indices, batch_idx)
                support_feats = torch.cat([run[:,:,:n_shots] for run in all_runs.values()], axis=2)

                if args.augmentation_best_select and n_shots>1: # select best crops based on their distance to the 
                    # support_feats get centroid : 
                    centroid = torch.mean(all_runs['normal'][:,:,:n_shots], dim = 2).reshape(batch_few_shot_runs, args.n_ways, 1, dim)
                    centroid = centroid/torch.norm(centroid, dim = 2, keepdim=True) # re-project on the sphere

                    cos = torch.nn.CosineSimilarity(dim=3)

                    support_feats_augmented = torch.cat([run[:,:,:n_shots] for k,run in all_runs.items() if k!='normal'], axis=2).reshape(batch_few_shot_runs, args.n_ways, -1, dim) # support features, only the augmented ones (crops)
                    cosine_distances = cos(support_feats_augmented, centroid) # cosine distance from centroid
                    threshold = cosine_distances.mean()-2*cosine_distances.std() # threshold from centroid (1std from mean)
                    filtered_support_feats_aug = (support_feats_augmented*(cosine_distances>=threshold).reshape(batch_few_shot_runs, args.n_ways, args.n_augmentation*n_shots, 1).int()) # get the close crops to the centroid, the other ones have zero features
                    filtered_support_feats = torch.cat([all_runs['normal'][:,:,:n_shots], filtered_support_feats_aug], axis=2) # add the original images, maybe later include the normal ones in the threshold
                    means = (filtered_support_feats.sum(axis=2)/(n_shots+(cosine_distances>=threshold).reshape(batch_few_shot_runs, args.n_ways, args.n_augmentation*n_shots, 1).int().sum(axis=2))) # get the new mean by considering 
                    distances = torch.norm(all_runs['normal'][:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2) # compute the new distances

                else:
                    runs = torch.zeros(batch_few_shot_runs, args.n_ways, n_shots*(args.n_augmentation+1)+args.n_queries, dim).to(args.device)
                    runs[:,:,n_shots*(args.n_augmentation+1):]=all_runs['normal'][:,:,n_shots:]
                    runs[:,:,:n_shots*(args.n_augmentation+1)]=support_feats
                    means = torch.mean(runs[:,:,:n_shots*(args.n_augmentation+1)], dim = 2)
                    distances = torch.norm(runs[:,:,n_shots*(args.n_augmentation+1):].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)

            winners = torch.min(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        if args.augmentation_best_select and n_shots>1:
            print(f'cosine mean: {cosine_distances.mean()}, std: {cosine_distances.std()}, threshold:{threshold}')
        return stats(scores, "")

def transductive_ncm(train_features, features, run_classes, run_indices, n_shots, n_iter_trans = args.transductive_n_iter, n_iter_trans_sinkhorn = args.transductive_n_iter_sinkhorn, temp_trans = args.transductive_temperature, alpha_trans = args.transductive_alpha, cosine = args.transductive_cosine):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features)
        if cosine:
            features = features / torch.norm(features, dim = 2, keepdim = True)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            if cosine:
                means = means / torch.norm(means, dim = 2, keepdim = True)
            for _ in range(n_iter_trans):
                if cosine:
                    similarities = torch.einsum("bswd,bswd->bsw", runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, dim))
                    soft_sims = torch.softmax(temp_trans * similarities, dim = 2)
                else:
                    similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                    soft_sims = torch.exp( -1 * temp_trans * similarities)
                for _ in range(n_iter_trans_sinkhorn):
                    soft_sims = soft_sims / soft_sims.sum(dim = 2, keepdim = True) * args.n_ways
                    soft_sims = soft_sims / soft_sims.sum(dim = 1, keepdim = True) * args.n_queries
                new_means = ((runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", soft_sims, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])))) / runs.shape[2]
                new_means = new_means / torch.norm(new_means, dim = 2, keepdim = True)
                means = means * alpha_trans + (1 - alpha_trans) * new_means
                means = means / torch.norm(means, dim = 2, keepdim = True)
            if cosine:
                winners = torch.max(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            else:
                winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def ncm_cosine(train_features, features, run_classes, run_indices, n_shots):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features)
        features = sphering(features)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            means = sphering(means)
            distances = torch.einsum("bwysd,bwysd->bwys",runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim))
            winners = torch.max(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

import os

def get_features(model, loader, dataset='train'):
    """
        Get features depending if data augmentation is enabled
        If data aug enabled, the data loader for val and novel is randomly augmenting images
    """
   
    if args.n_augmentation>0 and dataset!='train' and dataset!='val':
        normal_loader, augmented_loader = loader
        augmented_features = {'normal':get_features_(model, normal_loader)}

        feats = []
        for n in range(args.n_augmentation):
            print(f'----- Augmentation number:{n}')
            feats.append(get_features_(model, augmented_loader, augmentation_num=n, dataset=dataset))

        features_stacked = torch.stack(feats, dim=2)
        augmented_features['augmented'] = features_stacked

        if args.save_augmented_features != "" and dataset=='novel':
            torch.save(augmented_features, args.save_augmented_features)
            
        return augmented_features
    else: 
        return get_features_(model, loader[0])
def get_features_(model, loader, augmentation_num=None, dataset='train'):
    model.eval()
    all_features, offset, max_offset = [], 100000, 0
    for batch_idx, (data, target) in enumerate(loader):        
        with torch.no_grad():
            data, target = data.to(args.device), target.to(args.device)
            if args.save_images != '' and dataset=='novel':
                if augmentation_num == None:
                    save_folder = os.path.join(f'{args.save_images}', 'normal')
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_path = os.path.join(save_folder, f'images_{batch_idx}')
                else:
                    save_folder = os.path.join(f'{args.save_images}', f'augment_{augmentation_num}')
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_path = os.path.join(save_folder, f'images_{batch_idx}')

                torch.save({'target':target, 'data':data}, save_path)

            _, features = model(data)
            all_features.append(features)
            offset = min(min(target), offset)
            max_offset = max(max(target), max_offset)
    num_classes = max_offset - offset + 1
    print(".", end='')
    return torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])

def eval_few_shot(train_features, val_features, novel_features, val_run_classes, val_run_indices, novel_run_classes, novel_run_indices, n_shots, transductive = False):
    if transductive:
        return transductive_ncm(train_features, val_features, val_run_classes, val_run_indices, n_shots), transductive_ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots)
    else:
        #return ncm(train_features, val_features, val_run_classes, val_run_indices, n_shots), ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots)
        return (0.99, 0.01), (0.99, 0.01) #ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots)


def update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data):

    if "M" in args.preprocessing or args.save_features != '':
        train_features = None #get_features(model, train_clean, dataset='train')
    else:
        train_features = torch.Tensor(0,0,0)
    val_features = None #get_features(model, val_loader, dataset='val')
    novel_features = get_features(model, novel_loader, dataset='novel')

    res = []
    for i in range(len(args.n_shots)):
        res.append(evaluate_shot(i, train_features, val_features, novel_features, few_shot_meta_data, model = model))

    return res

def evaluate_shot(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    (val_acc, val_conf), (novel_acc, novel_conf) = (0.1,0.1), (0.1,0.1)  #eval_few_shot(train_features, val_features, novel_features, few_shot_meta_data["val_run_classes"][index], few_shot_meta_data["val_run_indices"][index], few_shot_meta_data["novel_run_classes"][index], few_shot_meta_data["novel_run_indices"][index], args.n_shots[index], transductive = transductive)
    
    if val_acc > few_shot_meta_data["best_val_acc"][index]:
        if val_acc > few_shot_meta_data["best_val_acc_ever"][index]:
            few_shot_meta_data["best_val_acc_ever"][index] = val_acc
            if args.save_model != "":
                if len(args.devices) == 1:
                    torch.save(model.state_dict(), args.save_model + str(args.n_shots[index]))
                else:
                    torch.save(model.module.state_dict(), args.save_model + str(args.n_shots[index]))
            if args.save_features != "":
                torch.save(torch.cat([train_features, val_features, novel_features], dim = 0), args.save_features + str(args.n_shots[index]))
        few_shot_meta_data["best_val_acc"][index] = val_acc
        few_shot_meta_data["best_novel_acc"][index] = novel_acc
    return val_acc, val_conf, novel_acc, novel_conf

print("eval_few_shot, ", end='')
