import torch
import numpy as np
from args import *
from utils import *

n_runs = args.n_runs
batch_few_shot_runs = 100
assert(n_runs % batch_few_shot_runs == 0)
n_ways = args.n_ways
n_queries = args.n_queries

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

def ncm(features, run_classes, run_indices, n_shots):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(features)
        score = 0
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            score += (winners == targets).float().mean().item()
        return score / (batch_idx + 1)

def get_features(model, loader):
    model.eval()
    all_features, offset, max_offset = [], 100000, 0
    for batch_idx, (data, target) in enumerate(loader):        
        with torch.no_grad():
            data, target = data.to(args.device), target.to(args.device)
            _, features = model(data)
            all_features.append(features)
            offset = min(min(target), offset)
            max_offset = max(max(target), max_offset)
    num_classes = max_offset - offset + 1
    return torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])

def eval_few_shot(val_features, test_features, val_run_classes, val_run_indices, test_run_classes, test_run_indices, n_shots):
    return ncm(val_features, val_run_classes, val_run_indices, n_shots), ncm(test_features, test_run_classes, test_run_indices, n_shots)

def update_few_shot_meta_data(model, test_loader, val_loader, few_shot_meta_data):
    val_features = get_features(model, val_loader)
    test_features = get_features(model, test_loader)
    val_acc_5, test_acc_5 = eval_few_shot(val_features, test_features, few_shot_meta_data["val_run_classes_5"], few_shot_meta_data["val_run_indices_5"], few_shot_meta_data["novel_run_classes_5"], few_shot_meta_data["novel_run_indices_5"], n_shots = 5)
    if val_acc_5 > few_shot_meta_data["best_val_acc_5"]:
        if val_acc_5 > few_shot_meta_data["best_val_acc_5_ever"]:
            few_shot_meta_data["best_val_acc_5_ever"] = val_acc_5
            if args.save_model != "":
                if len(args.devices) == 1:
                    torch.save(model.state_dict(), args.save_model + "5")
                else:
                    torch.save(model.module.state_dict(), args.save_model + "5")
            if args.save_features != "":
                torch.save(test_features, args.save_features + "5")
        few_shot_meta_data["best_val_acc_5"] = val_acc_5
        few_shot_meta_data["best_test_acc_5"] = test_acc_5
    val_acc_1, test_acc_1 = eval_few_shot(val_features, test_features, few_shot_meta_data["val_run_classes_1"], few_shot_meta_data["val_run_indices_1"], few_shot_meta_data["novel_run_classes_1"], few_shot_meta_data["novel_run_indices_1"], n_shots = 1)
    if val_acc_1 > few_shot_meta_data["best_val_acc_1"]:
        if val_acc_1 > few_shot_meta_data["best_val_acc_1_ever"]:
            few_shot_meta_data["best_val_acc_1_ever"] = val_acc_1
            if args.save_model != "":
                if len(args.devices) == 1:
                    torch.save(model.state_dict(), args.save_model + "1")
                else:
                    torch.save(model.module.state_dict(), args.save_model + "1")
            if args.save_features != "":
                torch.save(test_features, args.save_features + "1")
        few_shot_meta_data["best_val_acc_1"] = val_acc_1
        few_shot_meta_data["best_test_acc_1"] = test_acc_1
    return val_acc_1, test_acc_1, val_acc_5, test_acc_5
            
