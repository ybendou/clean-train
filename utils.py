from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
import numpy as np

# function to display timer
def format_time(duration):
    duration = int(duration)
    s = duration % 60
    m = (duration // 60) % 60
    h = (duration // 3600)
    return "{:d}h{:02d}m{:02d}s".format(h,m,s)

def stats(scores):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    return np.mean(scores), float(low), float(up)

class ncm_output(nn.Module):
    def __init__(self, indim, outdim):
        super(ncm_output, self).__init__()
        self.linear = nn.Linear(indim, outdim, bias = False)
        with torch.no_grad():
            self.linear.weight.data = self.linear.weight.data / torch.norm(self.linear.weight.data, dim = 1, p = 2, keepdim = True) * torch.mean(torch.norm(self.linear.weight.data, dim = 1, p = 2))
        self.linear = nn.utils.weight_norm(self.linear)
        self.temp = nn.Parameter(torch.zeros(1) - 1)

    def forward(self, x):
        x = x / torch.norm(x + 1e-6, dim = 1, p = 2, keepdim = True)
        return self.temp * torch.norm(x.reshape(x.shape[0], 1, -1) - self.linear.weight_v.transpose(0,1).reshape(1, -1, x.shape[1]), dim = 2)

def linear(indim, outdim):
    if args.ncm_loss:
        return ncm_output(indim, outdim)
    else:
        return nn.Linear(indim, outdim)

def criterion_episodic(features, n_shots = 1):
    feat = features.reshape(args.n_ways, -1, features.shape[1])
    means = torch.mean(feat[:,:n_shots], dim = 1)
    dists = torch.norm(feat[:,n_shots:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3, p = 2).reshape(-1, args.n_ways)
    target = torch.arange(args.n_ways).repeat_interleave((features.shape[0] // args.n_ways) - n_shots).to(args.device)
    return torch.nn.CrossEntropyLoss()(-args.temperature * dists, target)
    
