import os 
import numpy as np
import torch
device = 'cpu'
path = '/ssd2/data/AugmentedSamples/features/miniImagenet/changing_input_size/standardTrainingFeatures/'
files = os.listdir(os.path.join(path, 'adversarial'))
sorted_idx = np.argsort([f.split('_')[-3] for f in files]).tolist()
ordered_files = [os.path.join(path, 'adversarial', files[f]) for f in sorted_idx]
closest_crops = torch.cat([torch.load(f, map_location=device) for f in ordered_files])
save_path = os.path.join(path, f'{ordered_files[0].split(".pt")[0]}.pt')
print(f'Saving at: {save_path}')
torch.save(closest_crops, save_path)