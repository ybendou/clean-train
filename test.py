from torchvision import transforms, datasets
from args import args
import numpy as np
import torch
import pickle
from PIL import Image


def miniImageNet():
    datasets = {}
    classes = []
    total = 60000
    count = 0
    resize = transforms.Resize(84)
    print("loading all images in memory, that might take a while")
    for subset in ["train", "validation", "test"]:
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
                    image = resize(Image.open(args.dataset_path + "miniimagenetimages/" + "images/" + fn).convert('RGB'))
                    data.append(image)
                    if count % 1000 == 0:
                        print("\r{:d}/{:d}".format(count, total), end = '')
        dicti = {"data":data, "labels":torch.LongTensor(target)}
        with open("/users/local/miniimagenet/" + subset + ".pkl", 'wb') as handle:
            pickle.dump(dicti, handle)
    print()
    
miniImageNet()
