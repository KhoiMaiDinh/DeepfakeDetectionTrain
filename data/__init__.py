import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import os
from .datasets import dataset_folder

from models.cnnDetection.validate import CNNmethod
from models.selfblended.validate import SelfBlendedMethod
from models.universalFake.validate import UniversalFakeMethod

def get_dataset(parent_folder = "./datasets/train/progan" + '/'):
    dset_lst = []
    classes = os.listdir(parent_folder)
    print("get_dataset()")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnnMethod = CNNmethod(device)
    selfBlendedMethod = SelfBlendedMethod(device)
    universalMethod = UniversalFakeMethod(device)
    for cls in classes:
        root = parent_folder + cls
        dset = dataset_folder(
            root, 
            cnnMethod.validate, 
            selfBlendedMethod.validate, 
            universalMethod.validate
        )
        print(dset[0])
        print(dset.classes)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(dataset):
    # shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    shuffle = False
    # dataset = get_dataset()
    # sampler = get_bal_sampler(dataset) if opt.class_bal else None
    sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(4)
    )
    return data_loader

