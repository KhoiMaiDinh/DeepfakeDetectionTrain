import cv2
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from models.cnnDetection.validate import CNNmethod
from models.selfblended.validate import SelfBlendedMethod
from models.universalFake.validate import UniversalFakeMethod


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def calculate_target(path, original_target):
    cnn_sc=CNNmethod.validate(path)
    sb_sc=SelfBlendedMethod.validate(path)
    uf_sc=UniversalFakeMethod.validate(path)
    new_target=0
    if original_target == 1: 
        max_sc=max(cnn_sc, sb_sc, uf_sc)
        if max_sc == cnn_sc: new_target=0
        if max_sc == sb_sc: new_target=1
        if max_sc == uf_sc: new_target=2
    else: 
        min_sc=min(cnn_sc, sb_sc, uf_sc)
        if min_sc == cnn_sc: new_target=0
        if min_sc == sb_sc: new_target=1
        if min_sc == uf_sc: new_target=2
    return new_target
    
def dataset_folder(root):
    original_dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                transforms.ToTensor(),
            ])
            )
    
    images = []
    targets = []
    
    for image, target in original_dset:
        images.append(image)
        image_paths = original_dset.imgs[len(images) - 1][0]
        targets.append(calculate_target(image_paths, target))
    
    dset = CustomDataset(images, targets)
    # dset = datasets.ImageFolder(
    #         root, 
    #         transforms.Compose([
    #             transforms.ToTensor(),
    #         ])
    #         )
    # transforms.Compose([
    #             rz_func,
    #             transforms.Lambda(lambda img: data_augment(img, opt)),
    #             crop_func,
    #             flip_func,
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ])
    return dset

# def binary_dataset(root):
