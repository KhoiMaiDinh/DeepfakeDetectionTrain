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
import random
import os
from models.cnnDetection.validate import CNNmethod
from models.selfblended.validate import SelfBlendedMethod
from models.universalFake.validate import UniversalFakeMethod


class SearchImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(SearchImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes()
        self.classes = classes
        self.class_to_idx = class_to_idx
        # self.cnn_val = cnn_val
        # self.self_blended_val = self_blended_val
        # self.uni_val = uni_val
        
    def _find_classes(self):
        classes = ['CNN', 'Self Blended', 'Universal Fake']
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # print(classes, class_to_idx)
        return classes, class_to_idx

    def __getitem__(self, index):
        original_img, target = super(SearchImageFolder, self).__getitem__(index)
        path, _ = self.samples[index]
        print(original_img)
        cnn_sc=CNNmethod.validate(path)
        sb_sc=SelfBlendedMethod.validate(path)
        uf_sc=UniversalFakeMethod.validate(path)
        new_target=0
        if target == 1: 
            max_sc=max(cnn_sc, sb_sc, uf_sc)
            if max_sc == cnn_sc: new_target=0
            if max_sc == sb_sc: new_target=1
            if max_sc == uf_sc: new_target=2
        else: 
            min_sc=min(cnn_sc, sb_sc, uf_sc)
            if min_sc == cnn_sc: new_target=0
            if min_sc == sb_sc: new_target=1
            if min_sc == uf_sc: new_target=2
        return original_img, new_target
    
def dataset_folder(root):
    dset = SearchImageFolder(
            root,
            transforms.Compose([
                transforms.ToTensor(),
            ])
            )
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
