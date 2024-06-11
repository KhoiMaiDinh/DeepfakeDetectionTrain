import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from .models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
import random
import shutil
from scipy.ndimage.filters import gaussian_filter

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)



def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    

class UniversalFakeMethod:
    def __init__(self, device, arch="CLIP:ViT-L/14"):
        model = get_model(arch)
        state_dict = torch.load("weights/fc_weights.pth", map_location='cpu')
        model.fc.load_state_dict(state_dict)
        print ("Model loaded..")
        model.eval()
        model.to(device)
        
        self.model = model
        self.device = device
        self.arch=arch

    def validate(self, img, find_thres=False):
        
        

        img=transformImg(img, self.arch)
        with torch.no_grad():
            y_true, y_pred = [], []
            in_tens = img.unsqueeze(0)
            in_tens = in_tens.to(self.device)
            prob = self.model(in_tens).sigmoid().item()

            print(prob)
        return prob

def transformImg(img, arch="CLIP:ViT-L/14"):
    img = Image.open(img).convert("RGB")
    stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
    transform=transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
    img = transform(img)
    return img


# class RealFakeDataset(Dataset):
#     def __init__(self,  real_path, 
#                         fake_path, 
#                         data_mode, 
#                         max_sample,
#                         arch,
#                         jpeg_quality=None,
#                         gaussian_sigma=None):

#         assert data_mode in ["wang2020", "ours"]
#         self.jpeg_quality = jpeg_quality
#         self.gaussian_sigma = gaussian_sigma
        
#         # = = = = = = data path = = = = = = = = = # 
#         if type(real_path) == str and type(fake_path) == str:
#             real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
#         else:
#             real_list = []
#             fake_list = []
#             for real_p, fake_p in zip(real_path, fake_path):
#                 real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
#                 real_list += real_l
#                 fake_list += fake_l

#         self.total_list = real_list + fake_list


#         # = = = = = =  label = = = = = = = = = # 

#         self.labels_dict = {}
#         for i in real_list:
#             self.labels_dict[i] = 0
#         for i in fake_list:
#             self.labels_dict[i] = 1

#         stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
#         self.transform = transforms.Compose([
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
#         ])


#     def read_path(self, real_path, fake_path, data_mode, max_sample):

#         if data_mode == 'wang2020':
#             real_list = get_list(real_path, must_contain='0_real')
#             fake_list = get_list(fake_path, must_contain='1_fake')
#         else:
#             real_list = get_list(real_path)
#             fake_list = get_list(fake_path)


#         if max_sample is not None:
#             if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
#                 max_sample = 100
#                 print("not enough images, max_sample falling to 100")
#             random.shuffle(real_list)
#             random.shuffle(fake_list)
#             real_list = real_list[0:max_sample]
#             fake_list = fake_list[0:max_sample]

#         assert len(real_list) == len(fake_list)  

#         return real_list, fake_list



#     def __len__(self):
#         return len(self.total_list)

#     def __getitem__(self, idx):
        
#         img_path = self.total_list[idx]

#         label = self.labels_dict[img_path]
#         img = Image.open(img_path).convert("RGB")

#         if self.gaussian_sigma is not None:
#             img = gaussian_blur(img, self.gaussian_sigma) 
#         if self.jpeg_quality is not None:
#             img = png2jpg(img, self.jpeg_quality)

#         img = self.transform(img)
#         return img, label





if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")


    opt = parser.parse_args()

    

    # for dataset_path in (dataset_paths):
    #     set_seed()


    #     ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)