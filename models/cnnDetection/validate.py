import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from .networks.resnet import resnet50

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--file', default='examples_realfakedir')
parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')

class CNNmethod:
    def __init__(self, device):
        model = resnet50(num_classes=1)
        state_dict = torch.load("weights/blur_jpg_prob0.5.pth", map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model.eval()
        model.to(device)
        
        self.model = model
        self.device = device

    def validate(self, img):

        trans_init = []
        print('Not cropping')
        trans = transforms.Compose(trans_init + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = trans(Image.open(img).convert('RGB'))

        with torch.no_grad():
            in_tens = img.unsqueeze(0)
            in_tens = in_tens.to(self.device)
            prob = self.model(in_tens).sigmoid().item()

        print('probability of being synthetic: {:.2f}%'.format(prob * 100))
        return prob