import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from .model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from .preprocess import extract_face
import warnings
import cv2
warnings.filterwarnings('ignore')

class SelfBlendedMethod:
    model = None
    device = None
    
    @classmethod
    def load_model(cls, device):
        seed=1
        random.seed(seed)   
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model=Detector()
        model=model.to(device)
        cnn_sd=torch.load("weights/FFraw.tar", map_location='cpu')["model"]
        model.load_state_dict(cnn_sd)
        model.eval()
        
        cls.model=model
        cls.device=device

    @classmethod
    def validate(cls, img):
        if cls.model is None:
            raise ValueError("Model not loaded. Call `load_model` first.")
        
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_detector = get_model("resnet50_2020-07-20", max_size=max(frame.shape),device=cls.device)
        face_detector.eval()

        face_list=extract_face(frame,face_detector)

        with torch.no_grad():
            if len(face_list) > 0:
                img=torch.tensor(face_list).to(cls.device).float()/255
            else:
                image_size=(380,380)
                frame = cv2.resize( frame,dsize=image_size).transpose((2,0,1))
                frames =[frame]
                img=torch.tensor(frames).to(cls.device).float()/255
            pred=cls.model(img).softmax(1)[:,1].cpu().data.numpy().tolist()

        print('Self blended fakeness: {:.2f}%'.format(max(pred) * 100))
        return max(pred)



