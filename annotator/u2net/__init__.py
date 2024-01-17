import os
from skimage import io, transform
import torch
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import glob

from .data_loader import RescaleT
from .data_loader import ToTensor
from .data_loader import ToTensorLab
from .data_loader import SalObjDataset

from .models import U2NET # full size version 173.6 MB

class U2Net(object):
    def __init__(self, ckpt_path=None, device=None,):
        self.ckpt_path = 'annotator/u2net/weights/u2net.pth' if ckpt_path==None else ckpt_path
        self.device = torch.device('cuda') if device==None else device
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device)

        self.u2net = U2NET(3,1).to(self.device)
        self.u2net.load_state_dict(self.ckpt)
        self.u2net.eval()

        self.transform = transforms.Compose([
            transforms.Resize(320, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])


    def __call__(self, input_image, threshold:float=0.5, resize:bool=True):
        '''
        input image should be a normalized torch tensor of shape [B, 3, 320, 320]
        set threshold to None for non binary values
        '''
        if input_image.shape[-1] != 320:
            input_image = F.interpolate(input_image, size=(320,320), mode='bilinear', recompute_scale_factor=False, align_corners=False)
        with torch.no_grad():
            mask,d2,d3,d4,d5,d6,d7= self.u2net(input_image)
        del d2,d3,d4,d5,d6,d7

        if isinstance(threshold, float):
            mask = torch.clamp(mask,0,1)
            ones = torch.ones_like(mask)
            zeros = torch.zeros_like(mask)
            mask = torch.where(mask>threshold, ones, zeros)
        
        if resize==True:
            mask = transforms.Resize(512, transforms.InterpolationMode.BILINEAR)(mask)
            
        return mask
    
if __name__ == "__main__":
    net = U2Net()
    image = Image.open('/workspace/Uni-ControlNet/81.jpg').convert('RGB')
    tr = transforms.Compose([
            transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor()
    ])
    image = tr(image).unsqueeze(0).to(torch.device('cuda'))
    x = net(image, threshold=0.5)

    topil = transforms.Compose([
            transforms.ToPILImage()
    ])
    topil(x[0]).save('u2.jpg')
    print()

    

