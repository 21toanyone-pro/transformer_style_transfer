
import os
import numpy as np
import math
import option
# torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from PIL import Image

def make_dataset(dirs):
    filenames = os.listdir(dirs)
    images = []
    for filename in filenames:
        full_list = os.path.join(dirs, filename)
        images.append(full_list)
    return images

class CustomdataLoad(torch.utils.data.Dataset):
    def __init__(self, dirs, transforms=None):
        self.data_dirX = make_dataset(dirs)
        self.Xsize = len(self.data_dir)
        self.transforms = transforms
    def __len__(self):
        return self.size    
        
    def __getitem__(self, idx):
        path = self.data_dir[idx % self.size]
        img = Image.open(path)
        sample = {'image':image, 'path':path}
        if self.transforms:
            sample = self.transforms(sample)

        return sample
            





class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=1, padding=1, norm = 'bnorm'):
        super(Generator, self).__init__()
        layer =[]
        layer +=[nn.Linear(input_nc, output_nc)]
        layer +=[nn.ReLU()]

        layer += [nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding)]
        if norm == 'bnorm'
            layer += [nn.BatchNorm2d(num_features =output_nc)]
        elif norm == 'Inorm'
            layer += [nn.InstanceNorm2d(num_features=output_nc)]

        
if __name__ == '__main__':
    dirs = './data'    

