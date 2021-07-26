import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class STGAN():
    def set_input(input, device):
        real_A = input['A' ].to(device)
        real_B = input['B'].to(device)
        return real_A, real_B

    def GANloss(gan_mode):
        if gan_mode == 'lsgan':
            loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'projection':
            loss = nn.MSELoss()
        return loss

