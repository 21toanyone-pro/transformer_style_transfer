from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np 
from PIL import Image
import itertools
from tqdm import tqdm
import option
from dataset import ImageDataset
import os
import gc
import time
import logging
#from network import Generators, Discriminators, define_D, define_G
# Import PyTorch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.style.dataset import CustomDataset
from network import ST_Trainsformer, Net
from utils import weights_init
from utils import LambdaLR
from gan_model import STGAN
import torchvision.models as models
from torchvision.utils import save_image

if __name__ =='__main__':
    opt = option.opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms_ = [ transforms.Resize(int(opt.img_size), Image.BICUBIC), 
                transforms.RandomCrop(opt.img_size),       
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batch_size, shuffle=True, num_workers=0)

    vgg = models.vgg19(pretrained=True).features
    vgg_encoder = Net(vgg)
    vgg_encoder = vgg_encoder.to(device)
    
    model = ST_Trainsformer(d_model=opt.d_model, d_embedding=opt.d_embedding, 
                               n_head=opt.n_head, dim_feedforward=opt.dim_feedforward, img_size=opt.img_size, 
                               patch_size=opt.patch_size, 
                               num_encoder_layer=opt.num_encoder_layer, num_decoder_layer=opt.num_decoder_layer,
                               dropout=opt.dropout)
    model = model.train()
    model = model.to(device)
    model.apply(weights_init)

    # define loss
    MSE_loss = nn.MSELoss()

    # optimizer & LR schedulers
    gen_optimizer = torch.optim.Adam(model.parameters(), lr= 0.0005)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Input, output setting`
    for epoch in range(0, opt.n_epochs):
        model.train()
        tgt_mask = model.generate_square_subsequent_mask(opt.max_len - 1, device)

        for i, img in enumerate(dataloader):
            img_s, img_c = STGAN.set_input(img, device)
            save_image(img_s, f'./checkpoint/gen_data/{epoch}_cc.jpg', nrow=4, normalize=True, scale_each=True)
            save_image(img_c, f'./checkpoint/gen_data/{epoch}_ss.jpg', nrow=4, normalize=True, scale_each=True)
            gen_optimizer.zero_grad()
            gen_data = model(img_c, img_s)
            save_image(gen_data, f'./checkpoint/gen_data/{epoch}_fake.jpg', nrow=4, normalize=True, scale_each=True)
            ss_data = model(img_s, img_s)
            cc_data = model(img_c, img_c)

            L_id = F.mse_loss(cc_data, img_c)+F.mse_loss(ss_data, img_s)
            #loss_cc,loss_ss = vgg_encoder(img_c, img_s, gen_data)
            loss_c, loss_s, loss_cc = vgg_encoder(img_c, img_s, gen_data, ss_data, cc_data)
            
            loss_c = loss_c.mean()
            loss_s = loss_s.mean()
            loss_cc = loss_cc.mean()
            #loss_mse = F.mse_loss(img_c, gen_data)
            
            loss_style = 10*loss_c + 7*loss_s + 50*L_id + 1*loss_cc

            loss_style.backward()
            gen_optimizer.step()
            lr_scheduler_D_B.step()
            
        print("[Epoch %d/%d] [Batch %d/%d] [Style loss: %f]" %(epoch, opt.n_epochs, i % len(dataloader), len(dataloader), (loss_style)))