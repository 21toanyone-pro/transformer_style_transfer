from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from transformers import AdamW
from optimizer import Ralamb
from scheduler import WarmupLinearSchedule
import argparse
import torch.nn as nn

def optimizer_select(model, args):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.w_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                              args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=args.lr, eps=1e-8)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    elif args.optimizer == 'Ralamb':
        optimizer = Ralamb(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr)
    else:
        raise Exception("Choose optimizer in ['AdamW', 'Adam', 'SGD', 'Ralamb']")
    return optimizer

def shceduler_select(optimizer, dataloader_dict, args):
    if args.scheduler == 'constant':
        scheduler = StepLR(optimizer, step_size=len(dataloader_dict['train']), gamma=1)
    elif args.scheduler == 'warmup':
        scheduler = WarmupLinearSchedule(optimizer, 
                                        warmup_steps=int(len(dataloader_dict['train'])*args.n_warmup_epochs), 
                                        t_total=len(dataloader_dict['train'])*args.num_epochs)
    elif args.scheduler == 'reduce_train':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(len(dataloader_dict['train'])*1.5),
                                      factor=0.5)
    elif args.scheduler == 'reduce_valid':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    elif args.scheduler == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_lambda ** epoch)
    else:
        raise Exception("Choose shceduler in ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']")
    return scheduler

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)