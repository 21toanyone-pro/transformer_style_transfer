import argparse 
import os
import torch
from utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--dataroot', type=str, default='datasets/data/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--img_size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--init_type", type=str, default='normal', help="normal | orth | xavier_uniform")
parser.add_argument("--gan_mode", type=str, default='lsgan', help="lsgan | projection | ")
parser.add_argument("--netD", type=str, default='Projetion', help="basic | projection | ")
parser.add_argument("--netG", type=str, default='Projetion', help="Resnet | projection | ")
parser.add_argument('--beta1', type=float, default=0.99, help='momentum term of adam')
parser.add_argument('--loss_type', type=str, default='hinge', help='loss function name. hinge (default) or dcgan.')
parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',help='Apply relativistic loss or not. default: False')
parser.add_argument('--gen_dim_z', '-gdz', type=int, default=256,help='Dimension of generator input noise. default: 128')
parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',help='Input noise distribution: normal (default) or uniform.')
parser.add_argument('--triple_patch', default=False, type=str2bool,help='Triple patch testing; Default is False')
parser.add_argument('--patch_size', default=8, type=int, help='ViT patch size; Default is 9')
parser.add_argument('--d_model', default=512, type=int, help='Transformer model dimension; Default is 512')
parser.add_argument('--d_embedding', default=512, type=int, 
                        help='Transformer embedding word token dimension; Default is 256')
parser.add_argument('--n_head', default=8, type=int, 
                        help="Multihead Attention's head count; Default is 16")
parser.add_argument('--dim_feedforward', default=2048, type=int, 
                        help="Feedforward network's dimension; Default is 512")                        
parser.add_argument('--dropout', default=0., type=float, 
                        help="Dropout ration; Default is 0.1")
parser.add_argument('--num_encoder_layer', default=6, type=int, 
                        help="Number of encoder layers; Default is 12")
parser.add_argument('--num_decoder_layer', default=6, type=int, 
                        help="Number of decoder layers; Default is 12")             
parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token index; Default is 0')      
parser.add_argument('--max_len', default=300, type=int,
                        help='Maximum length of caption; Default is 300')
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False