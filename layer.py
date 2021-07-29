from torch import nn
from torch import Tensor
import torch
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast
from einops import repeat
from einops.layers.torch import Rearrange
import math



def calculate_attention(self, q, k ,v, mask):
    d_k = k.size(-1)
    attention_score = torch.matmul(q, k.transpose(-2,-1))
    attention_score = attention_score/math.sqrt(d_k)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    attention_prob = F.softmax(attention_score, dim=1)
    out = torch.matmul(attention_prob, v)

    return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, dim_feedforward: int = 2048, dropout: float = 0.1):
        '''
        TransformerEncoderLayer is made up of self-attn and feedforward network.
        This standard encoder layer is based on the paper "Attention Is All You Need".
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
        Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
        Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
        in a different way during application. (From PyTorch Documents)
        '''
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @autocast()
    def forward(self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None) -> Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, mask_attn, dim_feedforward=2048, dropout=0.1):
        '''
        TransformerDecoderLayer is made up of self-attn and feedforward network.
        This standard encoder layer is based on the paper "Attention Is All You Need".
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
        Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
        Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
        in a different way during application. (From PyTorch Documents)
        '''
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.multihead_attn = mask_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @autocast()
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class Decoder(nn.Module):
    def __init__(self, patch_size=8,in_channels=3,d_model=512):
        super(Decoder, self).__init__()
        #self.layer1 = nn.Sequential(nn.ConvTranspose2d)
        #num_output =1
        #self.unfatten = nn.Unflatten(1, torch.Size([32,32,1]))
        #self.reshape =#
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.decoder_conv_1 = nn.ConvTranspose2d(in_channels=512,out_channels=128, kernel_size=3, padding=1)
        self.decoder_conv_2 = nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=3, padding=1)
        self.decoder_conv_3 = nn.ConvTranspose2d(in_channels=64,out_channels=3, kernel_size=3, padding=1)

    def forward(self, z):
        z = self.upsample(z)
        z = F.relu(self.decoder_conv_1(z))
        z = self.upsample(z)
        z = F.relu(self.decoder_conv_2(z))
        z = self.upsample(z)
        z = F.relu(self.decoder_conv_3(z))

        return z