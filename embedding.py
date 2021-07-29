# Import modules
import math
from einops import repeat
from einops.layers.torch import Rearrange
# Import PyTorch
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast

class PatchEmbedding(nn.Module): # content embedding
    """
    Embedding which is consisted with under features
    1. projection : using conv layer to flatten and rearrange
    2. positions : adding positional information using parameters
    sum of all these features are output of Embedding
    then use Factorized embedding parameterization from ALBERT (Z Lan et al. 2019)
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 8, d_model: int = 512,
                 img_size: int = 128):
        super().__init__()
        # Hyper-parameter setting
        self.patch_size = patch_size

        self.projection= nn.Sequential(Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2 = patch_size),
                    nn.Linear(patch_size * patch_size * in_channels, d_model))
        # Patch projection & parameter setting
        # 이미지를 패치사이즈로 flatten
        # self.projection = nn.Sequential(
        #     nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
        #     Rearrange('b e (h) (w) -> b (h w) e')
        # )

        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2, d_model))
        #self.positions = positionalEncoding2D(d_model, img_size)

    @autocast()
    def forward(self, x: Tensor) -> Tensor:
        #b,_, _,_= x.shape
        # prepare settings
        x = self.projection(x)
        x += self.positions
        x_out = x

        return x_out

class PatchEmbedding_style(nn.Module): # style embedding
    """
    Embedding which is consisted with under features
    1. projection : using conv layer to flatten and rearrange
    2. positions : adding positional information using parameters
    sum of all these features are output of Embedding
    then use Factorized embedding parameterization from ALBERT (Z Lan et al. 2019)
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 8, d_model: int = 512,
                 img_size: int = 128):
        super().__init__()
        # Hyper-parameter setting
        self.patch_size = patch_size

        self.projection= nn.Sequential(Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2 = patch_size),
                    nn.Linear(patch_size * patch_size * in_channels, d_model))
        # Patch projection & parameter setting
        # 이미지를 패치사이즈로 flatten
        # self.projection = nn.Sequential(
        #     nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
        #     Rearrange('b e (h) (w) -> b (h w) e')
        # )


    @autocast()
    def forward(self, x: Tensor) -> Tensor:
        #b,_, _,_= x.shape
        # prepare settings
        x = self.projection(x)
        #x += self.positions
        x_out = x

        return x_out
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = d_model
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x : Tensor) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = Rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = Rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

def positionalEncoding2D(d_model, img_size):
    if d_model % 4 != 0:
        raise ValueError("Cannot")
    
    pe = torch.zeros(d_model, img_size, img_size)

    d_model = int(d_model/2)

    div_term = torch.exp(torch.arange(0., d_model, 2)*-(math.log(10000.0)/ d_model))

    pos_w = torch.arange(0., img_size).unsqueeze(1)
    pos_h = torch.arange(0., img_size).unsqueeze(1)

    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, img_size, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, img_size, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, img_size)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, img_size)

    return pe

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    @autocast()
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TransformerEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEncoding : adding positional information using sin, cos
    sum of all these features are output of Embedding
    """
    def __init__(self, vocab_size, d_model, embed_size, pad_idx=0, max_len=512, embedding_dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.linear_layer = nn.Linear(embed_size, d_model)
        self.position = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.embed_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(embedding_dropout)

    @autocast()
    def forward(self, sequence):
        x = self.dropout(F.gelu(self.linear_layer(self.token(sequence))))
        x = self.embed_norm(x + self.position(sequence))
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

