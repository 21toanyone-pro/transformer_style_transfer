# Import PyTorch
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.modules.activation import MultiheadAttention
from einops import repeat
from einops.layers.torch import Rearrange

# Import custom modules
from layer import TransformerDecoderLayer, TransformerEncoderLayer, Decoder
from embedding import PatchEmbedding

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content_images, style_images, stylized_images):
        style_feats = self.encode_with_intermediate(style_images)
        content_feat = self.encode(content_images)
        stylized_feats = self.encode_with_intermediate(stylized_images)

        loss_c = self.calc_content_loss(stylized_feats[-1], content_feat)
        loss_s = self.calc_style_loss(stylized_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i])
        return loss_c, loss_s


class ST_Trainsformer(nn.Module):
    def __init__(self, d_model: int = 256, d_embedding: int = 256, 
                 n_head: int = 8, dim_feedforward: int = 1024,
                 img_size: int = 128, patch_size: int = 8, num_encoder_layer: int = 6, num_decoder_layer: int = 6,
                 dropout: float = 0.1):
        super(ST_Trainsformer, self).__init__()

        # Hyper-parameter setting
        self.d_model = 64
        self.patch_size = patch_size
        #self.input_linear = nn.Linear(256, (self.bottom_width**2) * self.d_model)

        # Image embedding part(content)
        self.c_patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, img_size=img_size)

        # Image embedding part(style)
        self.s_patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, img_size=img_size)

        # Transformer Encoder part # d_model = 512, dim_feed =512
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Transformer Decoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        decoder_mask_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(d_model, self_attn, decoder_mask_attn,
                dim_feedforward, dropout=dropout) for i in range(num_decoder_layer)])

        #self.reshape = nn.View(1, 64, 512, 512)
        #self.deconv = nn.Conv2d(1024//64, 3,1,1,0)
        # Target linear part
        self.trg_dropout = nn.Dropout(dropout)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, 1)



        self.CNN_Decoder = Decoder()

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p) 

    def forward(self, content_img: Tensor, style_img: Tensor, tgt_mask: Tensor) -> Tensor:

        # Image embedding
        S_encoder_out = self.s_patch_embedding(style_img).transpose(0, 1)

        C_encoder_out = self.s_patch_embedding(content_img).transpose(0, 1)

        # Image embedding

        # Text embedding
        #decoder_out = self.c_patch_embedding(content_img).transpose(0, 1)

        #패치단위로 쪼개기

        #선형 투영(Linear Projection)

        # Transformer Encoder
        for encoder in self.encoders:
            C_encoder_out = encoder(S_encoder_out)

        # Transformer Encoder
        for encoder in self.encoders:
            S_encoder_out = encoder(C_encoder_out)

        # Transformer Decoder
        for decoder in self.decoders:
            decoder_out = decoder(C_encoder_out, S_encoder_out)#, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        
        #decoder_out = self.CNN_Decoder(decoder_out)
        # Target linear
        #[n,C,W,H]
        # decoder_out = decoder_out.transpose(0, 1).contiguous()
        # decoder_out = decoder_out.unsqueeze(0)
        decoder_out = decoder_out.view(decoder_out.size(1),-1, self.d_model)
        decoder_out = decoder_out.contiguous()
        decoder_out = decoder_out.transpose(0,1)
        #decoder_out = torch.nn.functional.fold(decoder_out.transpose(1,2).contiguous(),int(self.patch_size),self.d_model,stride=self.d_model)
        decoder_out = self.CNN_Decoder(decoder_out)
        #decoder_out = self.trg_output_norm(self.trg_dropout(F.gelu(self.trg_output_linear(decoder_out))))
       # decoder_out = self.trg_output_linear2(decoder_out)

        #decoder_out = self.CNN_Decoder(decoder_out)

        return decoder_out

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask