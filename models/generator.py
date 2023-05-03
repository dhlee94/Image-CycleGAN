import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from random  import random
import os
from torchsummary import summary as summary_

class DropBlock(nn.Module):
    def __init__(self, block_size: int, p: float = 0.5):
        super().__init__()
        self.block_size = block_size
        self.p = p
    def calculate_gamma(self, x: torch.Tensor) -> float:
        """Compute gamma, eq (1) in the paper
        Args:
            x (Tensor): Input tensor
        Returns:
            Tensor: gamma
        """      
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=self.block_size,
                stride=1,
                padding=self.block_size // 2
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x
    
class EfficientChannelAttention(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(EfficientChannelAttention, self).__init__()
        self.in_channel = in_channels
        # self.t = int(abs((log(self.in_channel, 2) + b) / gamma))
        # self.k = self.t if self.t % 2 else self.t + 1
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.Conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=((k_size - 1) // 2))
        self.S = nn.Sigmoid()
    def forward(self, x):
        y = self.GAP(x)
        y = self.Conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.S(y)
        return x.mul(y.expand_as(x))

class Attentionblock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(Attentionblock, self).__init__()
        self.block1 = nn.Sequential(
            LayerNorm(in_channels1, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(in_channels1, out_channels, kernel_size=3, padding=1, bias=False),      
            nn.MaxPool2d((2))
        )
        self.block2 = nn.Sequential(
            LayerNorm(in_channels2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(in_channels2, out_channels, kernel_size=3, padding=1, bias=False)
        )
        self.block3 = nn.Sequential(
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.Sigmoid(),            
        )
    def forward(self, x1, x2):
        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x = torch.add(x1, x2)
        x = self.block3(x)
        x = x.mul(x2)
        return x

class PSPPooling(nn.Module):
    def __init__(self, in_channel, out_channel, up_mode="bilinear", filter_size=[1, 2, 4, 8]):
        super(PSPPooling, self).__init__()
        self.PSPblock = nn.ModuleList()
        self.filter_size = filter_size
        self.filter_depth = len(filter_size)
        for i in self.filter_size:
            self.PSPblock.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=(i), stride=(i)),
                    nn.Upsample(scale_factor=(i), mode=up_mode, align_corners=True),
                    nn.Conv2d(in_channel, in_channel // self.filter_depth, kernel_size=1, stride=1, padding=0, bias=False),
                    LayerNorm(in_channel // self.filter_depth, eps=1e-6, data_format="channels_first")))
        self.out = nn.Sequential(
                    nn.Conv2d(2*in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                    LayerNorm(out_channel, eps=1e-6, data_format="channels_first"))

    def forward(self, x):
        total_x = [x]
        for i in range(self.filter_depth):
            sub_x = self.PSPblock[i](x)
            total_x.append(sub_x)
        total_x = torch.cat(total_x, dim=1)
        x = self.out(total_x)
        return x

class Stemblock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, drop_rate=0.25):
        super(Stemblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=strides, bias=False)
        self.ECA = EfficientChannelAttention(in_channels=out_channels)
    def forward(self, x):
        s = self.shortcut(x)
        x = torch.add(self.block(x), s)
        x = self.ECA(x)
        return x

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, block_size=7, p_value=0.5):
        super(Resblock, self).__init__()
        self.layer = nn.Sequential(
                LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                DropBlock(block_size=block_size, p=p_value),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
                LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.GELU(),        
                DropBlock(block_size=block_size, p=p_value),    
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)                
                )  
        self.shortcut = nn.Sequential(
                    LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
                    )
        self.ECA = EfficientChannelAttention(in_channels=out_channels)

    def forward(self, x):
        s = self.shortcut(x)
        x = self.layer(x)
        x = torch.add(x, s)
        x = self.ECA(x)
        return x

class SegmentationTask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationTask, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.layer(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Generator(nn.Module):
    def __init__(self, img_size=128, channels=1, out_channels=1, filtersize=[64, 128, 256, 512], up_mode="bilinear", check_sigmoid=False):
        super(Generator, self).__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.check_sigmoid = check_sigmoid
        self.filtersize = filtersize
        self.depth = len(self.filtersize)
        self.Stem = Stemblock(self.channels, self.filtersize[0])
        #####################################################################################################################
        #Make Encoder Layer
        #by Resblock and Maxpooling
        #####################################################################################################################
        self.encoder = nn.ModuleList()
        for idx in range(self.depth - 1):
            self.down_layer = nn.Sequential(
                LayerNorm(self.filtersize[idx], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(self.filtersize[idx], self.filtersize[idx+1], kernel_size=2, stride=2, padding=0, bias=False)
            )
            layer = [Resblock(self.filtersize[idx], self.filtersize[idx], block_size=7-2*(idx)), 
                     self.down_layer]
            self.encoder.append(nn.Sequential(*layer))
        #####################################################################################################################
        #Make PSPPooling Block
        #After last encoder layer and Before last Decoder layer
        #####################################################################################################################
        self.Pspp1 = PSPPooling(self.filtersize[-1], self.filtersize[-1], up_mode=up_mode)
        self.Pspp2 = PSPPooling(self.filtersize[0], self.filtersize[0], up_mode=up_mode)
        #####################################################################################################################
        #Make Decoder Layer
        #by Attentionblock, Resblock and Upsampling
        #####################################################################################################################
        self.decoder = nn.ModuleList()
        for idx in range(self.depth-1, 0, -1):
            self.Up = nn.Sequential(
                LayerNorm(self.filtersize[idx], eps=1e-6,  data_format="channels_first"),
                nn.ConvTranspose2d(self.filtersize[idx], self.filtersize[idx-1], kernel_size=2, stride=2, padding=0, bias=False)
            )
            layer = [Attentionblock(self.filtersize[idx-1], self.filtersize[idx], self.filtersize[idx]), 
                     self.Up, 
                     Resblock(self.filtersize[idx], self.filtersize[idx-1], block_size=1)]
            self.decoder.append(nn.Sequential(*layer))
        #####################################################################################################################
        #Make Out Layer
        #by SegmentationTask Layer and Sigmoid (if you didn't want to use sigmoid check_sigmoid=False)
        #####################################################################################################################
        self.segmentation_out = SegmentationTask(self.filtersize[0], self.out_channels)
        self.Sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        
    def forward(self, x):
        ##Encoder
        x = self.Stem(x)
        x_downsample = []
        x_downsample.append(x)
        for idx, layer_down in enumerate(self.encoder):
            x = layer_down(x)
            if idx != len(self.encoder)-1:
                x_downsample.append(x)# x1,x2 ...
        x = self.Pspp1(x)
        ##Decoder
        for idx, layer_up in enumerate(self.decoder):
            x = layer_up[0](x_downsample[-1-idx], x)
            x = layer_up[1](x)
            x = torch.cat((x, x_downsample[-1-idx]), dim=1)
            x = layer_up[2](x)
            
        x = self.Pspp2(x)
        ##model output
        x = self.segmentation_out(x)
        if self.check_sigmoid:
            x = self.Sigmoid(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            if isinstance(m, nn.BatchNorm2d) and m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                m.bias.data.fill_(0.01)

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = ResUnetA()
    model.cuda()
    summary_(model, (1, 128, 128), batch_size=1)