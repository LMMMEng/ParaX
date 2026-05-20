# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

try:
    from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
except Exception:
    MMDET_BACKBONES = None

try:
    from mmseg.models.builder import BACKBONES as MMSEG_BACKBONES
except Exception:
    MMSEG_BACKBONES = None

try:
    from .parax import ParaXAdapter, load_pretrained_weights, parax_parameter_dict
except Exception:
    from parax import ParaXAdapter, load_pretrained_weights, parax_parameter_dict


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x.contiguous()
    

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, depth=None,
                 parax_rank=128,
                 parax_kernel_sizes=None,
                 parax_enable_conv=False,
                 parax_router_hidden=16,
                 parax_ls_init_value=1,
                 parax_rs_init_value=1,
                 parax_force_fp16=False):
        super().__init__()
        self.parax_enable_conv = parax_enable_conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adapter = ParaXAdapter(
            dim, parax_rank, depth,
            ls_init_value=parax_ls_init_value,
            rs_init_value=parax_rs_init_value,
            parax_kernel_sizes=parax_kernel_sizes,
            enable_conv=parax_enable_conv,
            router_hidden=parax_router_hidden,
            force_fp16=parax_force_fp16)

    def forward(self, x):
        
        x, params = x

        B, C, H, W = x.shape

        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)

        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.parax_enable_conv:
            x = self.adapter(x, params, H, W)
        else:
            x = self.adapter(x, params)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        
        return (x, params)
    

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0, layer_scale_init_value=1e-6,
                 head_init_scale=1, pretrained=None, pretrained_cfg=None,
                 parax_rank=128,
                 parax_kernel_sizes=None,
                 parax_enable_conv=False,
                 parax_router_hidden=16,
                 parax_ls_init_value=1,
                 parax_rs_init_value=1,
                 parax_force_fp16=False,
                 ):
        super().__init__()
        self.pretrained = pretrained
        # self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0

        self.peft_params = nn.ParameterList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        depth=depths[i],
                        parax_rank=parax_rank,
                        parax_kernel_sizes=parax_kernel_sizes,
                        parax_enable_conv=parax_enable_conv,
                        parax_router_hidden=parax_router_hidden,
                        parax_ls_init_value=parax_ls_init_value,
                        parax_rs_init_value=parax_rs_init_value,
                        parax_force_fp16=parax_force_fp16) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

            dim = dims[i]
            num_layers = depths[i]
            parax_params = parax_parameter_dict(
                dim, parax_rank, num_layers=num_layers,
                parax_kernel_sizes=parax_kernel_sizes,
                enable_conv=parax_enable_conv)

            parax_params = nn.ParameterDict(parax_params)
            self.peft_params.append(parax_params)

        self.extra_norm = nn.ModuleList()
        for i in range(4):
            self.extra_norm.append(LayerNorm2d(dims[i]))

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
        
        if self.pretrained is not None:
            load_pretrained_weights(self, self.pretrained, strict=False)

        # freeze
        for name, param in self.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            elif 'peft_params' in name:
                param.requires_grad = True
            elif 'extra_norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward_features(self, x):
        outs = []
        for i in range(4):
            param = self.peft_params[i]
            if i == 0:
                x = self.downsample_layers[i](x)
            else:
                x = self.downsample_layers[i](x[0])
            x = (x, param)
            x = self.stages[i](x)
            outs.append(self.extra_norm[i](x[0]))
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


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


def convnext_base_parax(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    # 6.46
    return model



def convnext_large_parax(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    # 9.16
    return model


for _backbone_registry in (MMDET_BACKBONES, MMSEG_BACKBONES):
    if _backbone_registry is not None:
        _backbone_registry.register_module()(convnext_base_parax)
        _backbone_registry.register_module()(convnext_large_parax)