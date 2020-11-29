# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .position_encoding import build_scale_embedding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module,
                 train_backbone: bool,
                 num_channels: int,
                 return_interm_layers: bool = True,
                 hidden_dim: int = 256):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # For deformable-attn, we need intermediate activations
        assert return_interm_layers == True
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        else:
            return_layers = {'layer4': "2"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

        self.c3_conv = nn.Conv2d(backbone.inplanes//4, hidden_dim, kernel_size=1)
        self.c4_conv = nn.Conv2d(backbone.inplanes//2, hidden_dim, kernel_size=1)
        self.c5_conv = nn.Conv2d(backbone.inplanes, hidden_dim, kernel_size=1)
        self.c6_conv = nn.Conv2d(backbone.inplanes, hidden_dim, kernel_size=(3,3), stride=2)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None

        for name, x in xs.items():

            # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

            if name == '0':
                scale_map = self.c3_conv(x)
            elif name == '1':
                scale_map = self.c4_conv(x)
            else:
                scale_map = self.c5_conv(x)
            mask = F.interpolate(m[None].float(), size=scale_map.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(scale_map, mask)

        c6 = self.c6_conv(xs['2'])
        mask = F.interpolate(m[None].float(), size=c6.shape[-2:]).to(torch.bool)[0]
        out['3'] = NestedTensor(c6, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, scale_embedding):
        super().__init__(backbone, position_embedding, scale_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[int, NestedTensor] = {}
        pos: Dict[int, NestedTensor] = {}
        for name, x in xs.items():
            # position encoding

            pos_embedding = self[1](x).to(x.tensors.dtype)
            scale_embedding = self[2](x, int(name)).to(x.tensors.dtype)
            pos_scale = pos_embedding + scale_embedding
            pos[int(name)] = pos_scale
            out[int(name)] = x

        # make scale maps' size is ascent order.
        return [out[3], out[2], out[1], out[0]], [pos[3], pos[2], pos[1], pos[0]]


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    scale_embedding = build_scale_embedding(args)

    train_backbone = args.lr_backbone > 0

    # always return interm_layers
    # return_interm_layers = args.masks
    return_interm_layers = True
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding, scale_embedding)
    model.num_channels = backbone.num_channels
    return model
