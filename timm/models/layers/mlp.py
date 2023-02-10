""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import nn as nn


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., size=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.size = size

    def forward(self, x, size=None):
        if size is None:
            size = self.size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpDWBN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
        size=None
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = in_features * 3
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.size = size

    def forward(self, x):
        if len(x.shape) == 3:
            B, N, C = x.shape
            x_ = x.permute(0, 2, 1).reshape(B, C, self.size[0], self.size[1])

            x_ = self.fc1(x_)
            x_ = self.norm1(x_)
            x_ = self.act1(x_)
            x_ = self.dw3x3(x_)
            x_ = self.norm2(x_)
            x_ = self.act2(x_)
            x_ = self.drop(x_)
            x_ = self.fc2(x_)
            x_ = self.norm3(x_)
            x_ = self.act3(x_)
            x_ = self.drop(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x = x_
            return x

        elif len(x.shape) == 4:
            x = self.fc1(x)
            x = self.norm1(x)
            x = self.act1(x)
            x = self.dw3x3(x)
            x = self.norm2(x)
            x = self.act2(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.norm3(x)
            x = self.act3(x)
            x = self.drop(x)
            return x

        else:
            raise RuntimeError("Unsupported input shape: {}".format(x.shape))


class MixMlp(nn.Module):  # For FFN
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
        size=None
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = in_features * 3
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=3,
            stride=1,
            groups=in_features,
            padding=1,
        )
        self.dw5x5 = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=5,
            stride=1,
            groups=in_features,
            padding=2,
        )
        self.dw7x7 = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=7,
            stride=1,
            groups=in_features,
            padding=3,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)
        self.size = size

    def forward(self, x, size=None):
        if size is None:
            size = self.size
        if len(x.shape) == 3:
            B, N, C = x.shape
            x_ = x.permute(0, 2, 1).reshape(B, C, size[0], size[1])

            x_ = self.fc1(x_)
            x_ = self.norm1(x_)
            x_ = self.act1(x_)
            x3 = self.dw3x3(x_[:, :C])
            x5 = self.dw5x5(x_[:, C:-C])
            x7 = self.dw7x7(x_[:, -C:])
            x_ = torch.cat([x3, x5, x7], dim=1)
            x_ = self.norm2(x_)
            x_ = self.act2(x_)
            x_ = self.drop(x_)
            x_ = self.fc2(x_)
            x_ = self.norm3(x_)
            x_ = self.act3(x_)
            x_ = self.drop(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x = x_
            return x

        else:
            raise RuntimeError("Unsupported input shape: {}".format(x.shape))


class ConvLayerNormMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
        size=None
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = in_features * 3
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.LayerNorm(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(drop)
        self.size = size

    def forward(self, x):
        if len(x.shape) == 3:
            B, N, C = x.shape
            x_ = x.permute(0, 2, 1).reshape(B, C, self.size[0], self.size[1])

            x_ = self.fc1(x_)

            x_ = self.norm1(x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_ = self.act1(x_)
            x_ = self.dw3x3(x_)
            x_ = self.norm2(x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_ = self.act2(x_)
            x_ = self.drop(x_)
            x_ = self.fc2(x_)
            x_ = self.norm3(x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_ = self.act3(x_)
            x_ = self.drop(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x = x_
            return x

        else:
            raise RuntimeError("Unsupported input shape: {}".format(x.shape))


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 gate_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
