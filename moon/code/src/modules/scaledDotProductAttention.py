
import math
from math import sqrt
import torch
import torch.nn as nn
from torch.nn import init
from typing import Union
import numpy as np
from src.modules.base_generator import GeneratorAbstract

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1,isTransfer=False,isReplace=False):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)
        
        self.isTransfer = isTransfer
        self.isReplace = isReplace
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, y , attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        if self.isTransfer:
            y = y.reshape(y.shape[0],self.d_model,-1).permute(0,2,1)


        queries=y
        keys=y
        values=y
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        if self.isReplace:
            out = out.reshape(out.shape[0],-1,int(sqrt(out.shape[1])),int(sqrt(out.shape[1])))
        return out



class ScaledDotProductAttentionGenerator(GeneratorAbstract):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)


    @property
    def out_channel(self)->int:
        return self.in_channel

    def __call__(self,repeat:int=1):
        """
         repeat(=n) [d_k,d_v,h,dropout]
        """
        module = []
        d_k,d_v,h,dropout,isTransfer,isReplace = self.args
        for i in range(repeat):
            module.append(
                ScaledDotProductAttention(
                    d_model = self.in_channel,
                    d_k = d_k,
                    d_v = d_v,
                    h=h,
                    dropout=dropout,
                    isTransfer=isTransfer,
                    isReplace=isReplace
                )
            )
        
        return self._get_module(module)





class CoAtConv(nn.Module):
    """Standard convolution with batch normalization and activation."""

    def __init__(
        self,
        in_channel: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, None] = None,
        groups: int = 1,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Standard convolution with batch normalization and activation.

        Args:
            in_channel: input channels.
            out_channels: output channels.
            kernel_size: kernel size.
            stride: stride.
            padding: input padding. If None is given, autopad is applied
                which is identical to padding='SAME' in TensorFlow.
            groups: group convolution.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        # error: Argument "padding" to "Conv2d" has incompatible type "Union[int, List[int]]";
        # expected "Union[int, Tuple[int, int]]"
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channels,
            kernel_size,
            stride,
            padding=autopad(kernel_size, padding),  # type: ignore
            groups=groups,
            bias=False,
        )
        self.act = Activation(activation)()

        self.conv2 = nn.Conv2d(
            in_channel,
            out_channels,
            kernel_size,
            stride,
            padding=autopad(kernel_size, padding),  # type: ignore
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.conv2(self.act(self.conv1(x)))


class CoAtConvGenerator(GeneratorAbstract):
    """Conv2d generator for parsing module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self._get_divisible_channel(self.args[0] * self.width_multiply)


    def __call__(self, repeat: int = 1):
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        if repeat > 1:
            stride = 1
            # Important!: stride only applies at the end of the repeat.
            if len(args) > 2:
                stride = args[3]
                args[3] = 1

            module = []
            for i in range(repeat):
                if len(args) > 1 and stride > 1 and i == repeat - 1:
                    args[3] = stride

                module.append(CoAtConv(*args))
                args[0] = self.out_channel
        else:
            module = CoAtConv(*args)

        return self._get_module(module)

