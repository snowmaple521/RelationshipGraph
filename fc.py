"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network


    简单的非线性完全连接网络
    """
    #dims = [1024,1024]
    def __init__(self, dims, act='ReLU', dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i] #1024
            out_dim = dims[i+1] #1024
            if 0 < dropout:
                layers.append(nn.Dropout(dropout)) #添加一个dropout层 防止过拟合
            #添加一个权重标准化后的线性层
            layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias),dim=None))
            #添加一个Relu层 nn.ReLu层
            if '' != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias),dim=None))
        if '' != act and act is not None:
            layers.append(getattr(nn, act)())

        #将网络添加到sequential容器中
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
