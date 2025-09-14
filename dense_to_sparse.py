
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

torch.manual_seed(42)

num_points = 100
dim = 4
feats = torch.randn([num_points, dim]).cuda()
coords = torch.randint(0, 10, [num_points, 4]).cuda()
coords[:50, 0] = 0        # Set batch_idx to 0
coords[50:, 0] = 1

spatial_range = (2, 20, 20, 20)
x = torchsparse.SparseTensor(feats, coords, spatial_range = spatial_range)

y = x.dense()
y = y.permute(0, 4, 1, 2, 3) # (1, 4, 20, 20, 20)
# y = y.permute(0, 4, 1, 2, 3).squeeze() # (4, 20, 20, 20)


# # 创建一个示例的稠密张量（dense tensor）
# dense_tensor = torch.tensor([[0, 0, 3, 0],
#                              [0, 0, 0, 0],
#                              [0, 0, 0, 0],
#                              [0, 2, 0, 0]], dtype=torch.float32)

for batch_idx in range(y.size(0)):
    # 找出稠密张量中非零元素的坐标
    indices = torch.nonzero(y[batch_idx])
    # 获取第一列元素（即为通道列）等于0的行索引
    ind = torch.nonzero(indices[:, 0] == 0).squeeze()
    # 获取满足条件的行对应的值
    coor = indices[ind, :]
    # 将第一列改为batch_idx
    coor[:, 0] = batch_idx
    if batch_idx == 0:
        c = coor
    else:
        c = torch.cat((c, coor))
    non_zero_values = []
    for channel_idx in range(y.size()[1]):
        non_zero_indices = torch.nonzero(y[batch_idx, channel_idx, :, :, :])  # 获取非零元素的索引
        non_zero_values_channel = y[batch_idx, channel_idx][non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]
        non_zero_values.append(non_zero_values_channel) #❓为什么每个通道都是49个，因为是一个cube
    fuck = torch.stack(non_zero_values).t()
    if batch_idx == 0:
        f = fuck
    else:
        f = torch.cat((f, fuck))



sptensor1 = torchsparse.SparseTensor(f, c)

print(coor)




