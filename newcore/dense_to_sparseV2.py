import argparse
import random
from typing import Any, Dict

import torch
import torchsparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def dense_to_sparse(tensor: torch.Tensor) -> torchsparse.tensor.SparseTensor:
    '''
    tensor: (B, C, D, H, W)
    '''
    # 找出第二维度相加后不为零的下标
    non_zero_indices = torch.nonzero((tensor.sum(dim=(1))) != 0).squeeze()
    # 根据下标索引
    value = tensor[non_zero_indices[:, 0], :, non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]

    sptensor = torchsparse.SparseTensor(feats = value, coords = non_zero_indices.type(torch.int))

    return sptensor

# tensor = torch.ones(16, 32, 64, 64, 64).cuda()
# sptensor = dense_to_sparse(tensor)
# print(sptensor)



