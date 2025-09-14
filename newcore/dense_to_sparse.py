import argparse
import random
from typing import Any, Dict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


def dense_to_sparse(tensor: torch.Tensor) -> torchsparse.tensor.SparseTensor:
    '''
    x: (B, C, D, H, W)
    '''
    for b in range(tensor.size()[0]):
        vectors_list = []
        indices_list = []
        for i in range(tensor.size()[2]):
            for j in range(tensor.size()[3]):
                for k in range(tensor.size()[4]):
                    selected_elements = tensor[b, :, i, j, k]
                    if torch.any(selected_elements != 0):
                        vectors_list.append(selected_elements)
                        indices_list.append(torch.Tensor([b, i, j, k]))
        if b == 0:
            # 将列表中的张量堆叠起来
            f = torch.stack(vectors_list)
            feats = f.clone()
            c = torch.stack(indices_list)
            coords = c.clone()
        else:
            # 将列表中的张量堆叠起来
            f = torch.stack(vectors_list)
            feats = torch.cat((feats, f))
            c = torch.stack(indices_list)
            coords = torch.cat((coords, c)).type(torch.int)
            del f, c


    sptensor = torchsparse.SparseTensor(feats = feats, coords = coords)

    return sptensor




