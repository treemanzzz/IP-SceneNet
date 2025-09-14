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
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize 


class RandomDataset:

    def __init__(self, input_size: int, voxel_size: float) -> None:
        self.input_size = input_size # 点数
        self.voxel_size = voxel_size # 体素格子大小

    def __getitem__(self, index) -> Dict[str, Any]:
        inputs = np.random.uniform(-100, 100, size=(self.input_size, 4)) # 这里第一个参数是随机数下限，第二个是随机数上限，第三个是产生啥样子的随机数，比如这里是input_size行4列
        labels = np.random.choice(10, size=self.input_size) # 从0-10中抽数字，组成input_size大小的数组,作为inputs数据的label

        coords, feats = inputs[:, :3], inputs # coordinates是所有行的前三列，features就是所有
        coords = coords - np.min(coords, axis=0, keepdims=True) # 分别找出x,y,z中最小的值，所有坐标值去减（归一化，没有负数了）
        # sparse_quantize(): Voxelize x, y, z coordinates and remove duplicates.（似乎是采样出原始数据中的某些点，indices即为被采样出的点的下标）
        coords, indices = sparse_quantize(coords,
                                          self.voxel_size,
                                          return_index=True)

        # 转化成tensor
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float) # 将被采样出来的点的特征赋予体素点
        labels = torch.tensor(labels[indices], dtype=torch.long) # 将被采样出来的点的标签赋予体素点

        # 转化成sparsetensor
        input = SparseTensor(coords=coords, feats=feats)
        # label = SparseTensor(coords=coords, feats=labels)
        return {'input': input, 'label': labels}

    def __len__(self):
        return 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp_enabled', action='store_true')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = RandomDataset(input_size=10000, voxel_size=20)
    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=sparse_collate_fn, # Access the sparse tensors in the input list and call sparse_collate.
    )

    model = nn.Sequential(
        spnn.Conv3d(4, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 64, 2, stride=2),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        spnn.Conv3d(64, 64, 2, stride=2, transposed=True),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        spnn.Conv3d(64, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 10, 1),
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler(enabled=args.amp_enabled)

    for k, feed_dict in enumerate(dataflow):
        inputs = feed_dict['input'].to(device='cuda')
        labels = feed_dict['label'].to(device=args.device)

        with amp.autocast(enabled=args.amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs.feats, labels.feats)

        print(f'[step {k + 1}] loss = {loss.item()}')

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # enable torchsparse 2.0 inference
    model.eval()
    # enable fused and locality-aware memory access optimization
    torchsparse.backends.benchmark = True  # type: ignore
    
    with torch.no_grad():
        for k, feed_dict in enumerate(dataflow):
            inputs = feed_dict['input'].to(device=args.device).half()
            labels = feed_dict['label'].to(device=args.device)

            with amp.autocast(enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs.feats, labels.feats)

            print(f'[inference step {k + 1}] loss = {loss.item()}')