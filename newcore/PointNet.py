import torchsparse
import torchsparse.nn as spnn
import torch
from torch import nn
from torchsparse import SparseTensor

__all__ = ['PointNet']


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class PointNet(nn.Module):
    def __init__(self) -> None:
        super(PointNet, self).__init__()

        cs = [32, 32, 64, 128, 256]

        '''
        输入图片大小 W * W
        卷积核大小 F * F
        步长 S
        padding的像素数 P
        于是我们可以得出计算公式为：
        N = (W - F + 2P)/S + 1
        '''
        # (3, 512, 512, 512) ↓
        self.stem = nn.Sequential(
            spnn.Conv3d(3, cs[0], kernel_size=2, stride=2), # (32, 256, 256, 256)
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=2, stride=2), # (32, 128, 128, 128)
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1), # (32, 128, 128, 128)
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        # (32, 128, 128, 128) ↓
        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), # (32, 64, 64, 64)
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), # ≈ (32, 64, 64, 64)
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1), # ≈ (32, 64, 64, 64)
        )

        # (32, 64, 64, 64) ↓
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1), # (32, 32, 32, 32)
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1), # ≈ (64, 32, 32, 32)
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1), # (64, 32, 32, 32)
        )

        # (64, 32, 32, 32) ↓
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1), # (64, 16, 16, 16)
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1), # ≈ (128, 16, 16, 16)
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1), # ≈ (128, 16, 16, 16)
        )

        # (128, 16, 16, 16) ↓
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1), # (128, 8, 8, 8)
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1), # ≈ (256, 8, 8, 8)
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), # ≈ (256, 8, 8, 8)
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

def main():
    from typing import Any, Dict
    import numpy as np


    import torch
    import torch.utils.data

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
            inputs = np.random.uniform(-100, 100, size=(self.input_size, 3)) # 这里第一个参数是随机数下限，第二个是随机数上限，第三个是产生啥样子的随机数，比如这里是input_size行4列

            coords, feats = inputs[:, :3], inputs # coordinates是所有行的前三列，features就是所有
            coords = coords - np.min(coords, axis=0, keepdims=True) # 分别找出x,y,z中最小的值，所有坐标值去减（归一化，没有负数了）
            # sparse_quantize(): Voxelize x, y, z coordinates and remove duplicates.（似乎是采样出原始数据中的某些点，indices即为被采样出的点的下标）
            coords, indices = sparse_quantize(coords,
                                            self.voxel_size,
                                            return_index=True)

            # 转化成tensor
            coords = torch.tensor(coords, dtype=torch.int)
            feats = torch.tensor(feats[indices], dtype=torch.float) # 将被采样出来的点的特征赋予体素点

            # 转化成sparsetensor
            input = SparseTensor(coords=coords, feats=feats)
            return {'input': input}

        def __len__(self):
            return 10

    net = PointNet()
    net.cuda()
    # print(net)

    dataset = RandomDataset(input_size=100000, voxel_size=0.390625)
    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=sparse_collate_fn, # Access the sparse tensors in the input list and call sparse_collate.
    )

    for k, feed_dict in enumerate(dataflow):
        inputs = feed_dict['input'].to(device='cuda') 
        outputs = net.stage1(inputs)


        # outputs = outputs.dense()
        print(outputs.size())

    # from dataset import SceneDataset
    # import torch.utils.data
    # from torchsparse.utils.collate import sparse_collate_fn

    # net = PointNet()
    # net.cuda()
    # print(net)

    # dataset = SceneDataset("/home/ExtraData/SceneClass/Data/test_data", 0.005)
    # dataflow = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=16,
    #     num_workers=16,
    #     shuffle=True,
    #     collate_fn=sparse_collate_fn, # Access the sparse tensors in the input list and call sparse_collate.
    # )

    # for k, feed_dict in enumerate(dataflow):
    #     inputs = feed_dict['point'].to(device='cuda')
        
    #     outputs = net.stage1(inputs)
    #     print(outputs)
if __name__ == "__main__":
    main()

