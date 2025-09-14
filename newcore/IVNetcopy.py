import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from torchvision.models import resnet50
from PointNet import PointNet
from torchsparse import SparseTensor
from IVAttention import IVA_Module
from dense_to_sparseV2 import dense_to_sparse

class IVNet(torch.nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()
        self.imgEncoder = resnet50()
        self.pcEncoder = PointNet()
        self.attention1 = IVA_Module(img_dim = 256, voxel_dim = 2048)
        self.attention2 = IVA_Module(img_dim = 512, voxel_dim = 2048)
        self.attention3 = IVA_Module(img_dim = 1024, voxel_dim = 2048)
        self.attention4 = IVA_Module(img_dim = 2048, voxel_dim = 2048)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(
        #     nn.Linear(2304, 1024), 
        #     nn.ReLU(),
        #     nn.Linear(1024, num_class),
        #     nn.LogSoftmax(dim=1)
        # )
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048), #####################################################
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_class),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, img, pc):
        '''
        batch_size = 16
        '''
        img = self.imgEncoder.conv1(img)
        img = self.imgEncoder.bn1(img)
        img = self.imgEncoder.relu(img)
        img = self.imgEncoder.maxpool(img)
        img = self.imgEncoder.layer1(img) # img: (16, 256, 64, 64)
        pc = self.pcEncoder.stem(pc)
        pc = self.pcEncoder.stage1(pc) # pc: (16, 32, 64, 64, 64)
        spatial_range = (16, 64, 64, 64)
        pc = SparseTensor(coords = pc.coords, feats = pc.feats, spatial_range = spatial_range)
        img, pc = self.attention1(img, pc) # img: (16, 256, 64, 64); pc: (16, 32, 64, 64, 64)
        pc = dense_to_sparse(pc)
        img = self.imgEncoder.layer2(img) # img: (16, 512, 32, 32)
        pc = self.pcEncoder.stage2(pc) # pc: (16, 64, 32, 32, 32)
        spatial_range = (16, 32, 32, 32)
        pc = SparseTensor(coords = pc.coords, feats = pc.feats, spatial_range = spatial_range)  
        img, pc = self.attention2(img, pc) # img: (16, 512, 32, 32); pc: (16, 64, 32, 32, 32)
        pc = dense_to_sparse(pc)
        img = self.imgEncoder.layer3(img) # img: (16, 1024, 16, 16)
        pc = self.pcEncoder.stage3(pc) # pc: (16, 128, 16, 16, 16)             
        spatial_range = (16, 16, 16, 16)
        pc = SparseTensor(coords = pc.coords, feats = pc.feats, spatial_range = spatial_range)  
        img, pc = self.attention3(img, pc) # img: (16, 1024, 16, 16); pc: (16, 128, 16, 16, 16)
        pc = dense_to_sparse(pc)
        img = self.imgEncoder.layer4(img) # img: (16, 2048, 8, 8)
        pc = self.pcEncoder.stage4(pc) # pc: (16, 256, 8, 8, 8)             
        spatial_range = (16, 8, 8, 8)
        pc = SparseTensor(coords = pc.coords, feats = pc.feats, spatial_range = spatial_range)
        img, pc = self.attention4(img, pc) # img: (16, 2048, 8, 8); pc: (16, 256, 8, 8, 8) 
        pc_b, pc_c, pc_d, pc_h, pc_w = pc.shape
        pc = pc.contiguous().view(pc_b, pc_c * pc_d, pc_h, pc_w) # pc: (16, 2048, 8, 8) 
        # pc, _ = torch.max(pc, dim=4) # (16, 256, 8, 8)     
        img = self.avg_pool(img).squeeze() # (16, 2048)
        pc = self.avg_pool(pc).squeeze() # (16, 2048)
        fused_feature = torch.cat((img, pc), dim=-1) # (16, 4096)
        prob = self.fc(fused_feature)

        return prob  

class get_loss(nn.Module):
    def __init__(self) -> None:
        super(get_loss, self).__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, pred, target):
        # pred: [B, num_class]
        # target: [B, num_class]
        pred = pred.squeeze()
        loss = self.nll_loss(pred, target.long())

        return loss   

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
    from dataset import SceneDataset
    import torch.utils.data

    # class RandomDataset:

    #     def __init__(self, input_size: int, voxel_size: float) -> None:
    #         self.input_size = input_size # 点数
    #         self.voxel_size = voxel_size # 体素格子大小

    #     def __getitem__(self, index) -> Dict[str, Any]:
    #         inputs = np.random.uniform(-100, 100, size=(self.input_size, 3)) # 这里第一个参数是随机数下限，第二个是随机数上限，第三个是产生啥样子的随机数，比如这里是input_size行4列


    #         coords, feats = inputs[:, :3], inputs # coordinates是所有行的前三列，features就是所有
    #         coords = coords - np.min(coords, axis=0, keepdims=True) # 分别找出x,y,z中最小的值，所有坐标值去减（归一化，没有负数了）
    #         # sparse_quantize(): Voxelize x, y, z coordinates and remove duplicates.（似乎是采样出原始数据中的某些点，indices即为被采样出的点的下标）
    #         coords, indices = sparse_quantize(coords,
    #                                         self.voxel_size,
    #                                         return_index=True)

    #         # 转化成tensor
    #         coords = torch.tensor(coords, dtype=torch.int)
    #         feats = torch.tensor(feats[indices], dtype=torch.float) # 将被采样出来的点的特征赋予体素点


    #         # 转化成sparsetensor
    #         input = SparseTensor(coords=coords, feats=feats)
    #         img = torch.randn((3, 256, 256))
    #         # label = SparseTensor(coords=coords, feats=labels)
    #         return {'pc': input, 'img': img}

    #     def __len__(self):
    #         return 10
    
    # dataset = RandomDataset(input_size=100000, voxel_size=0.5)
    # dataflow = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=16,
    #     collate_fn=sparse_collate_fn, # Access the sparse tensors in the input list and call sparse_collate.
    # )
    dataset = SceneDataset("/home/ExtraData/SceneClass/Data/test_data", 0.00390625)
    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=16,
        shuffle=True,
        collate_fn=sparse_collate_fn, # Access the sparse tensors in the input list and call sparse_collate.
    )    
    net = IVNet(num_class=7).cuda()
    for k, feed_dict in enumerate(dataflow):
        pc = feed_dict['point'].to(device='cuda') 
        img = feed_dict['image'].to(device='cuda')
        prob = net(img, pc)
        target = torch.randint(0, 6, size=[16]).cuda()
        loss = get_loss()(prob, target)
        print(loss)

        # outputs = outputs.dense()

    # print(net)

    # img = torch.randn((16, 2048, 7, 7)).cuda()
    # pc = torch.randn((16, 768, 24, 24)).cuda()

    # result = net.forward(img, pc)
    # target = torch.randint(0, 6, size=[16]).cuda()
    # loss = get_loss()(result, target)
    # print(loss)


if __name__ == "__main__":
    main()  