import torch
import torch.nn as nn
from torchvision.models import resnet50
from core.PointNet import PointNet
from torchsparse import SparseTensor

class AttentionNet(torch.nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()
        self.imgEncoder = resnet50()
        self.pcEncoder = PointNet()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.img_linaerq1 = nn.Conv2d(256, 256, 1, 1)
        self.img_linaerv1 = nn.Conv2d(256, 256, 1, 1)
        self.pc_linaerk1 = nn.Conv2d(2048, 256, 1, 1)
        self.pc_linaerv1 = nn.Conv2d(2048, 2048, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(2304, 1024), 
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
        pc1 = SparseTensor(coords=pc.coords, feats=pc.feats, spatial_range = spatial_range)
        dense_pc1 = pc1.dense() # dense_pc1: (16, 64, 64, 64, 32)
        dense_pc1 = dense_pc1.permute(0, 4, 1, 2, 3) # dense_pc1: (16, 32, 64, 64, 64)
        dense_pc1_2D = dense_pc1.view(16, 32 * 64, 64, 64) # dense_pc1_2D: (16, 2048, 64, 64)
        del dense_pc1, pc1, spatial_range
        img_q1 = self.img_linaerq1(img).view(16, -1, 64*64).permute(0, 2, 1) # img_q: (16, 4096, 256)
        pc_k1 = self.pc_linaerk1(dense_pc1_2D).view(16, -1, 64*64) # pc_k: (16, 256, 4096)
        attention = self.softmax(torch.bmm(img_q1, pc_k1)) # attention: (16, 4096, 4096)
        del img_q1, pc_k1
        img_v1 = self.img_linaerv1(img).view(16, -1, 64*64) # img_v1: (16, 256, 4096)
        img_v1 = torch.bmm(img_v1, attention.permute(0, 2, 1)) # img_v1: (16, 256, 4096)
        img_v1 = img.view(16, 256, 64, 64)
        img = self.gamma1*img_v1 + img
        del img_v1
        pc_v1 = self.pc_linaerv1(dense_pc1_2D).view(16, -1, 64*64) # pc_v1: (16, 2048, 4096)
        pc_v1 = torch.bmm(pc_v1, attention.permute(0, 2, 1)) # pc_v1: (16, 2048, 4096)
        pc_v1 = pc_v1.view(16, 32, 64, 64, 64)       




        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out    


        img_feature = self.avg_pool(img).squeeze() # (B, 2048)
        pc_feature = self.avg_pool(pc).squeeze() # (B, 768)
        fused_feature = torch.cat((img_feature, pc_feature), dim=-1) # (B, 2816)
        prob = self.fc(fused_feature)

        return prob       