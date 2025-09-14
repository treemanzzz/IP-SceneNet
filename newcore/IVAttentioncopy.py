import torch
import torch.nn as nn

class IVA_Module(nn.Module):
    """ Image-Voxel attention module"""
    def __init__(self, img_dim, voxel_dim):
        super(IVA_Module, self).__init__()
        self.img_conv = nn.Conv2d(in_channels=img_dim, out_channels=256, kernel_size=1)
        self.voxel_conv = nn.Conv2d(in_channels=voxel_dim, out_channels=256, kernel_size=1)
        self.img_v = nn.Conv2d(in_channels=img_dim, out_channels=img_dim, kernel_size=1)
        self.voxel_v = nn.Conv2d(in_channels=voxel_dim, out_channels=voxel_dim, kernel_size=1)

        self.img_gamma = nn.Parameter(torch.ones(1))
        self.pc_gamma = nn.Parameter(torch.ones(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img, pc):
        pc = pc.dense() # pc: (16, 64, 64, 64, 32)
        pc = pc.permute(0, 4, 1, 2, 3) # pc: (16, 32, 64, 64, 64)
        pc_b, pc_c, pc_d, pc_h, pc_w = pc.shape
        img_b, img_c, img_h, img_w = img.size()
        pc = pc.contiguous().view(pc_b, pc_c * pc_d, pc_h, pc_w) # pc: (16, 2048, 64, 64)
        img_q = self.img_conv(img).view(img_b, -1, img_h*img_w).permute(0, 2, 1) # img_q: (16, 4096, 256)
        pc_k = self.voxel_conv(pc).view(pc_b, -1, pc_h*pc_w) # pc_k: (16, 256, 4096)
        attention = self.softmax(torch.bmm(img_q, pc_k)) # attention: (16, 4096, 4096)
        del img_q, pc_k
        img_v = self.img_v(img).view(img_b, -1, img_h*img_w) # img_v1: (16, 256, 4096)
        img_v = torch.bmm(img_v, attention.permute(0, 2, 1)) # img_v1: (16, 256, 4096)
        img_v = img_v.view(img_b, img_c, img_h, img_w) # img_v1: (16, 256, 64, 64)
        # img = self.img_gamma*img_v + img
        # del img_v
        pc_v = self.voxel_v(pc).view(pc_b, -1, pc_h*pc_w) # pc_v: (16, 2048, 4096)
        pc_v = torch.bmm(pc_v, attention.permute(0, 2, 1)) # pc_v: (16, 2048, 4096)
        pc_v = pc_v.view(pc_b, pc_c * pc_d, pc_h, pc_w) # pc_v: (16, 2048, 64, 64)
        # pc = self.pc_gamma*pc_v + pc # pc: (16, 2048, 64, 64)
        # del pc_v
        pc_v = pc_v.view(pc_b, pc_c, pc_d, pc_h, pc_w) # pc: (16, 32, 64, 64, 64)
        

        return img_v, pc_v