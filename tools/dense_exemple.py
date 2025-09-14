import torch
import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as F

torch.manual_seed(42)

num_points = 100
dim = 4
feats = torch.randn([num_points, dim]).cuda()
coords = torch.randint(0, 10, [num_points, 4]).cuda()
coords[:, 0] = 0        # Set batch_idx to 0

spatial_range = (1, 10, 10, 10)
x = torchsparse.SparseTensor(feats, coords, spatial_range = spatial_range)

y = x.dense()
y = y.permute(0, 4, 1, 2, 3) # pc: (16, 32, 64, 64, 64)
pc_b, pc_c, pc_d, pc_h, pc_w = y.shape

y = y.contiguous().view(pc_b, pc_c * pc_d, pc_h, pc_w) # pc: (16, 2048, 64, 64)
print(x)
print(y)