import os
import numpy as np
from typing import Any, Dict
import tifffile
from PIL import Image
import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset

import torchsparse
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize 


class_dic = {
    'office building': 0,
    'industrial area': 1,
    'greenland': 2,
    'residential': 3,
    'transport': 4,
    'sport field': 5,
    'farmland': 6,
}

class SceneDataset(Dataset):
    def __init__(self, data_path: str, voxel_size: float) -> None:
        self.data_path = data_path # 数据路径

        self.voxel_size = voxel_size # 体素格子大小

        self.img_list = [] # 图像数据
        self.voxel_list = [] # 点云体素数据
        self.feat_list = [] # 点云特征数据
        self.label = [] # 标签

        for c1ass in os.listdir(self.data_path): # 类别的8个文件夹
            class_path = os.path.join(self.data_path, c1ass)
            for files in os.listdir(class_path):
                self.label.append(class_dic[c1ass]) # 根据class给label
                folder_path = os.path.join(class_path, files) # 各类别下的文件夹
                file_list = os.listdir(folder_path) # 文件夹内的具体文件
                for fi in file_list:
                    if fi == 'Point.npy':
                        loadData = np.load(os.path.join(folder_path, "Point.npy"))
                        centroid = np.mean(loadData, axis=0) # 求质心，实际上就是求均值
                        loadData = loadData - centroid # 平移质心
                        m = np.max(np.sqrt(np.sum(loadData ** 2, axis=1))) # 点云最大半径
                        loadData = loadData / m # 对点云进行缩放
                        coords, feats = loadData[:, :3], loadData # coordinates是所有行的前三列，features就是所有
                        coords = coords - np.min(coords, axis=0, keepdims=True) # 分别找出x,y,z中最小的值，所有坐标值去减（归一化）
                        # sparse_quantize(): Voxelize x, y, z coordinates and remove duplicates.（似乎是采样出原始数据中的某些点，indices即为被采样出的点的下标）
                        coords, indices = sparse_quantize(coords, # 会得到很规整的coords，feats则通过indices索引保留原本风貌
                                          self.voxel_size,
                                          return_index=True)

                        self.voxel_list.append(coords)
                        self.feat_list.append(feats[indices])
                    else:
                        img = tifffile.TiffFile(os.path.join(folder_path, "img.tiff")).asarray().astype(np.uint8)  # C, H, W
                        img = Image.fromarray(img.transpose(1, 2, 0))  # H, W, C
                        img = img.resize((256, 256), Image.Resampling.LANCZOS)
                        img = np.asarray(img)  # H, W, C
                        img = img.transpose(2, 0, 1)  # C, H, W
                        self.img_list.append(img) 

    def __getitem__(self, index) -> Dict[str, Any]:
        label = int(self.label[index])
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # coords，feats转化成tensor
        coords = torch.tensor(self.voxel_list[index], dtype=torch.int)
        feats = torch.tensor(self.feat_list[index], dtype=torch.float) # 将被采样出来的点的特征赋予体素点
        # 转化成sparsetensor
        pc_tensor = SparseTensor(coords=coords, feats=feats)

        img_tensor = torch.tensor(self.img_list[index]).float()  # C, H, W

        return {'point': pc_tensor, 'image': img_tensor, 'label': label_tensor} 

    def __len__(self):
        return len(self.img_list)
    
# def main():
#     from torchsparse.utils.collate import sparse_collate_fn
#     dataset = SceneDataset("/home/ExtraData/SceneClass/Data/test_data", 0.005)
#     dataflow = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=16,
#         num_workers=16,
#         shuffle=True,
#         collate_fn=sparse_collate_fn, # Access the sparse tensors in the input list and call sparse_collate.
#     )

#     for k, feed_dict in enumerate(dataflow):
#         point = feed_dict['point'].to(device='cuda')
#         image = feed_dict['image'].to(device='cuda')
#         label = feed_dict['label'].to(device='cuda')
#         print("!")
#     # img_tensor, pc_tensor, label_tensor = dataset[0]
#     # img_tensor, pc_tensor, label_tensor = img_tensor.cuda(), pc_tensor.cuda(), label_tensor.cuda()
#     # print(dataset[0])
#     # print(img_tensor)
#     # print(pc_tensor)
#     # print(label_tensor)


# if __name__ == "__main__":
#     main()