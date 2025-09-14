# 图像特征提取网络
import torch
import torch.nn as nn
from torchvision.models import resnet50

__all__ = ['ImageNet']


class ImageNet(nn.Module):
    def __init__(self, backbone) -> None:
        super(ImageNet, self).__init__()
        self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
    
    def forward(self, img):
        img_feature = self.backbone(img) # [B, 2048, 8, 8]
        return img_feature
    
def main():
    from torchvision.models import resnet50
    resnet = resnet50()
    rs_img = torch.randn((2, 3, 256, 256))
    x = resnet.conv1(rs_img)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    f = resnet.layer1(x)
    print(f.size())


if __name__ == "__main__":
    main()