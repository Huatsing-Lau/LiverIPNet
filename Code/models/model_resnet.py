# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet import resnet50_baseline
# from resnet_custom import resnet50_baseline

class Resmode(nn.Module):
    def __init__(self,
                 dropout=False,
                 requires_grad=True,
                 input_channel=5):

        super().__init__()

        self.resnet_baseline = resnet50_baseline(pretrained=True,
                                                 requires_grad=requires_grad)
        # 将平均池化改为最大池化
        self.resnet_baseline.avgpool = nn.AdaptiveMaxPool2d(output_size=1)

        if input_channel != 3:
            assert requires_grad == True, 'One should not change the first convolution layer when requires_grad is False.'
            self.resnet_baseline.conv1 = nn.Conv2d(input_channel,
                                                   64,
                                                   kernel_size=(7, 7),
                                                   stride=(2, 2),
                                                   padding=(3, 3),
                                                   bias=False)
            
        size = [1024, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LeakyReLU()]
        fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.LeakyReLU()])
        fc.append(nn.Dropout(0.25))


        self.fc = nn.Sequential(*fc)
        self.classifier = nn.Linear(in_features=512,
                                    out_features=2,
                                    bias=True)

    def forward(self, images, attention_only=False):
        """
        return list of classification logit
        """
        # images.shape: (N, 3, 256, 256)
        # out: (N, 1024)
        h = self.resnet_baseline(images)
        
        h = self.fc(h)
 
        output = self.classifier(h)
        return output

if __name__ == "__main__":
    # 单模态
    # input_channel 原来是视为rgb三通道。可以尝试一下使用层厚为通道？先不做更改
    model = Resmode()
    images = torch.tensor(np.zeros((10,5,32,32))).to(torch.float32)
    output = model(images)
    print(output)
