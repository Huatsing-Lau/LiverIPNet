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

        self.resnet_layers = nn.ModuleList()
#         self.gates = []
        self.gates = nn.Parameter(torch.zeros((1)))
        self.gates_c = None
        for k in range(5):
            r = resnet50_baseline(pretrained=True, requires_grad=requires_grad)
            r.avgpool = nn.AdaptiveMaxPool2d(output_size=1)
            r.conv1 = nn.Conv2d(input_channel,
                                64,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3),
                                bias=False)
            self.resnet_layers.append(r)
#             self.gates.append(nn.Parameter(
#                 torch.randn(1), requires_grad=True))

        size = [1024, 512, 128]
        fc = [nn.Linear(size[0], size[1]), nn.LeakyReLU()]
        fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.LeakyReLU()])
        fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[2]), nn.LeakyReLU()])
        fc.append(nn.Dropout(0.25))


        self.fc = nn.Sequential(*fc)
        self.classifier = nn.Linear(in_features=128,
                                    out_features=2,
                                    bias=True)
        self.gate_linear = nn.Linear(in_features=32,
                                     out_features=1,
                                     bias=True)

    def forward(self, images, attention_only=False):
        """
        return list of classification logit
        """
        # images.shape: (N, 3, 256, 256)
        # out: (N, 1024)
        h = []
        gates = []
        for i in range(5):
            image = images[:, i*5:i*5+5, :, :]
            h0 = self.resnet_layers[i](image)
#             gate = self.gates[i].cuda()
#             gate = torch.sigmoid(gate)
            h0 = self.fc(h0)
            h0 = self.classifier(h0)
            gate = (torch.sgn(self.gates)+ 1)/2
            h0 = h0 * gate
            h0 = torch.unsqueeze(h0, -1)
            h.append(h0)
            gates.append(gate)

        h = torch.cat([h[0], h[1], h[2], h[3], h[4]], dim=-1)
#         gates = torch.relu(self.gates)
#         output = h * gates
        output = torch.sum(h, dim=-1)        
        self.gates_c = gates
        return output, (h, gates)
    
    def get_gates(self):
        gates = self.gates_c
        gates = torch.cat([i for i in gates], dim=-1)
        return gates

if __name__ == "__main__":
    # 单模态
    # input_channel 原来是视为rgb三通道。可以尝试一下使用层厚为通道？先不做更改
    model = Resmode().cuda()
    images = torch.tensor(torch.randn((10,25,32,32))).to(torch.float32).cuda()
    output, g = model(images)
    print(output, g)
