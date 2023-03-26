import torch
from torch import nn
import numpy as np

class SpatialAttention(nn.Module):
    '''Spatial Attention Module'''
    def __init__(self, kernelsize = 7, stride = 1, padding = 0):
        super(SpatialAttention, self).__init__()

        assert kernelsize in (3, 7)
        padding = 3 if kernelsize == 7 else 1

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernelsize, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        result = self.sigmoid(x)

        return result

    def check_composition(self):
        print("{} SpatialAttention Module".format(self.id))
        print(self.conv)
        print(self.sigmoid)


        