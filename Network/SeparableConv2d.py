import torch
from torch import nn

class SeparableConv2d(nn.Module):
    '''Separable convolution module'''
    def __init__(self, in_Channels, out_Channels, kernel_size = 3, stride = 1, padding = 0, dilation = 1):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_Channels, out_channels=in_Channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=in_Channels, bias=True)
        self.bn = nn.BatchNorm2d(in_Channels)
        self.pointwise = nn.Conv2d(in_channels=in_Channels, out_channels=out_Channels, kernel_size=1, stride=1, padding=0,
                                   dilation=1, groups=1)

    def check_composition(self):
        print("{} SeparableConv2d Module".format(self.id))
        print(self.conv)
        print(self.bn)
        print(self.pointwise)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)

        return x




