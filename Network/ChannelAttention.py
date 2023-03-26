import torch
from torch import nn


class ChannelAttention(nn.Module):
    '''Channel Attention Module'''
    def __init__(self, channelscount, hideunitscount):
        '''
        :param channelscount:Number of channels of input sample
        :param hideunitscount:Number of neurons in MLP hidden layer
        '''
        super(ChannelAttention, self).__init__()

        self.maxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Liner1 = nn.Linear(in_features=channelscount, out_features=hideunitscount)
        self.relu = nn.ReLU()
        self.Liner2 = nn.Linear(in_features=hideunitscount, out_features=channelscount)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):

        m = self.maxPool(x)
        m = torch.flatten(m, 1)
        m = self.Liner1(m)
        m = self.relu(m)
        m = self.Liner2(m)

        a = self.avgPool(x)
        a = torch.flatten(a, 1)
        a = self.Liner1(a)
        a = self.relu(a)
        a = self.Liner2(a)

        mix = m + a
        result = self.sigmod(mix)

        return result






