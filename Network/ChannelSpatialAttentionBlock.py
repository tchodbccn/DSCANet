import torch
from torch import nn

from Network.ChannelAttention import ChannelAttention
from Network.SpatialAttention import SpatialAttention

class ChannelSpatialAttentionBlock(nn.Module):
    '''Space and Channel Attention Module'''
    def __init__(self, channelscount, hideunitscount):
        '''
        :param channelscount:Number of channels of input sample
        :param hideunitscount:Number of neurons in MLP hidden layer
        '''
        super(ChannelSpatialAttentionBlock, self).__init__()

        self.channelAttention = ChannelAttention( channelscount=channelscount, hideunitscount=hideunitscount)
        self.spatialAttention = SpatialAttention()

    def forward(self, x):
        channelattention = self.channelAttention(x)

        channelattention = torch.unsqueeze(channelattention, -1)
        channelattention = torch.unsqueeze(channelattention, -1)

        F_prime1 = torch.mul(x , channelattention)
        F_prime2 = self.spatialAttention(F_prime1) * F_prime1

        return F_prime2




