import torch
import torch.nn as nn

class FlattenBlock(nn.Module):
    def __init__(self):
        super(FlattenBlock, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        return x