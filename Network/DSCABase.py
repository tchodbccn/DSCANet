import torch.nn as nn

class DSCABase(nn.Module):
    def __init__(self):
        super(DSCABase, self).__init__()
        self.reluFirst = False
        self.residualLayer = None
        self.relulayer = nn.ReLU()