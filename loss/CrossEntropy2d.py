import torch.nn as nn
from torch.nn import functional as F

from config import cfg

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight = None, ignore_index = cfg.DATASET.IGNORE_LABEL, reduction = 'mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction = reduction,
                                   ignore_index = ignore_index)

    def forward(self, inputs, targets, do_rmi=None):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)