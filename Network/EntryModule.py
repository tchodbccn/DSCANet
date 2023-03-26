import torch
from torch import nn
import json
from collections import OrderedDict

class EntryModule(nn.Module):
    def __init__(self, paramsdict):
        super(EntryModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=paramsdict["conv1"]["inchannels"], out_channels=paramsdict["conv1"]["outchannels"],
                               kernel_size=paramsdict["conv1"]["kernelsize"],
                               stride=paramsdict["conv1"]["stride"], padding=paramsdict["conv1"]["padding"],
                               bias=paramsdict["conv1"]["bias"])
        if paramsdict["conv1"]["norm"]:
            self.bn1 = nn.BatchNorm2d(paramsdict["conv1"]["outchannels"])
        if paramsdict["conv1"]["active"] == 'relu':
            self.active1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=paramsdict["conv2"]["inchannels"], out_channels=paramsdict["conv2"]["outchannels"],
                               kernel_size=paramsdict["conv2"]["kernelsize"],
                               stride=paramsdict["conv2"]["stride"], padding=paramsdict["conv2"]["padding"],
                               bias=paramsdict["conv2"]["bias"])
        if paramsdict["conv2"]["norm"]:
            self.bn2 = nn.BatchNorm2d(paramsdict["conv2"]["outchannels"])
        if paramsdict["conv1"]["active"] == 'relu':
            self.active2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.active1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.active2(x)
        return x







