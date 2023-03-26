import torch.nn as nn
import torch


from Network.ChannelSpatialAttentionBlock import ChannelSpatialAttentionBlock
from Network.SeparableConv2d import SeparableConv2d
from Network.DSCABase import DSCABase

class DSCA(DSCABase):
    def __init__(self, paramsdict):
        super(DSCA, self).__init__()
        self.pipeline = nn.Sequential()
        self.reluFirst = paramsdict["reluFirst"]
        self.parts = paramsdict["part"]
        index = 1
        for i in range(len(self.parts)):
            key = self.parts[i]
            curconfig = paramsdict[key]
            if curconfig["type"] == "dp":
                self.pipeline.add_module("convdw" + str(index),SeparableConv2d(in_Channels=curconfig["dwinchannels"],
                                        out_Channels=curconfig["dwoutchannels"], kernel_size=curconfig["dwkernelsize"],
                                        stride=curconfig["dwstride"], padding=curconfig["dwpadding"]))
                index += 1
                self.pipeline.add_module("conv" + str(index), nn.Conv2d(in_channels=curconfig["dwoutchannels"],
                                        out_channels=curconfig["convoutchannels"],kernel_size=curconfig["convkernelsize"],
                                        stride=curconfig["convstride"], padding=curconfig["convpadding"]))
                index += 1
                if curconfig["bn"] is True:
                    self.pipeline.add_module("bn" + str(index), nn.BatchNorm2d(curconfig["convoutchannels"]))
                    index += 1
                if curconfig["active"] != "none":
                    if curconfig["active"] == "relu":
                        self.pipeline.add_module("relu" + str(index), nn.ReLU())
                    else:
                        self.pipeline.add_module("relu" + str(index), nn.ReLU())
                    index += 1
            elif curconfig["type"] == "maxpool":
                self.pipeline.add_module("maxpool" + str(index), nn.MaxPool2d(kernel_size=curconfig["kerneksize"],
                                        stride=curconfig["stride"], padding=curconfig["padding"]))
                index += 1
            elif curconfig["type"] == "gavgpool":
                self.pipeline.add_module("gavgpool" + str(index), nn.AdaptiveAvgPool2d((1, 1)))
                index += 1
            elif curconfig["type"] == "attention":
                self.pipeline.add_module("attention" + str(index),
                                         ChannelSpatialAttentionBlock(channelscount=curconfig["channelscount"],
                                        hideunitscount=curconfig["hideuintscount"]))
                index += 1
        if paramsdict["res"] == True:
            self.residualLayer = torch.nn.Conv2d(in_channels=paramsdict["resparams"]["inchannels"],
                                             out_channels=paramsdict["resparams"]["out_channels"],
                                             kernel_size=paramsdict["resparams"]["kerneksize"],
                                             stride=paramsdict["resparams"]["stride"],
                                             padding=paramsdict["resparams"]["padding"])

    def forward(self, x):
        if self.reluFirst:
            x = self.relulayer(x)

        if self.residualLayer is not None:
            residual = self.residualLayer(x)
        x = self.pipeline(x)
        if self.residualLayer is not None:
            x.add_(residual)
        return x


