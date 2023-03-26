import torch
from torch import nn
import json
from collections import OrderedDict
from torch.nn import functional as F

from Network.DSCA import DSCA
from Network.EntryModule import EntryModule
from Network.ClassificationModule import ClassficationModule

class Dscanet(nn.Module):
    def __init__(self, configroot):
        super(Dscanet, self).__init__()
        path = configroot + '/dscanet.json'
        with open(path) as f:
            config = json.load(f, object_pairs_hook=OrderedDict)

        self.EntryModule = EntryModule(config["entrymodule"])

        self.module1 = DSCA(config["module1"])
        self.module2 = DSCA(config["module2"])
        self.module3 = DSCA(config["module3"])

        self.module4 = DSCA(config["module4_11"])
        self.module5 = DSCA(config["module4_11"])
        self.module6 = DSCA(config["module4_11"])
        self.module7 = DSCA(config["module4_11"])
        self.module8 = DSCA(config["module4_11"])
        self.module9 = DSCA(config["module4_11"])
        self.module10 = DSCA(config["module4_11"])
        self.module11 = DSCA(config["module4_11"])

        self.module12 = DSCA(config["module12"])
        self.module13 = DSCA(config["module13"])

        self.ClassficationModule = ClassficationModule(config["classification"])

    def forward(self, x):
        x = self.EntryModule(x)

        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)

        y = x.clone()

        x = self.module4(x)
        x = self.module5(x)
        x = self.module6(x)
        x = self.module7(x)
        x = self.module8(x)
        x = self.module9(x)
        x = self.module10(x)
        x = self.module11(x)

        z = torch.cat([x, y], 1)

        x = self.module12(z)
        x = self.module13(x)

        x, features = self.ClassficationModule(x)


        return F.softmax(x, dim = 1), features

