from torch import nn
from Network.FlattenBlock import FlattenBlock

class ClassficationModule(nn.Module):
    '''Classification layer'''
    def __init__(self, paramsdict):
        super(ClassficationModule, self).__init__()
        self.Flatten = FlattenBlock()
        self.fc1 = nn.Linear(in_features=paramsdict["inChannels"], out_features=paramsdict["hideUnits"])
        self.fc2 = nn.Linear(in_features=paramsdict["hideUnits"], out_features=paramsdict["outChannels"])
        if paramsdict["active"] == "relu":
            self.active = nn.ReLU()
        else:
            self.active == nn.Tanh()


    def forward(self, x):
        x = self.Flatten(x)
        x = self.fc1(x)
        x = self.active(x)
        features = x
        x = self.fc2(x)
        return x, features




