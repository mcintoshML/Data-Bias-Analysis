import torch
import torch.nn as nn


class simpleECG(nn.Module):
    def __init__(self,num_classes=5,num_channels=12,**kwargs):
        super(simpleECG, self).__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes

        k = 64
        M = 7
        modules = []
        #modules.append(nn.Identity())
        for i in range(0,M):
            if i ==0:
                modules.append(nn.Conv1d(self.num_channels,k,7,padding=3))
                modules.append(nn.BatchNorm1d(k))
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Conv1d(k,k*2,3,padding=1))
                modules.append(nn.BatchNorm1d(k*2))
                modules.append(nn.ReLU())
                modules.append(nn.AvgPool1d(2,2))
                k=k*2
        modules.append(nn.AdaptiveAvgPool1d((1)))
        
        self.features = nn.Sequential(*modules)
        #self.GP = nn.AdaptiveAvgPool1d((1))
        self.classifier = nn.Sequential(nn.Linear(k,self.num_classes))

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
