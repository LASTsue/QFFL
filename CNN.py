import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,64,3,1)
        self.conv2=nn.Conv2d(64,128,3,1)
        self.Maxpool2d=nn.MaxPool2d(2,2)
        self.linear1=nn.Linear(12*12*128,1024)
        self.dropout=nn.Dropout(0.5)
        self.linear2=nn.Linear(1024,10)

    def forward(self,x):
        x=self.conv1(x)
        x=nn.ReLU()(x)
        x=self.conv2(x)
        x=nn.ReLU()(x)
        x=self.Maxpool2d(x)
        x=torch.flatten(x,1)
        x=self.linear1(x)
        x=nn.ReLU()(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x