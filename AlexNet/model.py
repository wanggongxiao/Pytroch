import torch
import torch.nn as nn
from torch.nn import AvgPool2d
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        r1 = nn.ReLU()
        p1 = nn.AvgPool2d(kernel_size=3,stride=2)

        c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,padding=2)
        p2 = nn.AvgPool2d(kernel_size=3, padding=2)

        c3 = nn.Conv2d(in_channels=256, out_channels=384,kernel_size=3, padding=1)

        c4 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        flat = nn.Flatten()
        
