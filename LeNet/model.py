import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()

        # 创建网络
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5,padding=2)
        self.sig = nn.Sigmoid()
        self.bool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,padding=0)
        self.bool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()

        # 全连接层
        self.f1 = nn.Linear(400,120)
        self.f2 = nn.Linear(120,84)
        self.f3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.sig(self.c1(x))
        x = self.bool1(x)

        x = self.sig(self.c2(x))
        x = self.bool2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model,(1,28,28)))