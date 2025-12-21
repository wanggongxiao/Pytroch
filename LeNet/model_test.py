import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

def test_load_process():
    train_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=26), transforms.ToTensor()]),
                              download=True)
if __name__ == "__main":
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载数据
    test_load_process()