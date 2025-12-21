import copy
import time

from anaconda_navigator.utils.download_manager import Download

from model import LeNet
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import torch.nn as nn


def train_vai_data_load():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    # 随机分配训练集和验证集
    train_data,val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_data = Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_data   = Data.DataLoader(val_data,batch_size=32,shuffle=True,num_workers=2)

    return train_data, val_data

def train_model_process(model, train_data, val_data,num_eporchs):
    # 设定训练所用到的设备，有GPU用GPU,没有则用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
    # 设置优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.paramters(), lr=0.001)
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确率
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集的准确度
    train_acc_all = []
    # 验证集的准确度
    val_acc_all = []
    #当前时间
    since = time.time()
    for eporch in num_eporchs:
        print("Epoch {}/{}".format(eporch,num_eporchs))
        print("-"*10)


if __name__ == "__main__":
    # 加载需要的模型
    LeNet = LeNet()
    # 加载数据
    train_data, val_data = train_vai_data_load()
    # 利用现有的数据进行模型训练

