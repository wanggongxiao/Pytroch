import copy
import time

from model import LeNet
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


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

        # 初始化参数
        # 训练集的损失函数1
        train_loss = 0.0
        # 训练集的准确度
        train_acc = 0.0
        # 验证集的损失值
        val_loss= 0.0
        # 验证集的准确度
        val_acc = 0.0
        # 训练集的样本数
        train_num = 0;
        # 验证集的样本数
        val_num = 0
        # 对每一个mini_batch训练和计算
        for step, (b_x,b_y) in enumerate(train_data):
            # 将数据放到设备当中
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置为训练模式
            model.train()

            # 向前传播过程中，输入为一个1batch,输出为对应的batch对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batcg的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 跟新参数
            optimizer.step()
            train_acc += torch.sum(pre_lab == b_y.data)
            train_loss += loss.item() * b_x.size(0)

            # 当前样本用于训练的样本数
            train_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc.double().item()/train_num)

        # 计算并保存验收集的loss值
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_acc.double().item() / val_num)


        print("{} train loss:{:.4f} train acc: {:.4f}".format(eporch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(eporch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数，保存最优参数的模型
    torch.save(best_model_wts, "./best_model.pth")

    train_process = pd.DataFrame(data={"epoch": range(num_eporchs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    # 加载需要的模型
    LeNet = LeNet()
    # 加载数据
    train_data, val_data = train_vai_data_load()
    # 利用现有的数据进行模型训练
    train_process = train_model_process(LeNet,train_data,val_data,num_eporchs=20)
    matplot_acc_loss(train_process)

