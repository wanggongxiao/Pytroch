import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import LeNet

def test_load_process():
    train_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    test_data_load = Data.DataLoader(train_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)
    return test_data_load
def test_model_process(model, test_data_load):
    # 设定测试指定的设备，有GPU用GPU，反之用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 将模型放入到设备当中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只向前传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x,test_data_y in test_data_load:
            # 将数据放入到设备当中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 将模型设置为评估模式
            model.eval()
            # 向前传播 采集数据
            output = model(test_data_x)
            pre_lab = torch.argmax(output,dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y)
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)

if __name__ == "__main__":
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load('../LeNet/best_model.pth'))
    # 加载数据
    test_data_load = test_load_process()
    test_model_process(model,test_data_load)
