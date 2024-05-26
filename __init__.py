import os
import torch
import pandas as pd
from torch import optim
from MyDataset import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from ResNet import ResNet


# 以下两个函数已经过检验，图片地址与标签一一对应，并无差错
def get_dictionary(csv_path):
    content = pd.read_csv(csv_path)
    breed = content["breed"]
    kind = list(sorted(set(breed)))
    # 构建标签映射字典
    dictionary = {}
    index = 0
    for i in kind:
        dictionary[i] = index
        index += 1
    return dictionary


def getdata(data_path):
    # 读取训练集文件
    dict_path = os.path.join(data_path, "labels.csv")
    img_root = os.path.join(data_path, "train")
    content = pd.read_csv(dict_path)
    image_id = content["id"]
    breed = content["breed"]
    # 构建标签映射字典
    dictionary = get_dictionary(dict_path)
    # 设置文件路径和对应的标签
    labels = []
    image_names = []
    for i in range(len(content)):
        file_path = image_id[i] + ".jpg"
        labels.append(dictionary[breed[i]])
        image_names.append(os.path.join(img_root, file_path))
    return image_names, labels
    # elif mode == "dev":
    #     img_root = os.path.join(data_path, mode)
    #     image_names = []
    #     for root, sub_folder, file_list in os.walk(img_root):
    #         # 每张图片的地址的数组
    #         image_names += [os.path.join(img_root, file_path) for file_path in file_list]
    #     self.images = image_names


if __name__ == "__main__":
    # 一些变量
    device = ""
    lr = 0.01  # 学习率
    pre_epoch = 0  # 前一次训练的轮数
    batch_size = 512  # batch中的数据量
    writer = SummaryWriter("log")
    module = input("请输入训练模式：")
    epochs = int(input("请输入训练轮数："))

    if torch.cuda.is_available():  # 训练处理器
        device = "cuda"
    else:
        device = "cpu"

    # 获取训练数据集
    images, labels = getdata("./data")
    train_dataset = MyDataset(images, labels, "train")
    test_dataset = MyDataset(images, labels, "test")
    train_loader = DataLoader(train_dataset, batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size, drop_last=False)
    # 加载模型
    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()  # 设置误差函数
    params = filter(lambda p: p.requires_grad, model.parameters())  # 设置模型参数跟踪
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)  # 优化器
    try:
        position = torch.load("./model.pt", map_location=device)
        model.load_state_dict(position["model"])
        pre_epoch = position["epoch"]
        optimizer.load_state_dict(position["optimizer"])
    except FileNotFoundError:
        print("Not download model!")

    if module == "train":
        model.train()  # 进入训练模式
    elif module == "test":
        model.eval()  # 进入测试模式

    # 开始训练
    for epoch in range(epochs):
        count = 0
        correct = 0
        for x, y in train_loader:
            count += len(y)
            pred = model(x.to(device))
            optimizer.zero_grad()
            loss = criterion(pred, y.to(device))
            if module == "train":
                loss.backward()
                optimizer.step()  # 参数修改
            label = pred.argmax(1)
            for i in range(len(y)):
                if y[i] == label[i]:
                    correct += 1
        writer.add_scalar("Accuracy/Train", correct / count, epoch)  # 用于tensorboard的数据写入
        print("Current epoch is :", epoch + pre_epoch + 1, " Accuracy is :", correct / count)
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1 + pre_epoch}
        if module == "train":
            torch.save(state_dict, "./model.pt")
    print("Finished!!!")
    writer.close()
