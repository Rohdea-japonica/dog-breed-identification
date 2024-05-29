import os
import torch

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
from MyDataset import MyDataset
from torch.utils.data import DataLoader
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


def getdata(data_path, module):
    if module == "train":
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
    elif module == "test":
        img_root = os.path.join(data_path, "test")
        image_names = []
        for root, sub_folder, file_list in os.walk(img_root):
            image_names += [os.path.join(root, file_path) for file_path in file_list]
        return image_names


if __name__ == "__main__":
    # 一些变量
    device = ""
    lr = 0.01  # 学习率
    pre_epoch = 0  # 前一次训练的轮数
    batch_size = 512  # batch中的数据量
    module = input("请输入训练模式：")

    if torch.cuda.is_available():  # 训练处理器
        device = "cuda"
    else:
        device = "cpu"

    # 获取训练数据集
    images, labels = getdata("./data", module)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, train_size=0.75,
                                                        shuffle=True, random_state=12)
    train_dataset = MyDataset(x_train, y_train, "train")
    dev_dataset = MyDataset(x_test, y_test, "dev")
    train_loader = DataLoader(train_dataset, batch_size, drop_last=False)
    dev_loader = DataLoader(dev_dataset, batch_size, drop_last=False)
    # 获取预测数据集
    test_images = getdata("./data", module)
    test_dataset = MyDataset(test_images, "", "test")
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
        epochs = int(input("请输入训练轮数："))
        # 开始训练
        for epoch in range(epochs):
            count = 0
            correct = 0
            for x, y in train_loader:
                count += len(y)
                # 前向传播
                pred = model(x.to(device))
                loss = criterion(pred, y.to(device))
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # 参数修改
                label = pred.argmax(1)
                for i in range(len(y)):
                    if y[i] == label[i]:
                        correct += 1
            print("Current epoch is :", epoch + pre_epoch + 1, " Train_Accuracy is :", correct / count)
            state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": epoch + 1 + pre_epoch}
            torch.save(state_dict, "./model.pt")
        print("Finished!!!")

        with torch.no_grad():
            model.eval()  # 进入测试模式
            count = 0
            correct = 0
            for x, y in dev_loader:
                count += len(y)
                pred = model(x.to(device))
                label = pred.argmax(1)
                for i in range(len(y)):
                    if y[i] == label[i]:
                        correct += 1
            print(" Accuracy is :", correct / count)
    elif module == "test":
        labls = []
        with torch.no_grad():
            model.eval()  # 进入测试模式
            count = 0
            correct = 0
            for x in test_loader:
                pred = model(x.to(device))
                pred = nn.functional.softmax(pred)
                labels += pred.tolist()
        image_names = []
        for root, sub_folder, file_list in os.walk("./data/test"):
            image_names += [file_path for file_path in file_list]
        dictionary = get_dictionary("./data/labels.csv")
        dict_key = list(dictionary.keys())
        # 循环进入所有的标签概率列表
        num = 0
        for label in labels:
            # 每个列表的前面加上图片名
            para = {"id": image_names[num]}
            for idx in range(len(label)):
                para[dict_key[idx]] = label[idx]
            df = pd.DataFrame([para])
            num += 1
            if num == 0:
                df.to_csv("test_dict.csv", mode='a', header=True)
            else:
                df.to_csv("test_dict.csv", mode='a', header=False)