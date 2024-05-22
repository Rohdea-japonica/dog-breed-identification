import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

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

class MyDataset(Dataset):
    def __init__(self, mode):
        self.images = []
        self.labels = []
        if mode == 'train':  # 80%
            self.images = self.images[:int(0.9 * len(self.images))]
            self.labels = self.labels[:int(0.9 * len(self.labels))]
        elif mode == 'test':  # 20%
            self.images = self.images[int(0.9 * len(self.images)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = Image.open(img).convert("RGB")
        # ImageNet推荐的RGB通道均值、方差
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((500, 500)), transforms.Normalize(mean=mean, std=std)])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

    def getdata(self, data_path, mode):
        if mode == "train":
            # 读取训练集文件
            dict_path = os.path.join(data_path, "labels.csv")
            img_root = os.path.join(data_path, mode)
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
            print(dictionary)
            self.images = image_names
            self.labels = labels
        elif mode == "dev":
            img_root = os.path.join(data_path, mode)
            image_names = []
            for root, sub_folder, file_list in os.walk(img_root):
                # 每张图片的地址的数组
                image_names += [os.path.join(img_root, file_path) for file_path in file_list]
            self.images = image_names
