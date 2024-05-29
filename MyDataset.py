import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, images, labels, mode):
        self.images = images
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "dev":
            img, label = self.images[index], self.labels[index]
            img = Image.open(img).convert("RGB")
            # ImageNet推荐的RGB通道均值、方差
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((200, 200), antialias=True),
                                     transforms.Normalize(mean=mean, std=std)])
            img = tf(img)
            label = torch.tensor(label)
            return img, label
        elif self.mode == "test":
            img = self.images[index]
            img = Image.open(img).convert("RGB")
            # ImageNet推荐的RGB通道均值、方差
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True),
                                     transforms.Normalize(mean=mean, std=std)])
            img = tf(img)
            return img
