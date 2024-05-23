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


print(get_dictionary("./data/labels.csv"))

images, labels = getdata("./data")
print("images' lengths is :", len(images), "  labels‘ length is :", len(labels))
for i in range(len(images)):
    print(images[i], labels[i])
