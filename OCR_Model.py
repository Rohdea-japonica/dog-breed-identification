import torch.nn as nn


# N=(W-F+2P)/S+1
class OCR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, 4, 2, 0),  # size = 200*200*3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 0),  # size = 99*99*64
            # Conv2
            nn.Conv2d(64, 128, 3, 2, 0),  # size = 49*49*64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 0),  # size = 24*24*128
            # Conv3
            nn.Conv2d(128, 256, 3, 2, 0),  # size = 23*23*128
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 0),  # size = 11*11*256
            # Conv4
            nn.Conv2d(256, 512, 2, 1, 0),  # size = 10*10*256
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 0),  # size = 9*9*512
            # Conv5
            nn.Conv2d(512, 512, 2, 1, 0),  # size = 8*8*512
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 0),  # size = 7*7*512  ->  output = 6*6*512
        )
        self.dense = nn.Sequential(
            # FC1
            nn.Linear(18432, 9216),
            nn.ReLU(),
            nn.Dropout(0.5),
            # FC2
            nn.Linear(9216, 4108),
            nn.ReLU(),
            nn.Dropout(0.5),
            # FC3
            nn.Linear(4108, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            # FC4
            nn.Linear(2048, 120),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
