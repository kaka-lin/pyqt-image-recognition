import torch
import torch.nn as nn
import torch.nn.functional as F

from models.binarized_utils import BinarizeLinear, BinarizeConv2d

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 25, 3)
        self.conv2 = nn.Conv2d(25, 50, 3)
        self.fc1 = nn.Linear(1250, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1250)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # raw_output
        return x


class BNN(nn.Module):
    def __init__(self):
        super(BNN, self).__init__()

        self.conv1 = BinarizeConv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.htanh1 = nn.Hardtanh()
        self.conv2 = BinarizeConv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.htanh2 = nn.Hardtanh()

        self.fc1 = BinarizeLinear(1024, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.htanh3 = nn.Hardtanh()
        self.fc2 = BinarizeLinear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.htanh1(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh2(x)

        #print(x.size())
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.htanh3(x)

        x = self.fc2(x)

        return x
