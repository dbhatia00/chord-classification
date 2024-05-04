import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 20, 11)
        self.pool = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(20, 40, 7)
        self.conv3 = nn.Conv1d(40, 60, 4)
        self.conv4 = nn.Conv1d(60, 80, 4)
        self.conv5 = nn.Conv1d(80, 80, 4)
        self.fc1 = nn.Linear(1680, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 49)
        # self.fc1 = nn.Linear(256, 256) 
        # self.fc2 = nn.Linear(256, 128) 
        # self.fc3 = nn.Linear(128, 49) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        return x
