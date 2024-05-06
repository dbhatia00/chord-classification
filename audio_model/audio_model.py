import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv1d(1, 80, 41)
        # self.pool = nn.MaxPool1d(3)
        # self.conv2 = nn.Conv1d(80, 160, 16)
        # self.conv3 = nn.Conv1d(160, 320, 9)
        # self.conv4 = nn.Conv1d(320, 320, 9)
        # self.conv5 = nn.Conv1d(320, 320, 9)
        # self.conv6 = nn.Conv1d(320, 320, 9)
        # self.conv7 = nn.Conv1d(320, 320, 9)
        # self.fc1 = nn.Linear(5440, 720)
        # self.fc2 = nn.Linear(720, 120)
        # self.fc3 = nn.Linear(120, 49)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(750, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 49)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = F.relu(self.conv2(x))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = F.relu(self.conv5(x))
        # x = self.pool(F.relu(self.conv6(x)))
        # x = self.pool(F.relu(self.conv7(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = self.dropout(x)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x
