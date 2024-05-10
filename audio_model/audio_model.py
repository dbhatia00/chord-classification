import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(300, 300)
        # self.fc2 = nn.Linear(300, 300)
        # self.fc3 = nn.Linear(300, 300)
        # self.fc4 = nn.Linear(300, 150)
        # self.fc5 = nn.Linear(150, 49)
        # self.fc1 = nn.Linear(450, 450)
        # self.fc2 = nn.Linear(450, 300)
        # self.fc3 = nn.Linear(300, 300)
        # self.fc4 = nn.Linear(300, 150)
        # self.fc5 = nn.Linear(150, 49)
        # self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(450, 450)
        self.fc2 = nn.Linear(450, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 150)
        self.fc5 = nn.Linear(150, 49)
        self.dropout = nn.Dropout(0.2)
        # self.fc1 = nn.Linear(750, 500)
        # self.fc2 = nn.Linear(500, 250)
        # self.fc3 = nn.Linear(250, 49)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = self.pool(F.relu(self.conv5(x)))
        # x = self.pool(F.relu(self.conv6(x)))
        # x = self.pool(F.relu(self.conv7(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc5(x))
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = F.sigmoid(self.fc3(x))
        return x
