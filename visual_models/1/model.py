# Train: 41% pressed, 75% open acc. Test: 37% pressed, 74% open acc.
import torch
import torch.nn as nn
import torch.nn.functional as F

class GuitarTabCNN(nn.Module):
    def __init__(self):
        super(GuitarTabCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5120, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 6 * 21)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Softmax each set of 21 fret logits. This represents the probability that a particular fret is pressed for that string.
        x = F.softmax(x.view(-1, 21), dim=1).view(-1, 6*21)
        return x

