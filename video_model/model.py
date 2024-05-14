import torch.nn as nn
import torch

class GuitarTabCNN(nn.Module):
    def __init__(self):
        super(GuitarTabCNN, self).__init__()
        self.sequential = nn.Sequential(
          nn.Conv2d(1, 64, kernel_size=11, stride=3),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
          nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=2),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
          nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(384),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
          nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(384),
          nn.ReLU(),
          nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(384),
          nn.ReLU(),
          nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
          nn.Flatten(),
          nn.Linear(5120, 2000),
          nn.BatchNorm1d(2000),
          nn.ReLU(),
          nn.Linear(2000, 1000),
          nn.BatchNorm1d(1000),
          nn.ReLU(),
          nn.Linear(1000, 1000),
          nn.BatchNorm1d(1000),
          nn.ReLU(),
          nn.Linear(1000, 6 * 21)
        )
        

    def forward(self, x):
        x = self.sequential(x)

        # Softmax each set of 21 fret logits. This represents the probability that a particular fret is pressed for that string.
        x = torch.nn.functional.softmax(x.view(-1, 21), dim=1).view(-1, 6*21)
        return x