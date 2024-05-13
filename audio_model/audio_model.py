import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
          nn.Linear(450, 450),
          nn.BatchNorm1d(450),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(450, 300),
          nn.BatchNorm1d(300),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(300, 300),
          nn.BatchNorm1d(300),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(300, 150),
          nn.BatchNorm1d(150),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(150, 49)
        )

    def forward(self, x):
        x = self.sequential(x)
        x = F.sigmoid(x)
        return x
