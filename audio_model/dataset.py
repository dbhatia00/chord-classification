import math
import numpy as np
import torch
from torch.utils.data import Dataset

class GuitarDataset(Dataset):
  def __init__(self, x, y, batch_size, **kwargs):
    super().__init__(**kwargs)

    self.x = x.astype(np.float32)
    self.y = y.astype(np.float32)
    self.batch_size = batch_size
  
  def __len__(self):
    return math.ceil(len(self.x))
  
  def __getitem__(self, index):
    x = self.x[index]
    # x = np.expand_dims(x, axis=0)
    y= self.y[index]

    return x, y