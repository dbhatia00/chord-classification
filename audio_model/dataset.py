import math
import tensorflow as tf
import numpy as np

class GuitarDataset(tf.keras.utils.PyDataset):
  def __init__(self, x, y, batch_size, **kwargs):
    super().__init__(**kwargs)

    idx = np.random.permutation(x.shape[0])
    self.x = x[idx]
    self.y = y[idx]
    self.batch_size = batch_size
  
  def __len__(self):
    return math.ceil(len(self.x) / self.batch_size)
  
  def __getitem__(self, index):
    start = index * self.batch_size
    end = min(start + self.batch_size, len(self.x))
    batch_x = self.x[start:end]
    batch_x = np.expand_dims(batch_x, axis=2)
    batch_y = self.y[start:end]

    return batch_x, batch_y