import numpy as np

def generate_dft(samples):
  fft = np.fft.fft(samples)
  fft = np.abs(fft)
  fft = fft[:int(len(fft)/2)]
  return fft
