from util import generate_dft
import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
from preprocess import generate_labels

# E2 to E6, the range of a standard tuned guitar with 24 frets.
note_pitches = [82.41, 87.31, 92.50, 98.00, 103.83, 110.0, 116.54, 123.47,
                130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
                261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 
                523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 
                1046.50, 1108.73, 1174.66, 1244.51, 1318.51]

SAMPLE_FREQ = 2.0

# Pitches notes for open strings. First note is string 6 (lowest string).
open_pitch_notes = [40, 45, 50, 55, 59, 64]

def predict(sample, sample_rate):
  dft = generate_dft(sample)
  
  # Threshold
  max_value = np.max(dft)
  dft = np.where(dft < max_value * 0.2, 0, dft)

  # Select Peaks
  peak_idx, _ = scipy.signal.find_peaks(dft, distance=4)
  freqs = np.linspace(0, sample_rate, len(dft))

  # plt.plot(freqs, dft)
  # plt.title('Discrete-Fourier Transform', fontdict=dict(size=15))
  # plt.xlabel('Frequency', fontdict=dict(size=12))
  # plt.ylabel('Magnitude', fontdict=dict(size=12))
  # plt.show()

  peak_freqs = freqs[peak_idx]

  # copy = dft.copy()
  # mask = np.zeros(dft.shape, dtype=int)
  # mask[peak_idx] = 1
  # copy[mask == 0] = 0
  # plt.plot(freqs, copy)
  # plt.title('Discrete-Fourier Transform', fontdict=dict(size=15))
  # plt.xlabel('Frequency', fontdict=dict(size=12))
  # plt.ylabel('Magnitude', fontdict=dict(size=12))
  # plt.show()

  peak_notes = np.tile(note_pitches, (peak_freqs.shape[0], 1))
  peak_freqs = np.tile(peak_freqs, (peak_notes.shape[1], 1)).T
  notes_idx = np.argmin(np.abs(peak_notes - peak_freqs), axis=1)

  # remove any dupes
  notes_idx = np.unique(notes_idx)
  print(notes_idx)
  
  # Predict matching frets

  # Predict potential slides

  # Softmax

def load_samples():
  signal, sample_rate = librosa.load('data/audio_mono-mic/00_BN1-129-Eb_comp_mic.wav')

  start_time = 5
  end_time = 5 + (1 / SAMPLE_FREQ)
  samples = signal[int(start_time * sample_rate) : int(end_time * sample_rate)]
  return samples, sample_rate

def main():
  samples, sample_rate = load_samples()
  pred = predict(samples, sample_rate)

if __name__ == '__main__':
  main()