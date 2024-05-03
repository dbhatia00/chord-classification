from util import NUMBER_FRETS, SAMPLE_FREQ
import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
from preprocess import generate_labels
from tqdm import tqdm

# E2 to E6, the range of a standard tuned guitar with 24 frets.
note_pitches = [82.41, 87.31, 92.50, 98.00, 103.83, 110.0, 116.54, 123.47,
                130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
                261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 
                523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 
                1046.50, 1108.73, 1174.66, 1244.51, 1318.51]

fret_board_notes = np.array([
  [x for x in range(0, NUMBER_FRETS+1)],
  [x for x in range(5, 5 + NUMBER_FRETS+1)],
  [x for x in range(10, 10 + NUMBER_FRETS+1)],
  [x for x in range(15, 15 + NUMBER_FRETS+1)],
  [x for x in range(19, 19 + NUMBER_FRETS+1)],
  [x for x in range(24, 24 + NUMBER_FRETS+1)]])

# Pitches notes for open strings. First note is string 6 (lowest string).
open_pitch_notes = [40, 45, 50, 55, 59, 64]

def accuracy(preds, labels):
  accs = []
  for i, label in enumerate(labels):
    pred, num_notes = preds[i]

    # Frets that are active during this sample
    label_active_frets = (label['frets'] == 1)
    label_active_frets_sum = label_active_frets.sum()

    # Get predicted frets that match label active frets
    pred_label_frets = (pred[label_active_frets] == 1)
    pred_label_frets_sum = pred_label_frets.sum()
    print(num_notes, label_active_frets_sum)


    # print(pred.reshape(6, 23))
    # print(label['frets'].reshape(6,23))
    # input()

    if label_active_frets_sum == 0:
      accs.append(1) 
    else:
      accs.append(pred_label_frets_sum/label_active_frets_sum)
  return np.stack(accs)

def predict(labels):
  preds = []

  sample_rate, signal = scipy.io.wavfile.read(labels[0]['file'])
  for i in tqdm(range(len(labels))):
    label = labels[i]
    start_time = label['time']
    end_time = label['time'] + (1 / SAMPLE_FREQ)
    samples = signal[int(start_time * sample_rate) : int(end_time * sample_rate)]

    yf = scipy.fft.fft(samples)
    yf = np.abs(yf)
    xf = scipy.fft.fftfreq(len(samples), 1 / sample_rate)
    
    # Remove background noise and threshold related to max frequency magnitude.
    max_value = np.max(yf)
    yf = np.where(yf < 200000, 0, yf)
    yf = np.where(yf < max_value * 0.05, 0, yf)

    # Find Peaks indices
    peak_idx, props = scipy.signal.find_peaks(yf, distance=5, prominence=250000)
    freqs = np.linspace(0, sample_rate, len(yf))

    # Plot fourier transform
    # plt.plot(freqs, yf)
    # plt.title('Discrete-Fourier Transform', fontdict=dict(size=15))
    # plt.xlabel('Frequency', fontdict=dict(size=12))
    # plt.ylabel('Magnitude', fontdict=dict(size=12))
    # plt.show()

    # Get peak frequencies
    peak_freqs = freqs[peak_idx]
    # print(peak_freqs)

    # Plot fourier transform with just selected peaks
    # copy = dft.copy()
    # mask = np.zeros(dft.shape, dtype=int)
    # mask[peak_idx] = 1
    # copy[mask == 0] = 0
    # plt.plot(freqs, copy)
    # plt.title('Discrete-Fourier Transform', fontdict=dict(size=15))
    # plt.xlabel('Frequency', fontdict=dict(size=12))
    # plt.ylabel('Magnitude', fontdict=dict(size=12))
    # plt.show()

    # Discard peak frequencies outside of guitar range (F6 and D#2)
    peak_freqs = peak_freqs[(peak_freqs < 1396.91) & (peak_freqs > 77.78)]
    # print(peak_freqs)

    peak_notes = np.tile(note_pitches, (peak_freqs.shape[0], 1))
    peak_freqs = np.tile(peak_freqs, (peak_notes.shape[1], 1)).T
    notes_idx = np.argmin(np.abs(peak_notes - peak_freqs), axis=1)

    # remove any dupes (peaks too close together)
    notes_idx = np.unique(notes_idx)
    # print(notes_idx)
    
    # Predict matching fret 
    frets = np.zeros((6, NUMBER_FRETS+1))
    for note in notes_idx:
      frets[fret_board_notes == note] = 1

    # TODO Predict potential slides

    # print(label['frets'].reshape(6, NUMBER_FRETS+1))
    # plt.plot(xf_copy, yf_copy)
    # plt.xlim([0, 500])
    # plt.show()

    preds.append((frets.reshape(-1), notes_idx.shape[0]))

  return preds

def load_samples():
  signal, sample_rate = librosa.load('data/audio_mono-mic/00_BN1-129-Eb_comp_mic.wav')

  start_time = 5
  end_time = 5 + (1 / SAMPLE_FREQ)
  samples = signal[int(start_time * sample_rate) : int(end_time * sample_rate)]
  return samples, sample_rate

def main():
  labels = generate_labels()
  pred = predict(labels)
  acc = accuracy(pred, labels)
  print(acc)
  print('Average accuracy: ' + str(np.mean(acc).item()))

if __name__ == '__main__':
  main()