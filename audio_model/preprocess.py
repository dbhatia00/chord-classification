import json
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
import math
from util import NUMBER_FRETS, SAMPLE_FREQ, WINDOW_SIZE, mel_filter_bank
import pickle
from tqdm import tqdm
from dataset import GuitarDataset
import math
import torch
import librosa


# MIDI notes for open strings. First note is string 6 (lowest string).
open_midi_notes = [40, 45, 50, 55, 59, 64]

def get_fret(string, midi_note):
  index = -string + 6
  return round(midi_note - open_midi_notes[index])

# Kind of a hack
mel_filter = None

def get_time_range(time):
  start_time = time
  end_time = time + (1.0 / SAMPLE_FREQ)
  return (start_time, end_time)

def load_sample(signal, sample_rate, time_range):
  samples = signal[int(time_range[0]* sample_rate) : int(time_range[1]* sample_rate)]
  if samples.size % 2 == 1:
    samples = samples[:-1]

  num_coefficients = 150 # Size of mfcc array
  min_hz = 0
  max_hz = 3000

  complex_spectrum = np.fft.fft(samples)
  # xf = scipy.fft.fftfreq(len(samples), 1 / sample_rate)
  power_spectrum = abs(complex_spectrum) ** 2
  filtered_spectrum = np.where(power_spectrum == 0.0, 1e-20, power_spectrum)
  global mel_filter
  if mel_filter is None:
    mel_filter = mel_filter_bank(power_spectrum.shape[0], num_coefficients, min_hz, max_hz)
  filtered_spectrum = np.dot(power_spectrum, mel_filter)
  filtered_spectrum = np.where(filtered_spectrum == 0.0, 1e-10, filtered_spectrum)
  log_spectrum = np.log(filtered_spectrum)
  dct_spectrum = scipy.fft.dct(log_spectrum, type=2) # MFCC
  return dct_spectrum 

# Generates mfccs for given labels.
def load_samples(labels):
  print("Generating MFCCs...")
  mfccs = []
  processed_labels = []
  mfcc_pairs = []
  label_pairs = []
  curr_file = None
  sample_rate = signal = max = None

  for i in tqdm(range(len(labels))):
    if labels[i]['file'] != curr_file:
      # Generate pairs
      if i != 0:
        mfccs = np.stack(mfccs)
        window_arrays = []
        for j in range(WINDOW_SIZE):
          end_index = -(WINDOW_SIZE-j-1)
          if end_index == 0:
            end_index = mfccs.shape[0]
          window_arrays.append(mfccs[j:end_index])
        mfccs = np.hstack(window_arrays)
        mfcc_pairs.append(mfccs)
        label_pairs.extend(processed_labels[WINDOW_SIZE//2:-WINDOW_SIZE//2+1])
        mfccs = []
        processed_labels = []

      # Normalize audio
      sample_rate, signal = scipy.io.wavfile.read(labels[i]['file'])
      signal = signal.astype(np.float32)
      max = np.max(signal)
      signal = signal / max
      curr_file = labels[i]['file']

    mfcc = load_sample(signal, sample_rate, get_time_range(labels[i]['time']))
    mfccs.append(mfcc)
    processed_labels.append(labels[i])
  
  return np.vstack(mfcc_pairs), label_pairs

def load_samples_from_file(filename):
  sample_rate, signal = scipy.io.wavfile.read(filename)
  signal = signal.astype(np.float32)
  max = np.max(signal)

  mfccs = []

  print("Generating MFCCs...")
  mfccs = []

  file_duration = int(signal.shape[0] / sample_rate)
  num_clips = file_duration * SAMPLE_FREQ

  sample_rate, signal = scipy.io.wavfile.read(filename)
  signal = signal.astype(np.float32)
  if len(signal.shape) > 1:
    signal = signal[:, 0]
  max = np.max(signal)
  signal = signal / max

  for i in tqdm(range(num_clips)):
    time = float(i) / SAMPLE_FREQ
    mfcc = load_sample(signal, sample_rate, get_time_range(time))
    mfccs.append(mfcc)
  
  mfccs = np.stack(mfccs)
  window_arrays = []
  for j in range(WINDOW_SIZE):
    end_index = -(WINDOW_SIZE-j-1)
    if end_index == 0:
      end_index = mfccs.shape[0]
    window_arrays.append(mfccs[j:end_index])
  mfccs = np.hstack(window_arrays)
  return mfccs

# Labels include file, timestamp, and strings. 
# Strings is an array of six fret values. [0] is string 6.
def generate_labels():
  annotation_dir = 'data/annotation/'
  data_dir = 'data/audio_mono-mic/'
  labels = []

  for filename in os.listdir(annotation_dir):
    filepath = os.path.join(annotation_dir, filename)
    print(f'Generating labels for {filename}')

    # Each annotation file corresponds to a wav file.
    data_filepath = os.path.join(data_dir, filename.replace('.jams', '_mic.wav'))

    with open(filepath) as file:
      jams = json.load(file)

      clip_length = jams['file_metadata']['duration']

      s6_midi = jams['annotations'][1]['data']
      s5_midi = jams['annotations'][3]['data']
      s4_midi = jams['annotations'][5]['data']
      s3_midi = jams['annotations'][7]['data']
      s2_midi = jams['annotations'][9]['data']
      s1_midi = jams['annotations'][11]['data']

      for note in s6_midi:
        note['string'] = 6
        note['fret'] = get_fret(6, note['value'])
      for note in s5_midi:
        note['string'] = 5
        note['fret'] = get_fret(5, note['value'])
      for note in s4_midi:
        note['string'] = 4
        note['fret'] = get_fret(4, note['value'])
      for note in s3_midi:
        note['string'] = 3
        note['fret'] = get_fret(3, note['value'])
      for note in s2_midi:
        note['string'] = 2
        note['fret'] = get_fret(2, note['value'])
      for note in s1_midi:
        note['string'] = 1
        note['fret'] = get_fret(1, note['value'])
    
      notes = s6_midi + s5_midi + s4_midi + s3_midi + s2_midi + s1_midi

      # Discretize clip into short segments.
      # Labels are multi-hot labels with 6*22 frets, starting with lowest string (sixth string).
      segment_frets = np.zeros((math.floor(clip_length * SAMPLE_FREQ), 6 * (NUMBER_FRETS+1)))
      segment_notes = np.zeros((math.floor(clip_length * SAMPLE_FREQ), 49)) # Guitars can play 49 notes
      valid = np.ones((math.floor(clip_length*SAMPLE_FREQ)))
      for note in notes:
        fret = np.round(note['fret'])
        fret_index = int((-note['string'] + 6) * (NUMBER_FRETS+1) + fret)
        begin_time_index = math.floor(note['time'] * SAMPLE_FREQ)
        end_time_index = math.ceil((note['time'] + note['duration']) * SAMPLE_FREQ)
        segment_frets[begin_time_index:(end_time_index), fret_index] = 1
        segment_notes[begin_time_index:(end_time_index), round(note['value'] - 40)] = 1

        # Segments where a note starts or ends are invalid
        if begin_time_index < valid.shape[0]:
          valid[begin_time_index] = 0
        if (end_time_index < valid.shape[0]):
          valid[end_time_index] = 0
      
      for i in range(segment_frets.shape[0]):
        if valid[i] == 1:
          labels.append({
            'file': data_filepath,
            'time': i / SAMPLE_FREQ,
            'frets': segment_frets[i],
            'notes': segment_notes[i]
          })

  return  labels

def generate_data():
    labels = generate_labels()
    samples, labels = load_samples(labels)

    print("Saving preprocessed data...")
    with open('data.pkl', 'wb') as out:
      target = []
      for label in labels:
        target.append(label['notes'])
      target = np.stack(target)
      pickle.dump((samples, target), out, pickle.HIGHEST_PROTOCOL)

def get_data(batch_size, test_split):
  if not os.path.isfile('data.pkl'):
    generate_data()
    
  with open('data.pkl', 'rb') as data:
    x, y = pickle.load(data)

  # Create triplets of samples
  # x_next = x[1:]
  # y_next = y[1:]
  # x_next2 = x[2:]
  # y_next2 = y[2:]
  # x = np.hstack([x[:-2], x_next[:-1], x_next2])
  # y = y_next2
  
  split = math.floor(x.shape[0] * (1 - test_split))
  idx = np.random.permutation(x.shape[0])
  x = x[idx]
  y = y[idx]
  train_dataset = GuitarDataset(x[:split], y[:split], batch_size)
  test_dataset = GuitarDataset(x[split:], y[split:], batch_size)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader

def main():
  generate_data()

if __name__ == '__main__':
  main()
