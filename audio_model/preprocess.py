import json
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
import math
from util import NUMBER_FRETS, SAMPLE_FREQ
import pickle
from tqdm import tqdm
from dataset import GuitarDataset
import math
import torch

# MIDI notes for open strings. First note is string 6 (lowest string).
open_midi_notes = [40, 45, 50, 55, 59, 64]

def get_fret(string, midi_note):
  index = -string + 6
  return round(midi_note - open_midi_notes[index])

def load_sample(label):
  sample_rate, signal = scipy.io.wavfile.read(label['file'])

  start_time = label['time']
  end_time = label['time'] + (1 / SAMPLE_FREQ)
  samples = signal[int(start_time * sample_rate) : int(end_time * sample_rate)]

  fft = scipy.fft.rfft(samples)
  fft = np.abs(fft)
  xf = scipy.fft.rfftfreq(len(samples), 1 / sample_rate)
  return fft

# Generates ffts for given labels.
def load_samples(labels):
  # Calculate size of array based on first sample.
  fft = load_sample(labels[0])
  sample_ffts = np.zeros((len(labels), fft.shape[0]))
  sample_ffts = []

  print("Generating FFTs...")
  for i in tqdm(range(len(labels))):
    fft = load_sample(labels[i])
    sample_ffts.append(fft)
  
  return np.stack(sample_ffts)

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
      for note in notes:
        fret = np.round(note['fret'])
        fret_index = int((-note['string'] + 6) * (NUMBER_FRETS+1) + fret)
        begin_time_index = math.floor(note['time'] * SAMPLE_FREQ)
        end_time_index = math.ceil((note['time'] + note['duration']) * SAMPLE_FREQ)
        segment_frets[begin_time_index:(end_time_index), fret_index] = 1
        segment_notes[begin_time_index:(end_time_index), round(note['value'] - 40)] = 1
      
      for i in range(segment_frets.shape[0]):
        labels.append({
          'file': data_filepath,
          'time': i / SAMPLE_FREQ,
          'frets': segment_frets[i],
          'notes': segment_notes[i]
        })

  return  labels

def get_data(batch_size, test_split):
  if os.path.isfile('data.pkl'):
    with open('data.pkl', 'rb') as data:
      x, y = pickle.load(data)
  else:
    labels = generate_labels()
    samples = load_samples(labels)

    print("Saving preprocessed data...")
    with open('data.pkl', 'wb') as out:
      target = []
      for label in labels:
        target.append(label['notes'])
      pickle.dump((samples, target), out, pickle.HIGHEST_PROTOCOL)

    x, y = (samples, target)
    
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
  labels = generate_labels()
  samples = load_samples(labels)

  print("Saving preprocessed data...")
  with open('data.pkl', 'wb') as out:
    target = []
    for label in labels:
      target.append(label['notes'])
    target = np.stack(target)
    pickle.dump((samples, target), out, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  main()
