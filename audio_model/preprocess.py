import json
import os
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
from util import generate_dft

SAMPLE_FREQ = 2.0
NUMBER_FRETS = 22

# MIDI notes for open strings. First note is string 6 (lowest string).
open_midi_notes = [40, 45, 50, 55, 59, 64]

def get_fret(string, midi_note):
  index = -string + 6
  return midi_note - open_midi_notes[index]

#TODO Loads audio for given labels.
def load_samples(labels):
  for label in labels:
    signal, sample_rate = librosa.load(label['file'])

    start_time = label['time']
    end_time = label['time'] + (1 / SAMPLE_FREQ)
    samples = signal[int(start_time * sample_rate) : int(end_time * sample_rate)]

    # Simple DFT or spectrogram? Spectrogram seems more suited for audio that changes over the frame (e.g. words). 
    fft = generate_dft(samples)
    print(fft.shape)

# Labels include file, timestamp, and strings. 
# Strings is an array of six fret values. [0] is string 6.
def generate_labels():
  annotation_dir = 'data/annotation/'
  data_dir = 'data/audio_mono-mic/'
  labels = []

  for filename in os.listdir(annotation_dir):
    filepath = os.path.join(annotation_dir, filename)
    print('Generating labels for filepath')

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

      # Approach 1: Concatenate notes into sequence, then iterate through and select samples.
      # notes.sort(key = lambda x: x['time'])

      # # Go through notes played and select samples.
      # # Multi-hot labels with 6 * 24 frets.
      # frets_playing = np.zeros(6*24)
      # for n in notes:

      # Approach 2: Discretize clip into half-second long segments.
      # Labels are multi-hot labels with 6*22 frets, starting with lowest string (sixth string).
      # TODO Verify all math.
      segment_frets = np.zeros((math.floor(clip_length * SAMPLE_FREQ), 6 * (NUMBER_FRETS+1)))
      for note in notes:
        fret = np.round(note['fret'])
        fret_index = int((-note['string'] + 6) * (NUMBER_FRETS+1) + fret)
        begin_time_index = math.floor(note['time'] * SAMPLE_FREQ)
        end_time_index = math.floor((note['time'] + note['duration']) * SAMPLE_FREQ)
        segment_frets[begin_time_index:(end_time_index+1), fret_index] = 1
        
      for i in range(segment_frets.shape[0]):
        labels.append({
          'file': data_filepath,
          'time': i / SAMPLE_FREQ,
          'frets': segment_frets[i]
        })

    break

  return  labels

def main():
  labels = generate_labels()
  samples = load_samples(labels)

if __name__ == '__main__':
  main()
