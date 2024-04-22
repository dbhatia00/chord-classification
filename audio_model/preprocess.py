import json
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

# MIDI notes for open strings. First note is string 6 (lowest string).
open_midi_notes = [40, 45, 50, 55, 59, 64]

def get_fret(string, midi_note):
  index = -string + 6
  return midi_note - open_midi_notes[index]

# Loads audio for given labels.
def load_samples(labels):
  for label in labels:
    signal, sample_rate = librosa.load(label.file)

    plt.figure(figsize=(20, 5))
    librosa.display.waveplot(signal, sr=sample_rate)
    plt.title('Waveplot', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Amplitude', fontdict=dict(size=15))
    plt.show()

# Labels include file, timestamp, and strings. 
# Strings is an array of six fret values. [0] is string 6.
def generate_labels():
  data_dir = 'data/annotation/'
  for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    print(filepath)

    with open(filepath) as file:
      jams = json.load(file)
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
      notes.sort(key = lambda x: x['time'])

      # Go through notes played and select samples.
      # TODO How should labels be formatted? Multi-hot labels with 6 * 24 frets?
      frets_playing = np.zeros(6*24)
      for n in notes:
        
    break

def main():
  labels = generate_labels()
  # test_segment = {
  #   'file': ,
  #   'time': ,
  #   'strings': []
  # }
  samples = load_samples(labels)

if __name__ == '__main__':
  main()
