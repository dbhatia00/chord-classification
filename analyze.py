import numpy as np
import os
from audio_model.util import note_strings
from video_model.util import guitar_notes_dict, chords
import matplotlib.pyplot as plt

# Function to calculate Euclidean distance between two sets of notes
def euclidean_distance(notes1, notes2):
    note_indices = {note: i for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}
    vec1 = np.zeros(12)
    vec2 = np.zeros(12)
    for note in notes1:
        vec1[note_indices[note[:-1]]] = 1
    for note in notes2:
        vec2[note_indices[note[:-1]]] = 1
    return np.linalg.norm(vec1 - vec2)

# Function to find the closest chord to a given set of notes
def find_closest_chord(notes):
    min_distance = float('inf')
    closest_chord = None
    for chord, chord_notes in chords.items():
        distance = euclidean_distance(notes, chord_notes)
        if distance < min_distance:
            min_distance = distance
            closest_chord = chord
    return closest_chord

def analyzeProbs(audioTimes, audioProbs, videoTimes, videoProbs, hash):
    # Determine the minimum number of rows
    min_rows = min(len(audioProbs), len(videoProbs))
    
    # Trim the lists to have the same number of rows
    audioProbs = audioProbs[:min_rows]
    videoProbs = videoProbs[:min_rows]
    audioTimes = audioTimes[:min_rows]
    videoTimes = videoTimes[:min_rows]


    # Convert lists to numpy arrays
    # Element-wise multiplication
    resultProbs = np.array(audioProbs) * np.array(videoProbs)

    notes = []
    for p in resultProbs:
        note = np.where(p >= 0.20)
        notes.append(note[0].tolist())
    
    notes = [[note_strings[idx] for idx in indices] for indices in notes]
    print(notes)
    #plotResults(audioTimes, notes)

    # Calculate the closest chords
    closest_chords = []
    for i in notes:
        if (len(i) >= 2):
            closest = find_closest_chord(i)
            closest_chords.append([closest])
        else:
            closest_chords.append([])
    print(closest_chords)
    notes_out = f"results/FINAL-{hash}.txt"

    # Write tensor to file
    with open(notes_out, 'w') as file:
        for timestamp, probabilities in zip(audioTimes, notes):
            file.write(f"{timestamp:.3f}: {list(probabilities)}\n")

    return resultProbs
