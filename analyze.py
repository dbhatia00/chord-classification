import numpy as np
import os
from audio_model.util import note_strings
from video_model.util import guitar_notes_dict, chords
import matplotlib.pyplot as plt

def print_fretboard(chord):
    # Define the fretboard as a list of lists
    fretboard = [['-' for _ in range(6)] for _ in range(5)]  # 6 strings, 5 frets

    # Define the frets where fingers are placed for the given chord
    chord_fingers = {
        'E_major': [0, 2, 2, 1, 0, 0],
        'E_minor': [0, 2, 2, 0, 0, 0],
        'A_major': [0, 0, 2, 2, 2, 0],
        'D_major': [0, 0, 0, 2, 3, 2],
        'G_major': [3, 2, 0, 0, 0, 3],
        'C_major': [0, 3, 2, 0, 1, 0],
        'F_major': [0, 2, 2, 1, 0, 0],
        'B_major': [0, 1, 3, 3, 3, 1],
        'D_minor': [0, 0, 0, 2, 3, 1],
        'E7': [0, 2, 0, 1, 0 ,0]
    }

    # Update the fretboard with the fingers for the given chord
    for string, fingers in enumerate(chord_fingers[chord]):
        fretboard[fingers][string] = 'O'  # Place finger on the fret

    # Print the fretboard
    for fret in range(4, -1, -1):  # Print frets from 4 to 0
        print('|', end='')
        for string in range(6):
            print(fretboard[fret][string], end='|')
        print()

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


    # NOTE: One of these probs should be videoProbs
    # Replaced for demo purposes
    '''
    resultProbs = np.array(audioProbs) * np.array(audioProbs)
    #print(resultProbs)
    notes = []
    for p in resultProbs:
        note = np.where(p >= 0.95)
        notes.append(note[0].tolist())
    '''
    resultProbs = np.array(audioProbs) * np.array(videoProbs)
    #print(resultProbs)
    notes = []
    for p in resultProbs:
        note = np.where(p >= 0.35)
        notes.append(note[0].tolist())

    notes = [[note_strings[idx] for idx in indices] for indices in notes]
    #print(notes)
    print()
    # Calculate the closest chords
    closest_chords = []
    for i in notes:
        if (len(i) >= 2):
            closest = find_closest_chord(i)
            closest_chords.append([closest])
        else:
            closest_chords.append([])

    notes_out = f"results/FINAL-{hash}.txt"

    # Write tensor to file
    with open(notes_out, 'w') as file:
        for timestamp, probabilities in zip(audioTimes, resultProbs):
            file.write(f"{timestamp:.3f}: {list(probabilities)}\n")

    # Flatten the list of lists
    unique_chords = [chord for chord in closest_chords if chord]  # Remove empty lists
    unique_chords_no_duplicates = remove_adjacent_duplicates(unique_chords)
    return unique_chords_no_duplicates

def remove_adjacent_duplicates(lst):
    result = []
    prev = None
    for item in lst:
        if item != prev:
            result.append(item)
            prev = item
    return result
