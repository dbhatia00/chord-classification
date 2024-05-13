import numpy as np
import math

SAMPLE_FREQ = 8 # Number of samples per second to generate.
NUMBER_FRETS = 22
WINDOW_SIZE = 3 # Number of samples to combine for each prediction. Must be odd.

# E2 to E6, the range of a standard tuned guitar with 24 frets.
note_pitches = [82.41, 87.31, 92.50, 98.00, 103.83, 110.0, 116.54, 123.47,
                130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
                261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 
                523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 
                1046.50, 1108.73, 1174.66, 1244.51, 1318.51]

note_strings = ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3',
                'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4',
                'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5',
                'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6']

fret_board_notes = np.array([
  [x for x in range(0, NUMBER_FRETS+1)],
  [x for x in range(5, 5 + NUMBER_FRETS+1)],
  [x for x in range(10, 10 + NUMBER_FRETS+1)],
  [x for x in range(15, 15 + NUMBER_FRETS+1)],
  [x for x in range(19, 19 + NUMBER_FRETS+1)],
  [x for x in range(24, 24 + NUMBER_FRETS+1)]])

# See https://stackoverflow.com/questions/5835568/how-to-get-mfcc-from-an-fft-on-a-signal
def mel_filter_bank(blockSize, num_coefficients, min_hz, max_hz):
    numBands = int(num_coefficients)
    maxMel = int(freqToMel(max_hz))
    minMel = int(freqToMel(min_hz))

    # Create a matrix for triangular filters, one row per filter
    filterMatrix = np.zeros((numBands, blockSize))

    melRange = np.array(range(numBands + 2))

    melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel

    # each array index represent the center of each triangular filter
    aux = np.log(1 + 1000.0 / 700.0) / 1000.0
    aux = (np.exp(melCenterFilters * aux) - 1) / 22050
    aux = 0.5 + 700 * blockSize * aux
    aux = np.floor(aux)  # Arredonda pra baixo
    centerIndex = np.array(aux, int)  # Get int values

    for i in range(numBands):
        start, centre, end = centerIndex[i:i + 3]
        k1 = np.float32(centre - start)
        k2 = np.float32(end - centre)
        up = (np.array(range(start, centre)) - start) / k1
        down = (end - np.array(range(centre, end))) / k2

        filterMatrix[i][start:centre] = up
        filterMatrix[i][centre:end] = down

    return filterMatrix.transpose()

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)