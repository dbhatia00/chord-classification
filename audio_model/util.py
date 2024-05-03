import numpy as np

SAMPLE_FREQ = 4
NUMBER_FRETS = 22

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
