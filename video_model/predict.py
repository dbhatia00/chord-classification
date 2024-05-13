import typer
import numpy as np
import cv2
from video_model.util import SAMPLES_PER_SECOND, guitar_notes
from audio_model.util import note_strings

# Function to generate probabilities tensor from a frame
def generate_probabilities(frame):
    # Generate a 6x21 tensor of em
    # TODO: Call actual model with frame
    em_chord = np.zeros((6, 21))
    em_chord[3,1] = 1
    em_chord[4,1] = 1
    return em_chord

def reorder_probabilities(probability_tensor, threshold):
    # Initialize a list to store the reordered probabilities
    reordered_probabilities = []
    
    # Iterate over each timestamp in the probability tensor
    for timestamp_probs in probability_tensor:
        # Get the indices where probabilities are above the threshold
        above_threshold_indices = np.where(timestamp_probs > threshold)
        
        # Reorder the probabilities according to the note order
        reordered_timestamp_probs = [timestamp_probs[string_index, fret_index] 
                                      for string_index, fret_index in zip(*above_threshold_indices)]
        
        # Sum up the probabilities for duplicate notes across different strings
        summed_probs = np.zeros(len(note_strings))
        for string_index, fret_index in zip(*above_threshold_indices):
            note = guitar_notes[string_index][fret_index]
            note_index = note_strings.index(note)
            summed_probs[note_index] += timestamp_probs[string_index, fret_index]
        
        # Append the reordered probabilities to the list
        reordered_probabilities.append(summed_probs)
    
    # Convert the list of probabilities to a numpy array
    reordered_probabilities = np.array(reordered_probabilities)
    
    return reordered_probabilities


def videoPredict(model: str = typer.Option('model.pt'), 
         filepath: str = typer.Option('file.wav'), 
         raw_out: str = typer.Option('results/output.txt'),
         notes_out: str = typer.Option('results/notes.txt')):

    # Open video file
    video_path = filepath
    cap = cv2.VideoCapture(video_path)

    # Parameters
    frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))
    # Calculate frame interval for 0.125 frames per second
    frame_interval = int(frames_per_second * 0.125)
    # Calculate timestamp increment based on original frames per second
    timestamp_increment = 1.0 / frames_per_second

    # Generate probabilities tensor while iterating over frames at specified frequency
    probabilities_tensor = []
    timestamps = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            probabilities = generate_probabilities(frame)
            probabilities_tensor.append(probabilities)
            # Append the timestamp with the correct increment
            timestamps.append(timestamp_increment * frame_count)
        frame_count += 1

    # Close video file
    cap.release()

    # Convert to numpy array
    probabilities_tensor = np.array(probabilities_tensor)

    # Note probabilities should now match the structure of the output
    note_probabilities = reorder_probabilities(probability_tensor=probabilities_tensor, threshold=0.5)

    # Write tensor to file
    with open(raw_out, 'w') as file:
        for timestamp, probabilities in zip(timestamps, note_probabilities):
            file.write(f"{timestamp:.3f}: {list(probabilities)}\n")

    return timestamps, note_probabilities

if __name__ == '__main__':
    typer.run(videoPredict)
