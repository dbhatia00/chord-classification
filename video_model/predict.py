import typer
import numpy as np
import cv2
import torch
from video_model.model import GuitarTabCNN
from PIL import Image 
from video_model.util import SAMPLES_PER_SECOND, guitar_notes
from audio_model.util import note_strings
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.color import rgb2gray

# Function to generate probabilities tensor from a frame
def generate_probabilities(frame, model):
    # Convert image to grayscale using weighted sum
    image_gray = rgb2gray(frame)

    # Crop the image (example values)
    left = image_gray.shape[1] // 3
    top = image_gray.shape[0] // 2
    image_cropped = image_gray[top:, left:]
    #image_cropped = image_gray
    # Resize the image (example dimensions)
    width = 320
    height = 270
    image_resized = np.array(Image.fromarray(image_cropped).resize((width, height)))

    # Normalize the image
    image_normalized = image_resized / 255.0    
    data = np.load('video_model/mean_std.npy')
    mean = data[0]
    stddev = data[1]
    image_normalized = (image_resized - mean) / stddev

    # Convert NumPy array to PyTorch tensor
    image_tensor = torch.tensor(image_normalized, dtype=torch.float32)
    # Add batch dimension to the image tensor
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Pass the image tensor to the model
    with torch.no_grad():
        output = model(image_tensor)

    output = output.reshape(6, 21).numpy()

    return output

# Define the reorder_probabilities function
def reorder_probabilities(probability_tensor, threshold):
    # Initialize a list to store the reordered probabilities
    reordered_probabilities = []
    
    # Iterate over each timestamp in the probability tensor
    for timestamp_probs in probability_tensor:
        # Get the indices where probabilities are above the threshold
        above_threshold_indices = np.where(timestamp_probs > threshold)
        
        # Sum up the probabilities for each unique note without considering duplicates
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

def videoPredict(modelpath: str = typer.Option('model.pt'), 
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

    # Instantiate the model
    model = GuitarTabCNN()
    # Load the trained weights into the model
    model.load_state_dict(torch.load(modelpath, map_location=torch.device("cpu")))
    # Set the model to evaluation mode
    model.eval()

    # Generate probabilities tensor while iterating over frames at specified frequency
    probabilities_tensor = []
    timestamps = []
    frame_count = 0
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_count % frame_interval == 0) and (frame_count != 0):
                probabilities = generate_probabilities(frame, model)
                probabilities_tensor.append(probabilities)
                # Append the timestamp with the correct increment
                timestamps.append(timestamp_increment * frame_count)
            frame_count += 1
            pbar.update(1)

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
