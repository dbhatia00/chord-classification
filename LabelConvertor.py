import numpy as np
import os

def convert_labels_to_one_hot(labels):
    n = labels.shape[0]
    one_hot_tabs = np.zeros((n, 6, 21), dtype=int)

    for i in range(n):
        for finger in range(4):
            press = labels[i, finger, 0]
            fret = labels[i, finger, 1]
            string = labels[i, finger, 2] - 1

            if press == 1 and fret > 0:
                one_hot_tabs[i, string, :] = 0
                one_hot_tabs[i, string, fret] = 1
            elif press == 2 and finger == 0:
                for j in range(6):
                    one_hot_tabs[i, j, :] = 0
                    one_hot_tabs[i, j, fret] = 1

    return one_hot_tabs

def process_folder(input_folder, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .npy files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npy'):
            input_path = os.path.join(input_folder, file_name)
            labels = np.load(input_path)
            one_hot_labels = convert_labels_to_one_hot(labels)
            
            output_path = os.path.join(output_folder, file_name)
            np.save(output_path, one_hot_labels)
            print(f"Processed and saved: {output_path}")

# Usage
input_folder = 'transcription_data/tablature_labels'
output_folder = 'transcription_data/tablature_labels_converted'
process_folder(input_folder, output_folder)