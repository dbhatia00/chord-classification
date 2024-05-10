import cv2
import mediapipe as mp
import numpy as np
import os

# This file augments the visual dataset by overlaying hough lines and handtracker skeletons

class HandTracker:
    def __init__(self):
        # Initialize mediapipe hands module
        self.mphands = mp.solutions.hands
        self.mpdrawing = mp.solutions.drawing_utils
        self.hands = self.mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __del__(self):
        # Release resources
        self.hands.close()

    def track_hands(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        processFrames = self.hands.process(rgb_frame)

        # Draw landmarks on the original frame
        if processFrames.multi_hand_landmarks:
            for hand_landmarks in processFrames.multi_hand_landmarks:
                self.mpdrawing.draw_landmarks(frame, hand_landmarks, self.mphands.HAND_CONNECTIONS)

        return frame

def auto_canny_and_hough(frame, initial_threshold1=100, initial_threshold2=150, desired_line_count=100, max_iterations=2):
    # Adjust the thresholds based on the number of detected lines
    threshold1, threshold2 = initial_threshold1, initial_threshold2
    iteration = 0
    lines = None

    while iteration < max_iterations:
        # Apply Canny edge detection
        edges = cv2.Canny(frame, threshold1, threshold2)

        # Apply Hough Line Segment Transformation
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=175, maxLineGap=75)
        
        line_count = len(lines) if lines is not None else 0
        print(f"Iteration {iteration}: Thresholds=({threshold1}, {threshold2}), Lines Detected={line_count}")

        # Check if we have enough lines
        if line_count == desired_line_count:
            break
        elif line_count < desired_line_count:
            threshold1 -= 25  # Decrease threshold to make edge detection less strict
            threshold2 -= 25
        else:
            threshold1 += 25  # Increase threshold to make edge detection more strict
            threshold2 += 25

        iteration += 1

        # Safety check to prevent thresholds from going negative
        threshold1 = max(threshold1, 1)
        threshold2 = max(threshold2, 1)

    # Draw the detected lines on the frame for visualization
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

def process_images(folder_path, output_folder):
    # Check if the input folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return  # Stop the function if the folder doesn't exist

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the hand tracker
    hand_tracker = HandTracker()

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other file types if needed
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                new_width = int(image.shape[1] * 4)
                new_height = image.shape[0]
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Process the frame for Hough transformation
                image = auto_canny_and_hough(image)

                # Track hands and overlay the hand skeleton
                image = hand_tracker.track_hands(image)

                # Save the processed image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image)
                print(f"Processed and saved {output_path}")
            else:
                print(f"Warning: Unable to load image '{image_path}'. Skipping...")

if __name__ == "__main__":
    folder_path = 'chord_images'  # Update this to your folder path
    output_folder = 'augment_overlay'  # Update this to your desired output folder
    process_images(folder_path, output_folder)