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

def process_frame(frame):
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Apply Hough Line Segment Transformation
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=175, maxLineGap=75)

    # Draw the detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return frame

def process_images(folder_path, output_folder):
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
                # Process the frame for Hough transformation
                image = process_frame(image)

                # Track hands and overlay the hand skeleton
                image = hand_tracker.track_hands(image)

                # Save the processed image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image)
                print(f"Processed and saved {output_path}")

if __name__ == "__main__":
    folder_path = 'path_to_your_input_folder'  # Update this to your folder path
    output_folder = 'path_to_your_output_folder'  # Update this to your desired output folder
    process_images(folder_path, output_folder)