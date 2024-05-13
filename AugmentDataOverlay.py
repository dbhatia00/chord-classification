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

        landmarks_list = []  # List to store hand landmarks

        # Draw landmarks on the original frame and collect landmark data
        if processFrames.multi_hand_landmarks:
            for hand_landmarks in processFrames.multi_hand_landmarks:
                self.mpdrawing.draw_landmarks(frame, hand_landmarks, self.mphands.HAND_CONNECTIONS)
                # Collect each landmark x, y coordinates
                landmarks = [(lmk.x * frame.shape[1], lmk.y * frame.shape[0]) for lmk in hand_landmarks.landmark]
                landmarks_list.append(landmarks)

        return frame, landmarks_list

def hough_transform_and_join(frame, hand_frame, initial_threshold1=350, initial_threshold2=400, desired_line_count=2, max_iterations=2):
    # Adjust the thresholds based on the number of detected lines
    threshold1, threshold2 = initial_threshold1, initial_threshold2
    iteration = 0
    lines = None

    while iteration < max_iterations:
        # Apply Canny edge detection
        edges = cv2.Canny(frame, threshold1, threshold2)

        # Apply Hough Line Segment Transformation
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=500, maxLineGap=150)
        
        line_count = len(lines) if lines is not None else 0
        print(f"Iteration {iteration}: Thresholds=({threshold1}, {threshold2}), Lines Detected={line_count}")

        # Check if we have enough lines
        if line_count == desired_line_count:
            break
        elif line_count < desired_line_count:
            threshold1 -= 50  # Decrease threshold to make edge detection less strict
            threshold2 -= 50
        else:
            threshold1 += 50  # Increase threshold to make edge detection more strict
            threshold2 += 50

        iteration += 1

        # Safety check to prevent thresholds from going negative
        threshold1 = max(threshold1, 1)
        threshold2 = max(threshold2, 1)
        print(line_count)

    # Draw the detected lines on the frame for visualization
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hand_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return hand_frame

def calculate_angle(p1, p2):
    # Calculate the angle from horizontal
    return -np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def rotate_image(image, angle, center):
    # Get the rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)


    # Perform the rotation
    return cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

def process_images_with_rotation_and_cropping(folder_path, output_folder_combination, output_folder_hand, hand_tracker):
    desired_width = 1219  # Desired width of the cropped image
    desired_height = 500  # Desired height of the cropped image

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                image_copy = image.copy()
                hand_image, landmarks_list = hand_tracker.track_hands(image)

                if len(landmarks_list) >= 2:
                    hands = sorted(landmarks_list, key=lambda x: np.mean([lm[0] for lm in x]))
                    left_hand, right_hand = hands[0], hands[1]

                    # Calculate centroids of each hand
                    left_centroid = (np.mean([lm[0] for lm in left_hand]), np.mean([lm[1] for lm in left_hand]))
                    right_centroid = (np.mean([lm[0] for lm in right_hand]), np.mean([lm[1] for lm in right_hand]))

                    # Midpoint between the hands
                    mid_point_y = int((left_centroid[1] + right_centroid[1]) / 2)

                    # Calculate the angle for rotation
                    angle = calculate_angle(left_centroid, right_centroid)
                    center = (left_centroid[0], mid_point_y)  # Rotate about the left hand and the midpoint vertically

                    # Rotate the image and the hands
                    rotated_image = rotate_image(image_copy, -angle, center)
                    rotated_hand_image = rotate_image(hand_image, -angle, center)

                    # New coordinates of the left hand after rotation
                    new_left_hand_x = int((left_centroid[0] - center[0]) * np.cos(np.radians(-angle)) - (left_centroid[1] - center[1]) * np.sin(np.radians(-angle)) + center[0])

                    # Calculate the crop dimensions
                    crop_x = max(new_left_hand_x, 0)
                    crop_y = max(mid_point_y - desired_height // 2, 0)
                    crop_width = min(desired_width, rotated_image.shape[1] - crop_x)
                    crop_height = min(desired_height, rotated_image.shape[0] - crop_y)

                    # Crop the image to the desired dimensions
                    cropped_image = rotated_image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
                    cropped_hand_image = rotated_hand_image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

                    # Save image with only hand tracker overlay
                    output_path_hand = os.path.join(output_folder_hand, filename)
                    cv2.imwrite(output_path_hand, cropped_hand_image)
                    print(f"Processed and saved {output_path_hand}")

                    joined_image = hough_transform_and_join(cropped_image, cropped_hand_image)

                    # Save image with combination overlay
                    output_path_combination = os.path.join(output_folder_combination, filename)
                    cv2.imwrite(output_path_combination, joined_image)
                    print(f"Processed and saved {output_path_combination}")
                else:
                    print(f"Not enough hands detected in {filename}. Skipping...")
            else:
                print(f"Warning: Unable to load image '{image_path}'. Skipping...")


if __name__ == "__main__":
    folder_path = 'transcription_data/tablature_frames'
    output_folder_combination = 'transcription_data/tablature_images_combination'
    output_folder_hand = 'transcription_data/tablature_images_hand'

    hand_tracker = HandTracker()
    process_images_with_rotation_and_cropping(folder_path, output_folder_combination, output_folder_hand, hand_tracker)