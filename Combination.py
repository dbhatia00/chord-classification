import cv2
import mediapipe as mp
import numpy as np
import pyaudio

class HandTracker:
    def __init__(self):
        # Initialize mediapipe hands module
        self.mphands = mp.solutions.hands
        self.mpdrawing = mp.solutions.drawing_utils
        self.hands = self.mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    def __del__(self):
        # Release resources
        self.hands.close()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

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

    def get_sound_data(self):
        # Function to get sound data from the live stream for a specified duration
        duration_sec = 0.5
        num_frames = int(44100 / 1024 * duration_sec)
        sound_data = b''

        for _ in range(num_frames):
            sound_data += self.stream.read(1024)

        return sound_data

    def process_sound(self, sound_data):
        # Function to process sound data when no significant change in hand position is detected
        pass

def process_frame(frame):
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Apply Hough Line Segment Transformation
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Draw the detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return frame

def main():
    # Initialize video capture (0 = default camera)
    cap = cv2.VideoCapture(0)

    # Initialize the hand tracker
    hand_tracker = HandTracker()

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame

        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame for Hough transformation
        frame = process_frame(frame)

        # Track hands and overlay the hand skeleton
        frame = hand_tracker.track_hands(frame)

        # Display the frame
        cv2.imshow('Hand and Line Tracking', frame)

        # Exit on pressing the escape key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()