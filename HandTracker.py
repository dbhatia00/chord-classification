import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        # Initialize mediapipe hands module
        self.mphands = mp.solutions.hands
        self.mpdrawing = mp.solutions.drawing_utils
        self.hands = self.mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __del__(self):
        # Release resources
        self.hands.close()

    def track_hands_image(self, image_path):
        # Load image
        image = cv2.imread(image_path)

        # Track hands in the image
        return self._track_hands(image)

    def track_hands_live_stream(self):
        # Start capturing video from webcam
        vidcap = cv2.VideoCapture(0)

        while True:
            ret, frame = vidcap.read()
            if not ret:
                break

            # Track hands in the frame
            frame_with_hands = self._track_hands(frame)

            # Display the frame
            cv2.imshow('Hand Tracking', frame_with_hands)

            # Exit loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close windows
        vidcap.release()
        cv2.destroyAllWindows()

    def track_hands_video(self, video_path):
        # Start capturing video from saved file
        vidcap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = vidcap.read()
            if not ret:
                break

            # Track hands in the frame
            frame_with_hands = self._track_hands(frame)

            # Display the frame
            cv2.imshow('Hand Tracking', frame_with_hands)

            # Exit loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close windows
        vidcap.release()
        cv2.destroyAllWindows()

    def _track_hands(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        processFrames = self.hands.process(rgb_frame)

        # Create a blank canvas to draw only the landmarks
        landmarks_canvas = np.zeros_like(frame)

        # Draw landmarks on the canvas
        if processFrames.multi_hand_landmarks:
            for hand_landmarks in processFrames.multi_hand_landmarks:
                # Draw landmarks on the canvas
                self.mpdrawing.draw_landmarks(landmarks_canvas, hand_landmarks, self.mphands.HAND_CONNECTIONS)

        # Return the canvas with only the landmarks
        return landmarks_canvas