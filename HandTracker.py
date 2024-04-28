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
        self.counter = 0

    def __del__(self):
        # Release resources
        self.hands.close()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def track_hands_image(self, image_path):
        # Load image
        image = cv2.imread(image_path)

        # Track hands in the image
        return self._track_hands(image)

    def track_hands_live_stream(self):
        # Start capturing video from webcam
        vidcap = cv2.VideoCapture(0)
        
        # Initialize variable to store previous hand landmarks
        prev_landmarks = None

        while True:
            ret, frame = vidcap.read()
            if not ret:
                break

            # Track hands in the frame
            frame_with_hands = self._track_hands(frame)

            # Display the frame
            cv2.imshow('Hand Tracking', frame_with_hands)

            # Detect changes in hand position
            current_landmarks = self._get_hand_landmarks(frame)
            if prev_landmarks is not None and current_landmarks is not None:
                # Calculate Euclidean distance between corresponding landmarks
                # For simplicity, consider only the first hand detected (hopefully should be the fretting hand)
                prev_hand = prev_landmarks[0]
                curr_hand = current_landmarks[0]
                distance = np.linalg.norm(prev_hand - curr_hand)
                
                # Threshold for significant change in hand position
                threshold = 0.3  # TODO: Tweak for best fit

                if distance > threshold:
                    print("Changing Position")
                else:
                    print("")
                    # Call function to handle sound processing when no significant change detected
                    sound_data = self.get_sound_data()  # Get sound data from the live stream
                    self.process_sound(sound_data)  # Pass the sound data to the sound processing function


            # Update previous landmarks
            prev_landmarks = current_landmarks

            # Exit loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close windows
        vidcap.release()
        cv2.destroyAllWindows()


    def get_sound_data(self):
        # Ends up acting like a refresh rate on the video stream ig?
        duration_sec = 0.5
        # Function to get sound data from the live stream for a specified duration
        num_frames = int(44100 / 1024 * duration_sec)
        sound_data = b''

        # Read sound data in chunks
        for _ in range(num_frames):
            sound_data += self.stream.read(1024)

        return sound_data

    def process_sound(self, sound_data):
        # Function to process sound data when no significant change in hand position is detected
        pass

    def _get_hand_landmarks(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        processFrames = self.hands.process(rgb_frame)

        # Extract landmarks if hands are detected
        if processFrames.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in processFrames.multi_hand_landmarks:
                landmarks.append(np.array([[lmk.x, lmk.y] for lmk in hand_landmarks.landmark]))
            return landmarks
        else:
            return None


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