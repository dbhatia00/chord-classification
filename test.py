import cv2
import string
import random
import sys
import argparse
from HandTracker import HandTracker
from audio_model.predict import audioPredict

def main(video_path):
    # Initialize tracker object
    tracker = HandTracker()

    hash_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    # Track hands in a saved video
    output_path = f'vid_dump/tracked-{hash_code}.mp4'
    audio_path = f'vid_dump/tracked-{hash_code}.wav'
    data = tracker.track_hands_video(video_path, audio_path, output_path)

    print("Beginning Audio Prediction...")
    audioPredict(model="audio_model/model.pt", filepath=audio_path,
                 raw_out=f"./results/output-{hash_code}.txt", notes_out=f"./results/notes-{hash_code}.txt")

    print(f"Output Notes Dumped in ./results/notes-{hash_code}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform all video preprocessing and ")
    parser.add_argument("video_path", metavar="video_path", type=str, help="Path to the input video file")
    args = parser.parse_args()
    main(args.video_path)
