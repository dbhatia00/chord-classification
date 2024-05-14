import cv2
import string
import random
import sys
import argparse
from HandTracker import HandTracker
from seeFretboard import SeeFretboard
from audio_model.predict import audioPredict
from video_model.predict import videoPredict
from Analyze import analyzeProbs, print_fretboard

def main(video_path):
    # Initialize tracker object
    tracker = HandTracker()

    # Generate a hash to use for intermediate representations
    hash_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    # Track hands in a saved video
    print("Extracting Audio from Video...")
    output_tracked_path = f'vid_dump/tracked-{hash_code}.mp4'
    audio_path = f'vid_dump/tracked-{hash_code}.wav'
    data = tracker._extract_audio(video_path, audio_path)

    # Do Audio Prediction
    print("Beginning Audio Prediction...")
    audio_timestamps, audio_probabilities = audioPredict(model="audio_model/model.pt", filepath=audio_path,
                 raw_out=f"./results/output-{hash_code}-audio.txt", notes_out=f"./results/notes-{hash_code}-audio.txt")

    

    print("Beginning Video Prediction...")

    video_timestamps, video_probabilities = videoPredict(modelpath="video_model/model.pt", filepath=video_path, 
                                                         raw_out=f"./results/output-{hash_code}-video.txt")
    print("Combining Data for best guesses...")   

    chords = analyzeProbs(audio_timestamps, audio_probabilities, video_timestamps, video_probabilities, hash_code) 
    print(f"Data available in results/FINAL-{hash_code}.txt")

    print(f"Chord progression is - {chords}")
    for chord in chords:
        print(chord[0])
        print_fretboard(chord[0])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform all video preprocessing and ")
    parser.add_argument("video_path", metavar="video_path", type=str, help="Path to the input video file")
    args = parser.parse_args()
    main(args.video_path)
