from HandTracker import HandTracker
import cv2


tracker = HandTracker()
'''
# Track hands in an image
image_path = 'tempImage/GChord.png'
image_with_hands = tracker.track_hands_image(image_path)
cv2.imshow('Hand Tracking - Image', image_with_hands)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()
'''

# Track hands in a live stream
tracker.track_hands_live_stream()

'''
# Track hands in a saved video
video_path = 'tempImage/testVid.mp4'
tracker.track_hands_video(video_path)
'''
