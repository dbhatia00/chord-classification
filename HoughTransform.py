import cv2
import numpy as np

def process_frame(frame):
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    cv2.imshow("Edges", edges)

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

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame

        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame
        frame = process_frame(frame)

        # Display the frame
        cv2.imshow('Hough Lines', frame)

        # Exit on pressing the escape key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()