import cv2
import numpy as np

def auto_canny_and_hough(frame, initial_threshold1=100, initial_threshold2=150, desired_line_count=10, max_iterations=5):
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
            threshold1 -= 50  # Decrease threshold to make edge detection less strict
            threshold2 -= 50
        else:
            threshold1 += 50  # Increase threshold to make edge detection more strict
            threshold2 += 50

        iteration += 1

        # Safety check to prevent thresholds from going negative
        threshold1 = max(threshold1, 1)
        threshold2 = max(threshold2, 1)

    # Draw the detected lines on the frame for visualization
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, edges

# Example usage with an image

frame = cv2.imread('chord_images/A628.png')
new_width = int(frame.shape[1] * 4)
new_height = frame.shape[0]
frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

processed_frame, edges = auto_canny_and_hough(frame)
cv2.imshow("Processed Frame", processed_frame)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()