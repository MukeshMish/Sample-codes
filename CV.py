import cv2
import numpy as np

# Initialize camera (change index for multiple cameras, e.g., 1, 2)
cap = cv2.VideoCapture(0)

# Read first frame and convert to grayscale
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute difference between consecutive frames
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply lower threshold to increase sensitivity
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # Find contours to draw bounding boxes around movement
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small movements
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show original frame with bounding boxes
    cv2.imshow("Motion Detection", frame2)

    # Update previous frame
    gray1 = gray2

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()