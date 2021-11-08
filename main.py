import cv2
import numpy as np

cap = cv2.VideoCapture("./videos/test.mp4")
# cap = cv2.VideoCapture("/dev/video2")
car_cascade = cv2.CascadeClassifier("./cars.xml")

prev_frame = 0

while True:
    # Read frames from the video
    ret, frames = cap.read()

    # Convert the frames to gray scale
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Calculate the difference
    diff = cv2.absdiff(gray, prev_frame)

    # Store frame as new prev_frame
    prev_frame = gray

    # Image threshholding and dilating
    # Removing noise and stuff
    ret1, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Detect cars of different sizes
    cars = car_cascade.detectMultiScale(dilated, 1.1, 1)

    # Draw a rectangle around each car
    for (x, y, w, h) in cars:
        print(f"Found a car at: {(x, y)} with the size: {(w, h)}")
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Display the frames in a window
    cv2.imshow("Car detection", frames)
    cv2.imshow("Car detection dilated", dilated)
    # Wait for enter to stop
    if cv2.waitKey(33) == 13:
        break
