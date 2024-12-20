import sys

import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n-pose.pt")

# Open the webcam (0) or video file
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam or video file.")
    exit()

try:
    while True:
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame from video.")
            break

        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame in a window
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")
