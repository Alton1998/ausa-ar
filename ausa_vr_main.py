import sys
# https://pub.dev/packages/ultralytics_yolo/example
import cv2
from sympy.printing.pretty.pretty_symbology import annotated
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n-pose.pt")
left_shoulder_idx = 5
right_shoulder_idx = 6
left_elbow_idx = 7
right_elbow_idx = 8
right_hip_idx = 12

relevant_idxs = {
    "ecg": {
        "left_shoulder_idx" : left_shoulder_idx,
        "right_shoulder_idx": right_shoulder_idx,
        "right_hip_idx": right_hip_idx
    },
    "bp":{
        "left_shoulder_idx" : left_shoulder_idx,
        "right_shoulder_idx": right_shoulder_idx,
        "left_elbow_idx": left_elbow_idx,
        "right_elbow_idx":right_elbow_idx,
    }
}

mode = "ecg"


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
        confidence = results[0].keypoints.conf.detach().numpy()[0]
        normalized_coordinates = results[0].keypoints.xyn.detach().numpy()[0]

        annotated_frame = frame
        for points_interest in relevant_idxs[mode].values():
            if confidence[points_interest] > 0.5:
                x,y = normalized_coordinates[points_interest]
                if points_interest == right_hip_idx:
                    x = int((x+0.05) * frame.shape[1])
                    y = int((y-0.3) * frame.shape[0])
                else:
                    x = int(x * frame.shape[1])
                    y = int(y * frame.shape[0])
                annotated_frame = cv2.circle(annotated_frame,(x,y),5,(255, 0, 0),10)

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
