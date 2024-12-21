import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n-pose.pt")

# Define keypoint indices
left_shoulder_idx = 5
right_shoulder_idx = 6
left_elbow_idx = 7
right_elbow_idx = 8
right_hip_idx = 12

# Load the overlay image (ensure it's a transparent PNG)
image_path = "Ausa_BP_black.png"
overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Define relevant indices for the mode
relevant_idxs = {
    "ecg": {
        "left_shoulder_idx": left_shoulder_idx,
        "right_shoulder_idx": right_shoulder_idx,
        "right_hip_idx": right_hip_idx
    },
    "bp": {
        "left_shoulder_idx": left_shoulder_idx,
        "right_shoulder_idx": right_shoulder_idx,
        "left_elbow_idx": left_elbow_idx,
        "right_elbow_idx": right_elbow_idx,
    }
}

# Set the mode
mode = "bp"

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

        overlay_pos = None
        overlay_size = None
        angle = 0

        # Debug and overlay logic for the "bp" mode
        if mode == "bp":
            if confidence[right_shoulder_idx] > 0.5 and confidence[right_elbow_idx] > 0.5:
                # Extract coordinates
                x1, y1 = normalized_coordinates[right_shoulder_idx]
                x2, y2 = normalized_coordinates[right_elbow_idx]

                # Convert normalized coordinates to pixel values
                x1, y1 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0])
                x2, y2 = int(x2 * frame.shape[1]), int(y2 * frame.shape[0])

                # Debug: Plot points on the frame
                frame = cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)  # Red dot for (x1, y1)
                frame = cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)  # Green dot for (x2, y2)

                # Debug: Add text labels for the points
                frame = cv2.putText(frame, f"R-Shoulder ({x1}, {y1})", (x1 + 10, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                frame = cv2.putText(frame, f"R-Elbow ({x2}, {y2})", (x2 + 10, y2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Calculate the angle
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                overlay_pos = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Overlay image logic
                if overlay_pos:
                    overlay_size = int(0.2 * frame.shape[1]), int(0.2 * frame.shape[0])
                    resized_overlay = cv2.resize(overlay_img, overlay_size[:2])

                    # Rotate the image
                    h, w, _ = resized_overlay.shape
                    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    rotated_overlay = cv2.warpAffine(
                        resized_overlay, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
                    )

                    # Place the rotated overlay image on the frame
                    x, y = overlay_pos
                    h, w, _ = rotated_overlay.shape
                    x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
                    x2, y2 = min(frame.shape[1], x1 + w), min(frame.shape[0], y1 + h)

                    # Ensure dimensions match
                    overlay_cropped = rotated_overlay[:y2 - y1, :x2 - x1]

                    # Create masks for blending
                    alpha_overlay = overlay_cropped[:, :, 3] / 255.0
                    alpha_frame = 1.0 - alpha_overlay

                    for c in range(3):  # Blend each channel
                        frame[y1:y2, x1:x2, c] = (
                            alpha_overlay * overlay_cropped[:, :, c]
                            + alpha_frame * frame[y1:y2, x1:x2, c]
                        )

        # Show the updated frame
        cv2.imshow("YOLO11 Tracking with Rotation", frame)

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
