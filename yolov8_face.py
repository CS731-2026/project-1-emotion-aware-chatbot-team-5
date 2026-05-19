# ============================================================
# CS731 Emotion Recognition - Standalone Webcam Demo
# Uses YOLOv8-face to detect faces in real-time and
# classifies emotion on each detected face using our
# trained EfficientNet-B0 model.
#
# Run this file directly to test emotion detection
# without the full GUI: python yolov8_face.py
# Press 'q' to quit.
# ============================================================

import cv2
import torch
from ultralytics import YOLO
from inference import Inferencer

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
YOLO_MODEL_PATH    = 'yolov8n-face.pt'                   # YOLOv8 face detection weights
EMOTION_MODEL_PATH = 'checkpoints/efficientnet/best.pt'  # Our best trained model

# -----------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------

# YOLOv8-face — detects face locations in each frame
# This only finds WHERE faces are, not the emotion
face_model = YOLO(YOLO_MODEL_PATH)

# Our custom EfficientNet-B0 emotion classifier
# Loaded via Inferencer class from inference.py
emotion_inferencer = Inferencer(EMOTION_MODEL_PATH)

# -----------------------------------------------------------
# START WEBCAM
# -----------------------------------------------------------

# Open default webcam (index 0)
# Use CAP_V4L2 backend with MJPG format for WSL compatibility
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Webcam demo started. Press 'q' to quit.")

# -----------------------------------------------------------
# MAIN LOOP — process each webcam frame
# -----------------------------------------------------------
while True:
    # Read one frame from webcam
    ret, frame = cap.read()

    # If frame read failed, exit loop
    if not ret:
        print("Failed to read frame from webcam.")
        break

    # Run YOLOv8 face detection on current frame
    # Returns bounding box coordinates for each detected face
    results = face_model(frame, verbose=False, conf=0.4)

    # Process each detected face
    for result in results:
        # Get bounding box coordinates [x1, y1, x2, y2] for all faces
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Add padding around face crop for better context
            pad = 30
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            # Crop just the face region from the full frame
            face_img = frame[y1:y2, x1:x2]

            # Skip if face crop is empty
            if face_img.size == 0:
                continue

            # Run emotion classification on the face crop
            # Returns: label, display name, confidence score, emoji
            emotion_label, emotion_display, confidence, emoji = emotion_inferencer.predict(face_img)

            # Draw green bounding box around detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw emotion label above bounding box
            label_text = f"{emoji} {emotion_display}: {confidence:.0%}"
            cv2.putText(
                frame,
                label_text,
                (x1, max(y1 - 10, 20)),       # Position above box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,                           # Font size
                (0, 255, 0),                   # Green colour
                2                              # Line thickness
            )

    # Display the annotated frame in a window
    cv2.imshow("CS731 — Face Detection and Emotion Classification", frame)

    # Press 'q' key to quit the demo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------------------------------------
# CLEANUP — release webcam and close windows
# -----------------------------------------------------------
cap.release()
print("Webcam demo ended.")
