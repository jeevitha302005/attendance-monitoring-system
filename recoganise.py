import cv2
import imutils
import numpy as np
import pickle
import time
from datetime import datetime
import csv
import os

# Load mapping of student names to register numbers from "student.csv"
student_info = {}
with open("student.csv", "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        if len(row) >= 2:
            name, reg_number = row[0].strip(), row[1].strip()
            student_info[name] = reg_number

# Load the face detector, embedder, recognizer, and label encoder
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

embeddingModel = "openface_nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizerPath = "output/recognizer.pickle"
lePath = "output/le.pickle"
recognizer = pickle.loads(open(recognizerPath, "rb").read())
le = pickle.loads(open(lePath, "rb").read())

# Initialize the video stream
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# Prepare a dictionary for attendance and consecutive detections
attendance = {}
consecutiveCounts = {}  # key: student name, value: count of consecutive detections

# Get today's date to create a daily attendance CSV file
today_date = datetime.now().strftime("%Y-%m-%d")
attendance_filename = f"{today_date}_attendance.csv"

# Set recognition parameters
RECOGNITION_THRESHOLD = 0.9  # require 90% confidence
CONSECUTIVE_FRAMES_REQUIRED = 3  # require detection in 3 consecutive frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Prepare the frame for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)
        
        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        if fW < 20 or fH < 20:
            continue

        faceBlob = cv2.dnn.blobFromImage(
            face, 1.0 / 255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False
        )
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        preds = recognizer.predict_proba(vec.flatten().reshape(1, -1))[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]
        reg_number = student_info.get(name, "Unknown")

        # Only consider high-confidence predictions
        if proba > RECOGNITION_THRESHOLD:
            text = f"{reg_number}, {name}: {proba * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # Update consecutive detection count for this student
            if name in consecutiveCounts:
                consecutiveCounts[name] += 1
            else:
                consecutiveCounts[name] = 1

            # If the student has been detected in enough consecutive frames and not already marked
            if consecutiveCounts[name] >= CONSECUTIVE_FRAMES_REQUIRED and name not in attendance:
                current_time = datetime.now().strftime("%H:%M:%S")
                attendance[name] = (reg_number, current_time)
                print(f"[ATTENDANCE] {reg_number} - {name} marked at {current_time}")
        else:
            # Reset count for this student if detection confidence falls below threshold
            if name in consecutiveCounts:
                consecutiveCounts[name] = 0

    cv2.imshow("Attendance Monitoring", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save daily attendance to a CSV file
with open(attendance_filename, "w", newline="") as csvfile:
    fieldnames = ["RegisterNumber", "Name", "Time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for name, (reg_number, tstamp) in attendance.items():
        writer.writerow({"RegisterNumber": reg_number, "Name": name, "Time": tstamp})

print("Daily attendance saved in:", attendance_filename)
print("Attendance:", attendance)
