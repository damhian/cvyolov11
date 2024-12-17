import os
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

model = YOLO(r"D:\Ddisk\FWork\PyProject\cvyolo11\runs\detect\train2\weights\best.pt")  

# Initialize Tkinter and hide the main window
root = tk.Tk()
root.withdraw()

# Open a file dialog to select a video file
video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video Files", "*.mp4 *.avi *.mkv"), ("All Files", "*.*")]
)

# Check if a file was selected
if not video_path:
    print("No video file selected. Exiting...")
    exit()

# Open the video file or webcam feed
cap = cv2.VideoCapture(video_path)

# Create a folder for saving detected plates
output_folder = "detected_plates"
os.makedirs(output_folder, exist_ok=True)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be read.")
        break

    frame_count += 1

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (1020, 500))

    # Perform detection
    results_list = model.predict(frame)  # Run your YOLO model

    for result in results_list:
        if result.boxes:  # Ensure there are detections in this result
            for box in result.boxes:
                # Extract bounding box coordinates and class confidence
                bbox = box.xyxy[0].cpu().numpy()  # Convert tensor to NumPy array
                confidence = box.conf.item()  # Confidence score

                # Extract license plate area
                x1, y1, x2, y2 = map(int, bbox)
                license_plate = frame[y1:y2, x1:x2]

                # Save the license plate as an image
                plate_path = f"{output_folder}/plate_frame{frame_count}_conf{int(confidence*100)}.jpg"
                cv2.imwrite(plate_path, license_plate)

                # Draw bounding box and confidence on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Plate: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected license plates
    cv2.imshow("License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
