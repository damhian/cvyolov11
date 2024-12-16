import cv2
from ultralytics import YOLO
from speed import SpeedEstimator
import tkinter as tk
from tkinter import filedialog

# Load YOLOv11 model
model = YOLO("yolo11s.pt")

# Initialize global variable to store cursor coordinates
line_pts = [(0, 288), (1019, 288)]
names = model.model.names  # This is a dictionary

speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

# Mouse callback function to capture mouse movement
def RGB(event, x, y, flags, param):
    global cursor_point
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_point = (x, y)
        print(f"Mouse coordinates: {cursor_point}")
        
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

# Set up the window and attach the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file or webcam feed
# cap = cv2.VideoCapture('speed.mp4')
cap = cv2.VideoCapture(video_path)
count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be read.")
        break

    count += 1
    if count % 2 != 0:  # Skip some frames for speed (optional)
        continue

    frame = cv2.resize(frame, (1020, 500))
    
    # Perform object tracking
    tracks = model.track(frame, persist=True)
    
    im0 = speed_obj.estimate_speed(frame,tracks)
    
    # Display the frame with YOLOv11 results
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
