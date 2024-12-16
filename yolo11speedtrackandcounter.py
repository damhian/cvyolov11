import cv2
from ultralytics import YOLO
from speed import SpeedEstimator
import tkinter as tk
from tkinter import filedialog

# Load YOLOv11 model
model = YOLO("yolo11s.pt")

# Initialize global variable to store cursor coordinates
line_pts = [(0, 288), (1019, 288)]  # Static line definition
names = model.model.names  # This is a dictionary

# Object counter per lane
object_counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}

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
cap = cv2.VideoCapture(video_path)
count = 0

def is_crossing_line(obj_center, line_start, line_end):
    """Check if the object's center crosses the line."""
    x1, y1 = line_start
    x2, y2 = line_end
    cx, cy = obj_center

    # Line equation check
    if y1 - 5 <= cy <= y1 + 5:
        return True
    return False

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
    results_list = model.track(frame, persist=True)  # Returns a list

    for result in results_list:
        if result.boxes:  # Ensure there are detections in this result
            for box in result.boxes:
                # Extract class index, confidence, and bounding box coordinates
                obj_class = int(box.cls)  # Class index
                obj_name = names[obj_class]  # Map index to class name
                bbox = box.xyxy[0].cpu().numpy()  # Convert tensor to NumPy array
                obj_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

                # Check if the object crosses the defined line
                if is_crossing_line(obj_center, line_pts[0], line_pts[1]):
                    if obj_name in object_counts:
                        object_counts[obj_name] += 1

    # Draw the lane line
    cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 0), 2)

    # Display counters on the frame
    y_offset = 30
    for obj, count in object_counts.items():
        text = f"{obj}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 20

    im0 = speed_obj.estimate_speed(frame, results_list)

    # Display the frame with YOLOv11 results and counters
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()