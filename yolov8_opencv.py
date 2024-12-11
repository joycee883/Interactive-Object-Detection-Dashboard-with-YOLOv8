# Import necessary libraries
import random
import cv2  # OpenCV for video processing
import numpy as np  # NumPy for numerical operations
from ultralytics import YOLO  # YOLO model from ultralytics


# Load a list of class names from a COCO dataset file
with open(r"C:\Users\91939\Desktop\AI&DS\Data science projects\YOLO\utils\coco.txt", "r") as my_file:
    # Read the file content
    data = my_file.read()
    # Split the content by newline to get each class name
    class_list = data.split("\n")

# Generate random colors for each class
detection_colors = []
for _ in class_list:
    # Create a random RGB color
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    detection_colors.append(color)

# Load the YOLOv8 model (weights path and version specified)
model = YOLO("weights/yolov8n.pt", "v8")

# Define width and height to resize frames for optimization
frame_wid = 640
frame_hyt = 480

# Initialize video capture (can be a video file or webcam)
cap = cv2.VideoCapture(r"C:\Users\91939\Desktop\AI&DS\Data science projects\YOLO\Inputs\CarTraffic.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Cannot open video")
    exit()

# Add this before the while loop to create a resizable window
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 1280, 720)  # Set to a larger resolution or desired size

# Loop to continuously read and process video frames
while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (end of stream?). Exiting ...")
        break

    # Resize the frame for faster processing (optional)
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Use the YOLO model to detect objects in the frame
    detect_params = model.predict(source=[frame], conf=0.45, save=True)

    # Convert detection results to numpy array for processing
    DP = detect_params[0].numpy()
    print(DP)  # Print detection info for debugging (optional)

    # If there are any detected objects
    if len(DP) != 0:
        # Loop through each detected object
        for i in range(len(detect_params[0])):
            # Get bounding boxes for the current frame
            boxes = detect_params[0].boxes
            box = boxes[i]  # Get each bounding box
            
            # Extract class ID, confidence, and bounding box coordinates
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw bounding box on the frame with respective color
            # Inside your loop where bounding boxes are drawn, change the color to yellow
            cv2.rectangle(frame, 
              (int(bb[0]), int(bb[1])),  # Top-left corner
              (int(bb[2]), int(bb[3])),  # Bottom-right corner
              (0, 255, 255),              # Yellow color in BGR format
              3)                          # Thickness of the bounding box


            # Set font for text display
            font = cv2.FONT_HERSHEY_COMPLEX
            # Add class name and confidence score above the bounding box
            cv2.putText(
                frame,
                f"{class_list[clsID]} {round(conf * 100, 2)}%",  # Class name and confidence
                (int(bb[0]), int(bb[1]) - 10),  # Position above the bounding box
                font,
                1,  # Font size
                (255, 255, 255),  # White color for text
                2  # Thickness of text
            )

    # Display the frame with detection
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
        
