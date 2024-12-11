import streamlit as st
import cv2
import random
import tempfile
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
import time
import base64

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    .stSelectbox div {{
        color: white;
    }}
    .stSubheader {{
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with the uploaded background image
image_base64 = get_base64_image("background.jpg")  # Path to your uploaded image
set_background(image_base64)


# Title and description
st.title("🚦 Vision in Motion: Intelligent Object Detection with YOLOv8 🚗")
st.markdown('''
Welcome to the interactive object detection dashboard! 🎥  
This app allows you to:  

- ✨ Upload a video and detect objects in it using YOLOv8.  
- 🎯 Visualize detections with bounding boxes and class labels.  
- 🌈 Adjust detection confidence for more accurate results.  
- 📊 Gain insights into detection performance and speed.  

**Let's get started!** 🌟
''')

# Sidebar for user inputs
st.sidebar.header("Upload and Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.45, 
    step=0.05
)
frame_interval = st.sidebar.slider(
    "Frame Interval", 
    min_value=1, 
    max_value=10, 
    value=5, 
    step=1,
    help="Process every nth frame to reduce processing time"
)

# File uploader for video input
uploaded_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi"])

# Display placeholder for the output video
video_placeholder = st.empty()

# Load class names
coco_path = Path(r"C:\Users\91939\Desktop\AI&DS\Data science projects\YOLO\utils\coco.txt")
with open(coco_path, "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for bounding boxes
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in class_list
]

# Load YOLO model
model = YOLO("weights/yolov8n.pt")

# Check if CUDA (GPU) is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Process the uploaded video
if uploaded_file:
    st.sidebar.success("Video uploaded successfully! 🎉")
    
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    st.sidebar.info("Processing video...")

    # Capture video using OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary directory to store processed frames
    temp_output_dir = tempfile.mkdtemp()

    # Process video frame by frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))  # Reduce resolution to 640x480

        # Process every nth frame to reduce processing time
        if frame_count % frame_interval == 0:
            # Detect objects using YOLOv8
            results = model.predict(source=[frame], conf=confidence_threshold, device=device)
            detections = results[0]

            # Draw detections on the frame
            for i in range(len(detections.boxes)):
                box = detections.boxes[i]
                clsID = int(box.cls.numpy()[0])
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                # Draw bounding box
                cv2.rectangle(
                    frame, 
                    (int(bb[0]), int(bb[1])), 
                    (int(bb[2]), int(bb[3])), 
                    detection_colors[clsID], 
                    2
                )
                # Add label and confidence
                cv2.putText(
                    frame, 
                    f"{class_list[clsID]} {conf:.2f}", 
                    (int(bb[0]), int(bb[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )

            # Convert frame to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, caption=f"Frame {frame_count + 1}", use_column_width=True)

            # Add delay based on FPS to simulate video playback
            time.sleep(1 / fps)  # Adding a small delay for FPS control

        frame_count += 1

    cap.release()
    st.sidebar.success("Video processing complete! 🎬")

    st.sidebar.info("End of video. 🛑")

# Footer
st.markdown("Developed with ❤️ using Streamlit and YOLOv8.")



