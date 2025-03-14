#both cameras pull hand detection
import mediapipe as mp
import numpy as np 
import cv2 
import time
import pandas as pd
from datetime import datetime
import os
import pyrealsense2 as rs

def ask_save_data():
    """Shows a window to ask if user wants to save data"""
    window_name = 'Save Data?'
    cv2.namedWindow(window_name)
    print("Press 'y' for Yes or 'n' for No")
    while True:
        cv2.imshow(window_name, np.zeros((200, 400)))
        cv2.putText(
            img=np.zeros((200, 400)),
            text="Save data? Press 'y' for Yes, 'n' for No",
            org=(10, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=2
        )
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y'):
            cv2.destroyWindow(window_name)
            return True
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            return False

def ask_name():
    """Shows a window to input name"""
    name = ""
    window_name = 'Enter Name'
    cv2.namedWindow(window_name)
    print("Type your name and press Enter")
    while True:
        temp_img = np.zeros((200, 400))
        cv2.putText(
            img=temp_img,
            text=f"Name: {name}",
            org=(10, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=2
        )
        cv2.putText(
            img=temp_img,
            text="Press Enter when done",
            org=(10, 150),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=2
        )
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            if name:  # if name is not empty
                cv2.destroyWindow(window_name)
                return name
        elif key == 8:  # Backspace
            name = name[:-1]
        elif 32 <= key <= 126:  # Printable characters
            name += chr(key)

# Get user preferences using GUI
print("Welcome to Hand Tracking!")
save_data = ask_save_data()
user_name = ""
if save_data:
    user_name = ask_name()
    print(f"Data will be saved for user: {user_name}")
else:
    print("Data will not be saved")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture for regular camera
cap1 = cv2.VideoCapture(2)  # First camera

# Initialize Intel RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the RealSense pipeline
try:
    pipeline.start(config)
    realsense_initialized = True
    print("Intel RealSense camera initialized successfully")
except Exception as e:
    realsense_initialized = False
    print(f"Error initializing Intel RealSense camera: {e}")
    
# Check if first camera opened successfully
if not cap1.isOpened():
    print("Error: Could not open camera 1")
    if realsense_initialized:
        pipeline.stop()
    exit()
    
if not realsense_initialized:
    print("Error: Could not initialize Intel RealSense camera")
    cap1.release()
    exit()

# Initialize variables
data = []
start_time = time.time()
prev_time = time.time()

# Dictionary to map landmark indices to finger names
finger_map = {
    4: "THUMB_TIP",
    8: "INDEX_FINGER_TIP",
    12: "MIDDLE_FINGER_TIP",
    16: "RING_FINGER_TIP",
    20: "PINKY_TIP",
    3: "THUMB_IP",
    7: "INDEX_FINGER_DIP",
    11: "MIDDLE_FINGER_DIP",
    15: "RING_FINGER_DIP",
    19: "PINKY_DIP"
}

print("Starting hand tracking... Show your hand to the cameras.")
print("Press 'q' when you want to stop recording and save the data.")

while True:
    # Read frame from regular camera
    ret1, frame1 = cap1.read()
    
    # Get frames from RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    if not color_frame or not ret1:
        print("Error: Failed to grab frame from one or both cameras")
        break
    
    # Convert RealSense color frame to numpy array
    frame2 = np.asanyarray(color_frame.get_data())
    
    # Process first camera frame
    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    result1 = hands.process(rgb_frame1)
    
    # Draw hand landmarks on first camera frame
    if result1.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(result1.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get handedness (left/right)
            handedness = result1.multi_handedness[hand_idx].classification[0].label
            
            # Timestamp
            timestamp = time.time() - start_time
            
            # Landmarks 
            for idx, landmark in enumerate(hand_landmarks.landmark):
                finger_name = finger_map.get(idx, f"LANDMARK_{idx}")
                data.append([
                    timestamp,
                    "Camera1",
                    handedness,
                    finger_name,
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
    
    # Process RealSense camera frame
    rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    result2 = hands.process(rgb_frame2)
    
    # Draw hand landmarks on RealSense camera frame
    if result2.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(result2.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame2, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get handedness (left/right)
            handedness = result2.multi_handedness[hand_idx].classification[0].label
            
            # Timestamp
            timestamp = time.time() - start_time
            
            # Landmarks 
            for idx, landmark in enumerate(hand_landmarks.landmark):
                finger_name = finger_map.get(idx, f"LANDMARK_{idx}")
                # For RealSense camera, we can also get the depth value
                depth_value = 0
                if depth_frame:
                    # Convert normalized coordinates to pixel coordinates
                    h, w, _ = frame2.shape
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    
                    # Ensure pixel coordinates are within bounds
                    if 0 <= px < w and 0 <= py < h:
                        depth_value = depth_frame.get_distance(px, py)
                
                data.append([
                    timestamp,
                    "RealSense",
                    handedness,
                    finger_name,
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    depth_value  # Add depth data from RealSense
                ])
    
    # Calculate and display FPS
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    
    # Add on-screen instructions and FPS to both frames
    cv2.putText(frame1, "Regular Camera - Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(frame1, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    
    cv2.putText(frame2, "RealSense Camera - Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(frame2, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    
    # Display both camera frames
    cv2.imshow("Regular Camera", frame1)
    cv2.imshow("RealSense Camera", frame2)
    
    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nProcessing data...")
# Convert data to a pandas DataFrame
# Note: RealSense data has an additional depth column
columns = ["Time", "Camera", "Hand", "Finger_Position", "X", "Y", "Z"]
if any(len(row) > 7 for row in data):  # Check if we have any RealSense data
    columns.append("Depth")
    
df = pd.DataFrame(data, columns=columns)

if save_data:
    try:
        # Create a filename with username and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_tracking_{user_name}_{timestamp}.xlsx"
        
        # Create 'data' directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Save to Excel with descriptive column names
        filepath = os.path.join('data', filename)
        df.to_excel(filepath, index=False)
        print(f"Data saved successfully to '{filepath}'")
        print(f"Total frames captured: {len(data)}")
    except Exception as e:
        print(f"Error saving data: {e}")
else:
    print("Data was not saved as per user choice")
    print(f"Total frames captured: {len(data)}")

# Release resources
cap1.release()
pipeline.stop()
cv2.destroyAllWindows()
print("Program completed.")
