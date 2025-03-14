import mediapipe as mp
import numpy as np
import cv2
import time
import pandas as pd
from datetime import datetime
import os

def ask_save_data():
    """Popup box asking if the user wants to save data."""
    window_name = "Save Data?"
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(img, "Save Data? (y/n)", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(window_name, img)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            cv2.destroyWindow(window_name)
            return True
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            return False

def ask_name():
    """Popup box for user to enter their name."""
    window_name = "Enter Name"
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    name = ""

    while True:
        temp_img = img.copy()
        cv2.putText(temp_img, f"Name: {name}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(temp_img, "Press Enter when done", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(window_name, temp_img)

        key = cv2.waitKey(0) & 0xFF
        if key == 13 and name:  # Enter key
            cv2.destroyWindow(window_name)
            return name
        elif key == 8:  # Backspace
            name = name[:-1]
        elif 32 <= key <= 126:  # Printable characters
            name += chr(key)

# Get user preferences
print("Welcome to Hand & Pen Tracking!")
save_data = ask_save_data()
user_name = ask_name() if save_data else "anonymous"

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture for both cameras
cap1 = cv2.VideoCapture(2)  # First camera
cap2 = cv2.VideoCapture(3)  # Second camera - Intel RealSense camera

# Check if cameras opened successfully
if not cap1.isOpened():
    print("Error: Could not open camera 1")
    exit()
    
if not cap2.isOpened():
    print("Error: Could not open camera 2")
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

print("\nTracking started. Press 'q' to stop and save data.")

while True:
    # Read frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Error: Failed to grab frame from one or both cameras")
        break
    
    # Process first camera frame for hand detection
    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    result1 = hands.process(rgb_frame1)
    
    # Process first camera frame for pen detection
    hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    # Define color range for pen detection (adjust for your pen)
    lower_color = np.array([100, 150, 100])  # Example: Blue pen
    upper_color = np.array([140, 255, 255])
    mask1 = cv2.inRange(hsv_frame1, lower_color, upper_color)
    
    # Detect contours of the pen in camera 1
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pen_tip1_x, pen_tip1_y = None, None

    if contours1:
        largest_contour = max(contours1, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50:  # Minimum area threshold to avoid noise
            pen_tip = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])  # Tip of pen
            pen_tip1_x, pen_tip1_y = pen_tip

            # Draw pen tip
            cv2.circle(frame1, pen_tip, 5, (0, 255, 255), -1)
            cv2.putText(frame1, "Pen Tip", (pen_tip1_x, pen_tip1_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
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
    
    # Store pen data for camera 1
    if pen_tip1_x is not None and pen_tip1_y is not None:
        timestamp = time.time() - start_time
        data.append([
            timestamp,
            "Camera1",
            "PEN_TIP",
            "N/A",
            pen_tip1_x / frame1.shape[1],  # Normalize x coordinate
            pen_tip1_y / frame1.shape[0],  # Normalize y coordinate
            0  # Z coordinate (placeholder)
        ])
    
    # Process second camera frame for hand detection
    rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    result2 = hands.process(rgb_frame2)
    
    # Process second camera frame for pen detection
    hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv_frame2, lower_color, upper_color)
    
    # Detect contours of the pen in camera 2
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pen_tip2_x, pen_tip2_y = None, None

    if contours2:
        largest_contour = max(contours2, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50:  # Minimum area threshold
            pen_tip = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])  # Tip of pen
            pen_tip2_x, pen_tip2_y = pen_tip

            # Draw pen tip
            cv2.circle(frame2, pen_tip, 5, (0, 255, 255), -1)
            cv2.putText(frame2, "Pen Tip", (pen_tip2_x, pen_tip2_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw hand landmarks on second camera frame
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
                data.append([
                    timestamp,
                    "Camera2",
                    handedness,
                    finger_name,
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
    
    # Store pen data for camera 2
    if pen_tip2_x is not None and pen_tip2_y is not None:
        timestamp = time.time() - start_time
        data.append([
            timestamp,
            "Camera2",
            "PEN_TIP",
            "N/A", 
            pen_tip2_x / frame2.shape[1],  # Normalize x coordinate
            pen_tip2_y / frame2.shape[0],  # Normalize y coordinate
            0  # Z coordinate (placeholder)
        ])
    
    # Calculate and display FPS
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    
    # Add on-screen instructions and FPS to both frames
    for frame, camera_name in [(frame1, "Camera 1"), (frame2, "Camera 2")]:
        cv2.putText(frame, f"{camera_name} - Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    
    # Display both camera frames
    cv2.imshow("Camera 1 - Hand & Pen Tracking", frame1)
    cv2.imshow("Camera 2 - Hand & Pen Tracking", frame2)
    
    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nProcessing data...")
# Convert data to a pandas DataFrame
df = pd.DataFrame(data, columns=["Time", "Camera", "Hand_Object", "Finger_Position", "X", "Y", "Z"])

if save_data:
    try:
        # Create a filename with username and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_pen_tracking_{user_name}_{timestamp}.xlsx"
        
        # Create 'data' directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Save to Excel with descriptive column names
        filepath = os.path.join('data', filename)
        df.to_excel(filepath, index=False)
        print(f"Data saved successfully to '{filepath}'")
        print(f"Total data points captured: {len(data)}")
    except Exception as e:
        print(f"Error saving data: {e}")
else:
    print("Data was not saved as per user choice")
    print(f"Total data points captured: {len(data)}")

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Program completed.")
