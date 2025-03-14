import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import threading
import time
import datetime
import pandas as pd
import os
from queue import Queue

# Create directory for saving data
save_dir = "hand_tracking_data"
os.makedirs(save_dir, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec

# Initialize data structures
webcam_hand_data = Queue()
realsense_hand_data = Queue()
webcam_frames = Queue(maxsize=5)
realsense_frames = Queue(maxsize=5)

# Global flags
stop_event = threading.Event()
realsense_connected = threading.Event()
webcam_connected = threading.Event()

# Function to list all available cameras and their properties
def list_cameras():
    available_cameras = []
    
    # Try the first 10 indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Get camera information
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to get camera name (might not work on all platforms)
                name = f"Camera {i}"
                try:
                    name = cap.getBackendName()
                except:
                    pass
                
                available_cameras.append({
                    'index': i,
                    'name': name,
                    'resolution': f"{int(width)}x{int(height)}",
                    'fps': fps
                })
            cap.release()
    
    return available_cameras

# Function to process frames from HDMI webcam
def process_webcam(camera_index=1):  # Default to index 1 for HDMI webcam
    print(f"Trying to initialize HDMI webcam at index {camera_index}")
    
    # List available cameras first
    available_cameras = list_cameras()
    if available_cameras:
        print("Available cameras:")
        for camera in available_cameras:
            print(f"  Index {camera['index']}: {camera['name']} - {camera['resolution']} @ {camera['fps']}fps")
    
    # Initialize webcam - specifically target the HDMI camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open HDMI webcam at index {camera_index}")
        # Try to find any available camera
        for idx in range(10):
            if idx != camera_index:  # Skip the already tried index
                temp_cap = cv2.VideoCapture(idx)
                if temp_cap.isOpened():
                    ret, test_frame = temp_cap.read()
                    if ret:
                        print(f"Found alternative camera at index {idx}")
                        cap = temp_cap
                        camera_index = idx
                        break
                    temp_cap.release()
    
    if not cap.isOpened():
        print("Error: Could not find any working camera")
        return
    
    # Optimize camera settings for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request higher frame rate
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPG codec
    
    # Check what settings were actually applied
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"HDMI Webcam initialized at index {camera_index}")
    print(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
    
    webcam_connected.set()  # Signal that webcam is connected
    
    # Initialize hand tracking with optimized parameters for speed
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,  # Lower for faster detection
        min_tracking_confidence=0.5,   # Lower for faster tracking
        model_complexity=0) as hands:  # Lower complexity for faster processing
        
        frame_count = 0
        last_frame_time = time.time()
        fps_counter = 0
        fps = 0
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from HDMI webcam, retrying...")
                time.sleep(0.1)  # Shorter delay
                continue
            
            # Calculate FPS
            current_time = time.time()
            fps_counter += 1
            if current_time - last_frame_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_frame_time = current_time
            
            frame_count += 1
            
            # Store frame for potential saving (less frequently)
            if frame_count % 60 == 0:
                if webcam_frames.full():
                    webcam_frames.get()  # Remove oldest frame if queue is full
                webcam_frames.put((time.time(), frame.copy()))
            
            # Skip heavy preprocessing for speed - simple resize if needed
            if frame.shape[0] > 480 or frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480))
            
            # Convert to RGB for MediaPipe (required)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            # Set writeable=False for performance
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True
            
            # Draw hand landmarks
            annotated_frame = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)
                    
                    # Extract hand landmark data
                    timestamp = time.time()
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        hand_data = {
                            'source': 'hdmi_webcam',
                            'timestamp': timestamp,
                            'hand_idx': hand_idx
                        }
                        
                        # Store all 21 landmarks
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            hand_data[f'landmark_{i}_x'] = landmark.x
                            hand_data[f'landmark_{i}_y'] = landmark.y
                            hand_data[f'landmark_{i}_z'] = landmark.z
                            
                        webcam_hand_data.put(hand_data)
            
            # Add text to show it's active and display FPS
            cv2.putText(annotated_frame, f"HDMI Webcam - FPS: {fps}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                      
            # Display frame
            cv2.imshow('HDMI Webcam Hand Tracking', annotated_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                stop_event.set()
                break
                
    # Clean up
    cap.release()
    webcam_connected.clear()

# Function to process frames from RealSense
def process_realsense():
    # Configure RealSense pipeline with optimized settings
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream with lower resolution for speed
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    
    # Try to start streaming
    try:
        pipeline_profile = pipeline.start(config)
        print("RealSense camera initialized successfully")
        realsense_connected.set()  # Signal that RealSense is connected
        
        # Get device info
        device = pipeline_profile.get_device()
        print(f"Using RealSense: {device.get_info(rs.camera_info.name)}")
        
        # Configure for faster processing
        sensors = device.query_sensors()
        for sensor in sensors:
            if sensor.is_color_sensor():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)  # Smaller queue for lower latency
        
        # Initialize hand tracking with optimized parameters
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Lower threshold for faster detection
            min_tracking_confidence=0.5,   # Lower threshold for faster tracking
            model_complexity=0) as hands:  # Lower complexity for faster processing
            
            frame_count = 0
            fps_counter = 0
            fps = 0
            last_frame_time = time.time()
            
            while not stop_event.is_set():
                try:
                    # Wait for frame with shorter timeout
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        print("Warning: No valid color frame from RealSense, continuing...")
                        time.sleep(0.01)
                        continue
                    
                    # Calculate FPS
                    current_time = time.time()
                    fps_counter += 1
                    if current_time - last_frame_time >= 1.0:
                        fps = fps_counter
                        fps_counter = 0
                        last_frame_time = current_time
                    
                    frame_count += 1
                    
                    # Convert to numpy array
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Store frame for potential saving (less frequently)
                    if frame_count % 60 == 0:
                        if realsense_frames.full():
                            realsense_frames.get()
                        realsense_frames.put((time.time(), color_image.copy()))
                    
                    # Skip heavy preprocessing for speed
                    
                    # Process with MediaPipe
                    color_image.flags.writeable = False
                    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    color_image.flags.writeable = True
                    
                    # Draw hand landmarks
                    annotated_image = color_image.copy()
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS)
                            
                            # Extract hand landmark data
                            timestamp = time.time()
                            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                hand_data = {
                                    'source': 'realsense',
                                    'timestamp': timestamp,
                                    'hand_idx': hand_idx
                                }
                                
                                # Store all 21 landmarks
                                for i, landmark in enumerate(hand_landmarks.landmark):
                                    hand_data[f'landmark_{i}_x'] = landmark.x
                                    hand_data[f'landmark_{i}_y'] = landmark.y
                                    hand_data[f'landmark_{i}_z'] = landmark.z
                                    
                                realsense_hand_data.put(hand_data)
                    
                    # Add text to show it's active and FPS
                    cv2.putText(annotated_image, f"RealSense - FPS: {fps}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow('RealSense Hand Tracking', annotated_image)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        stop_event.set()
                        break
                        
                except RuntimeError as e:
                    print(f"Error getting RealSense frame: {e}")
                    # Handle specific errors for RealSense
                    if "timeout" in str(e).lower():
                        print("Timeout error - resetting pipeline")
                        try:
                            pipeline.stop()
                            time.sleep(0.5)
                            pipeline.start(config)
                        except Exception as restart_error:
                            print(f"Failed to restart RealSense: {restart_error}")
                    time.sleep(0.1)  # Short delay
                
    except Exception as e:
        print(f"RealSense initialization error: {e}")
    finally:
        # Stop streaming
        try:
            pipeline.stop()
        except:
            pass
        realsense_connected.clear()

# Function to save data periodically
def save_data():
    all_webcam_data = []
    all_realsense_data = []
    last_save_time = time.time()
    
    while not stop_event.is_set() or not (webcam_hand_data.empty() and realsense_hand_data.empty()):
        current_time = time.time()
        
        # Get webcam data
        while not webcam_hand_data.empty():
            all_webcam_data.append(webcam_hand_data.get())
            
        # Get realsense data
        while not realsense_hand_data.empty():
            all_realsense_data.append(realsense_hand_data.get())
        
        # Save data every 10 seconds if there's new data
        if (current_time - last_save_time >= 10) and (all_webcam_data or all_realsense_data):
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if all_webcam_data:
                webcam_df = pd.DataFrame(all_webcam_data)
                webcam_df.to_csv(f"{save_dir}/webcam_hand_data_{timestamp_str}.csv", index=False)
                print(f"Saved {len(all_webcam_data)} webcam hand tracking records")
                all_webcam_data = []
                
            if all_realsense_data:
                realsense_df = pd.DataFrame(all_realsense_data)
                realsense_df.to_csv(f"{save_dir}/realsense_hand_data_{timestamp_str}.csv", index=False)
                print(f"Saved {len(all_realsense_data)} RealSense hand tracking records")
                all_realsense_data = []
                
            last_save_time = current_time
        
        # Sleep a short time to avoid busy waiting
        time.sleep(0.1)
    
    # Final save before exit
    if all_webcam_data:
        webcam_df = pd.DataFrame(all_webcam_data)
        webcam_df.to_csv(f"{save_dir}/webcam_hand_data_final.csv", index=False)
        
    if all_realsense_data:
        realsense_df = pd.DataFrame(all_realsense_data)
        realsense_df.to_csv(f"{save_dir}/realsense_hand_data_final.csv", index=False)
    
    print("All data saved successfully")

# Function to monitor system status
def monitor_system():
    while not stop_event.is_set():
        webcam_status = "CONNECTED" if webcam_connected.is_set() else "DISCONNECTED"
        realsense_status = "CONNECTED" if realsense_connected.is_set() else "DISCONNECTED"
        
        status_image = np.zeros((100, 600, 3), dtype=np.uint8)
        
        # Add webcam status with color
        webcam_color = (0, 255, 0) if webcam_connected.is_set() else (0, 0, 255)
        cv2.putText(status_image, f"HDMI Webcam: {webcam_status}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, webcam_color, 2)
        
        # Add realsense status with color
        realsense_color = (0, 255, 0) if realsense_connected.is_set() else (0, 0, 255)
        cv2.putText(status_image, f"RealSense: {realsense_status}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, realsense_color, 2)
                  
        # Add exit instruction
        cv2.putText(status_image, "Press 'q' to exit", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('System Status', status_image)
        key = cv2.waitKey(500)  # Update more frequently
        if key & 0xFF == ord('q'):
            stop_event.set()
            break

# Main execution
if __name__ == "__main__":
    try:
        print("Starting Dual Camera Hand Tracking")
        print("Searching for available cameras...")
        
        # List available cameras
        available_cameras = list_cameras()
        if available_cameras:
            print("Available cameras:")
            for camera in available_cameras:
                print(f"  Index {camera['index']}: {camera['name']} - {camera['resolution']} @ {camera['fps']}fps")
            
            # Allow user to select HDMI camera index
            hdmi_index = 1  # Default to 1 since that's often the external camera
            
            user_input = input(f"Enter the index of your HDMI camera (default: {hdmi_index}): ")
            if user_input.strip() and user_input.isdigit():
                hdmi_index = int(user_input)
                
            print(f"Using camera index {hdmi_index} for HDMI webcam")
            print("Press 'q' in any window to stop tracking")
            
            # Start threads
            realsense_thread = threading.Thread(target=process_realsense)
            webcam_thread = threading.Thread(target=process_webcam, args=(hdmi_index,))
            data_save_thread = threading.Thread(target=save_data)
            monitor_thread = threading.Thread(target=monitor_system)
            
            realsense_thread.daemon = True  # Make threads daemon for better cleanup
            webcam_thread.daemon = True
            data_save_thread.daemon = True
            monitor_thread.daemon = True
            
            realsense_thread.start()
            webcam_thread.start()
            data_save_thread.start()
            monitor_thread.start()
            
            # Instead of just joining threads, use a main loop to handle keyboard interrupts
            try:
                while not stop_event.is_set():
                    time.sleep(0.1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
            except KeyboardInterrupt:
                print("Keyboard interrupt detected, stopping program...")
                stop_event.set()
            
        else:
            print("No cameras detected. Please check your camera connections.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        # Ensure stop event is set
        stop_event.set()
        print("Waiting for threads to complete...")
        time.sleep(1)  # Give threads time to clean up
        
        # Clean up
        cv2.destroyAllWindows()
        print("Program completed.")
