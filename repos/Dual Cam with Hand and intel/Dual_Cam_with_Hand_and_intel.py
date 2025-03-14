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
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize data structures
webcam_hand_data = Queue()
realsense_hand_data = Queue()
webcam_frames = Queue(maxsize=5)  # Store some frames for saving
realsense_frames = Queue(maxsize=5)

# Global flags
stop_event = threading.Event()
realsense_connected = threading.Event()
webcam_connected = threading.Event()

# Function to process frames from HDMI webcam
def process_webcam():
    # Initialize webcam - try different indices if the default doesn't work
    # For HDMI webcams, sometimes the index is 1 or higher
    webcam_indices = [0, 1, 2, 3]  # Try these indices
    cap = None
    
    for idx in webcam_indices:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            ret, test_frame = temp_cap.read()
            if ret:
                print(f"HDMI Webcam found at index {idx}")
                cap = temp_cap
                break
            temp_cap.release()
    
    if cap is None:
        print("Error: Could not find HDMI webcam. Please check connection.")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    webcam_connected.set()  # Signal that webcam is connected
    
    # Initialize hand tracking
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        frame_count = 0
        last_save_time = time.time()
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from HDMI webcam, retrying...")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # Store frame for potential saving (every 30th frame)
            if frame_count % 30 == 0:
                if webcam_frames.full():
                    webcam_frames.get()  # Remove oldest frame if queue is full
                webcam_frames.put((time.time(), frame.copy()))
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Draw hand landmarks
            annotated_frame = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Extract hand landmark data
                    timestamp = time.time()
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        hand_data = {
                            'source': 'webcam',
                            'timestamp': timestamp,
                            'hand_idx': hand_idx
                        }
                        
                        # Store all 21 landmarks
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            hand_data[f'landmark_{i}_x'] = landmark.x
                            hand_data[f'landmark_{i}_y'] = landmark.y
                            hand_data[f'landmark_{i}_z'] = landmark.z
                            
                        webcam_hand_data.put(hand_data)
            
            # Add text to show it's active
            active_text = "HDMI Webcam ACTIVE"
            cv2.putText(annotated_frame, active_text, (10, 30), 
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
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Set higher timeout for RealSense (previous error was frame timeout)
    try:
        # Start streaming with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pipeline_profile = pipeline.start(config)
                print(f"RealSense camera initialized successfully (Attempt {attempt+1})")
                realsense_connected.set()  # Signal that RealSense is connected
                break
            except RuntimeError as e:
                if attempt < max_retries - 1:
                    print(f"Failed to start RealSense (Attempt {attempt+1}): {e}. Retrying...")
                    time.sleep(2)
                else:
                    print(f"Failed to start RealSense after {max_retries} attempts: {e}")
                    return
        
        # Get device info
        device = pipeline_profile.get_device()
        print(f"Using RealSense: {device.get_info(rs.camera_info.name)}")
        
        # Set higher timeout for RealSense frames
        sensor = device.query_sensors()[0]
        sensor.set_option(rs.option.frames_queue_size, 3)  # Increase frame queue size
        
        # Initialize hand tracking
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            
            consecutive_errors = 0
            max_consecutive_errors = 5
            frame_count = 0
            
            while not stop_event.is_set():
                try:
                    # Wait for a coherent pair of frames with higher timeout
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        print("Warning: No valid color frame from RealSense, continuing...")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"Too many consecutive errors ({consecutive_errors}), restarting RealSense pipeline")
                            pipeline.stop()
                            time.sleep(1)
                            pipeline.start(config)
                            consecutive_errors = 0
                        continue
                    
                    consecutive_errors = 0  # Reset error counter on success
                    frame_count += 1
                    
                    # Convert to numpy array
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Store frame for potential saving (every 30th frame)
                    if frame_count % 30 == 0:
                        if realsense_frames.full():
                            realsense_frames.get()  # Remove oldest frame if queue is full
                        realsense_frames.put((time.time(), color_image.copy()))
                    
                    # Process with MediaPipe
                    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    results = hands.process(color_image_rgb)
                    
                    # Draw hand landmarks
                    annotated_image = color_image.copy()
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            
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
                    
                    # Add text to show it's active
                    active_text = "RealSense ACTIVE"
                    cv2.putText(annotated_image, active_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow('RealSense Hand Tracking', annotated_image)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        stop_event.set()
                        break
                        
                except RuntimeError as e:
                    consecutive_errors += 1
                    print(f"Error getting RealSense frame ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors, restarting RealSense pipeline")
                        try:
                            pipeline.stop()
                            time.sleep(1)
                            pipeline.start(config)
                            consecutive_errors = 0
                        except Exception as restart_error:
                            print(f"Failed to restart RealSense: {restart_error}")
                            time.sleep(2)  # Wait before trying again
                    continue
                
    except Exception as e:
        print(f"Unhandled exception in RealSense thread: {e}")
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
    frame_save_interval = 10  # Save frames every 10 seconds
    
    while not stop_event.is_set() or not (webcam_hand_data.empty() and realsense_hand_data.empty()):
        current_time = time.time()
        
        # Get webcam data
        while not webcam_hand_data.empty():
            all_webcam_data.append(webcam_hand_data.get())
            
        # Get realsense data
        while not realsense_hand_data.empty():
            all_realsense_data.append(realsense_hand_data.get())
        
        # Save data every 5 seconds if there's new data
        if (current_time - last_save_time >= 5) and (all_webcam_data or all_realsense_data):
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
            
        # Save frames periodically
        if current_time - last_save_time >= frame_save_interval:
            # Save webcam frames
            while not webcam_frames.empty():
                timestamp, frame = webcam_frames.get()
                frame_time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"{save_dir}/webcam_frame_{frame_time_str}.jpg", frame)
                
            # Save realsense frames
            while not realsense_frames.empty():
                timestamp, frame = realsense_frames.get()
                frame_time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"{save_dir}/realsense_frame_{frame_time_str}.jpg", frame)
            
        # Sleep a short time to avoid busy waiting
        time.sleep(0.1)
    
    # Final save before exit
    if all_webcam_data:
        webcam_df = pd.DataFrame(all_webcam_data)
        webcam_df.to_csv(f"{save_dir}/webcam_hand_data_final.csv", index=False)
        
    if all_realsense_data:
        realsense_df = pd.DataFrame(all_realsense_data)
        realsense_df.to_csv(f"{save_dir}/realsense_hand_data_final.csv", index=False)
    
    # Save any remaining frames
    while not webcam_frames.empty():
        timestamp, frame = webcam_frames.get()
        frame_time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(f"{save_dir}/webcam_frame_{frame_time_str}.jpg", frame)
        
    while not realsense_frames.empty():
        timestamp, frame = realsense_frames.get()
        frame_time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(f"{save_dir}/realsense_frame_{frame_time_str}.jpg", frame)
        
    print("All data saved successfully")

# Function to monitor system status
def monitor_system():
    while not stop_event.is_set():
        webcam_status = "CONNECTED" if webcam_connected.is_set() else "DISCONNECTED"
        realsense_status = "CONNECTED" if realsense_connected.is_set() else "DISCONNECTED"
        
        status_text = f"Status: HDMI Webcam: {webcam_status} | RealSense: {realsense_status}"
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
        key = cv2.waitKey(1000)
        if key & 0xFF == ord('q'):
            stop_event.set()
            break

# Main execution
if __name__ == "__main__":
    try:
        print("Starting Dual Camera Hand Tracking")
        print("Press 'q' in any window to stop tracking")
        
        # Start threads
        realsense_thread = threading.Thread(target=process_realsense)
        webcam_thread = threading.Thread(target=process_webcam)
        data_save_thread = threading.Thread(target=save_data)
        monitor_thread = threading.Thread(target=monitor_system)
        
        realsense_thread.start()
        webcam_thread.start()
        data_save_thread.start()
        monitor_thread.start()
        
        # Join threads
        realsense_thread.join()
        webcam_thread.join()
        data_save_thread.join()
        monitor_thread.join()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        # Ensure stop event is set
        stop_event.set()
        
        # Clean up
        cv2.destroyAllWindows()
        print("Program completed.")
        print("Press any key to continue...")
        input()