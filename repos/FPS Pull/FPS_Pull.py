# Add this to your imports at the top if not already present
import pandas as pd
from datetime import datetime
import os
import time

# Add this after your other initializations, but before the main loop
# Initialize FPS tracking data
fps_data = []
fps_log_interval = 1.0  # Log FPS every second
last_fps_log_time = time.time()
fps_prev_time_cam1 = time.time()
fps_prev_time_realsense = time.time()
cam1_fps = 0
realsense_fps = 0

# Inside your main loop, modify or replace your current FPS calculation:
while True:
    # Capture and process frame from regular camera (Camera 1)
    ret1, frame1 = cap1.read()
    
    # Calculate FPS for Camera 1
    current_time_cam1 = time.time()
    time_diff_cam1 = current_time_cam1 - fps_prev_time_cam1
    if time_diff_cam1 > 0:
        cam1_fps = 1.0 / time_diff_cam1
    fps_prev_time_cam1 = current_time_cam1
    
    # Get frames from RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    # Calculate FPS for RealSense
    current_time_realsense = time.time()
    time_diff_realsense = current_time_realsense - fps_prev_time_realsense
    if time_diff_realsense > 0:
        realsense_fps = 1.0 / time_diff_realsense
    fps_prev_time_realsense = current_time_realsense
    
    # Process frames and handle hand detection as in your original code
    
    # Log FPS data at regular intervals
    current_time = time.time()
    if current_time - last_fps_log_time >= fps_log_interval:
        timestamp = current_time - start_time
        fps_data.append([timestamp, "Camera1", cam1_fps])
        fps_data.append([timestamp, "RealSense", realsense_fps])
        last_fps_log_time = current_time
    
    # Display FPS on frames
    cv2.putText(frame1, f'FPS: {int(cam1_fps)}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    cv2.putText(frame2, f'FPS: {int(realsense_fps)}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    
    # Rest of your loop (displaying frames, checking for key press, etc.)

# After the main loop, add this code to save the FPS data
if save_data:
    try:
        # Create a filename with username and timestamp for FPS data
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fps_filename = f"camera_fps_{user_name}_{timestamp_str}.xlsx"
        
        # Create 'data' directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Convert FPS data to DataFrame
        fps_df = pd.DataFrame(fps_data, columns=["Time", "Camera", "FPS"])
        
        # Save FPS data to separate Excel file
        fps_filepath = os.path.join('data', fps_filename)
        fps_df.to_excel(fps_filepath, index=False)
        print(f"FPS data saved successfully to '{fps_filepath}'")
        print(f"Total FPS measurements: {len(fps_data)//2}")
    except Exception as e:
        print(f"Error saving FPS data: {e}")