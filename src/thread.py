import cv2
import numpy as np
import os
import threading
import time
from scipy import optimize
from collections import deque
from picamera2 import Picamera2
from matplotlib import pyplot as plt, cm, colors
import laneDetectionSend as LD
import sendCommandsArduino as SC

# Initialize camera and output configurations
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"format": 'SBGGR12_CSI2P', "size": (1000, 1000)}))
picam2.start()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
result_T = cv2.VideoWriter('trash.avi', fourcc, 20.0, (640,  480))
result_F = cv2.VideoWriter('final.avi', fourcc, 20.0, (640,  480))

# Global variables for shared data
deviation_buffer = deque(maxlen=10)  # buffer of last 10 deviations
buffer_lock = threading.Lock()       # lock to synchronize access to deviation buffer
old_deviation = 0                    # keep track of previous deviation for smoothing

def view_frame(title,frame):
    """ Display the frame in a window. """
    cv2.imshow(title, frame)

def capture_video(thresh, finalImg):
    """ Record the processed video frames. """
    result_F.write(finalImg)
    result_T.write(thresh)

def deviation_thread():
    """ Thread to capture the frame, calculate deviation, and store it in the buffer. """
    global deviation_buffer, buffer_lock
    while True:
        # Capture frame and process lane detection
        frame = picam2.capture_array()
        b, g1, g2, r = cv2.split(frame)
        altered_frame = cv2.merge([b, g1, g2])

        # Detect lane and calculate deviation
        deviation, birdView, thresh, finalImg = LD.detect_lane(altered_frame)
        finalThresh = cv2.merge([thresh, thresh, thresh])

        # Display frame and capture video
        # view_frame("F",finalImg)
        # view_frame("T",thresh)
        capture_video(finalThresh, finalImg)

        # Add deviation to buffer with thread-safe access
        with buffer_lock:
            deviation_buffer.append(deviation)

        # Exit on pressing 'Enter'
        if cv2.waitKey(1) == 13:
            print("Exit")
            with buffer_lock:
                deviation_buffer.append(100)
            break
    # deviation_thread.join(1)
    # steering_thread.join(1) 

def steer(deviation):
    """ Send steer commands to Arduino based on deviation. """
    
    global old_deviation
    # deviation -= 0.03  # Adjust deviation

    kd = 1.5

    print(deviation)

    turn = int((deviation + (deviation - old_deviation) * kd) * 1000 / 2)

    if turn < 0:
        turn *= -1
    turn += 30

    if turn > 100:
        turn = 100

    # print(turn)

    # Determine direction to send to Arduino
    if deviation > 0.060:
        SC.send_motor_command(2, 'F', turn)
    elif deviation < -0.060:
        SC.send_motor_command(2, 'B', turn)
    else:
        SC.send_motor_command(2, 'F', 0)

    old_deviation = deviation

def steering_thread():
    """ Thread to read deviation from buffer and send steering commands with delay. """
    global deviation_buffer, buffer_lock
    SC.send_motor_command(1, 'F', 80)

    while True:
        # Get oldest deviation with thread-safe access
        with buffer_lock:
            if deviation_buffer:
                current_deviation = deviation_buffer.popleft()
                if current_deviation == 100:
                    break
                steer(current_deviation)
                

        # Send steering commands based on deviation
        
        time.sleep(1)  # Short delay to simulate real-time adjustment rate

# Start threads for lane detection and steering
deviation_thread = threading.Thread(target=deviation_thread, daemon=True)
steering_thread = threading.Thread(target=steering_thread, daemon=True)

deviation_thread.start()
steering_thread.start()

# Wait for threads to complete

deviation_thread.join()
steering_thread.join() 

# Release resources
result_T.release()
result_F.release()
cv2.destroyAllWindows()

SC.send_motor_command(1, 'F', 0)
SC.send_motor_command(2, 'F', 0)
