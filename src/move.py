import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
import laneDetectionSend as LD
import sendCommandsArduino as my_serial
from picamera2 import Picamera2

#/usr/bin/python3 ./your_script.py

#CWD_PATH = os.getcwd()
def readVideo():

    # Read input video from current working directory
    # inpImage = cv2.VideoCapture(os.path.join(CWD_PATH, 'drive.mp4'))
    # frame = cv2.resize(inpImage,(640,480))
    inpImage = cv2.VideoCapture(os.path.join(CWD_PATH, '1.mp4'))

    return inpImage

#image = readVideo()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"format": 'SBGGR12_CSI2P', "size": (1000, 1000)}))
picam2.start()
cv2.imwrite("pista.jpg", picam2.capture_array())

old_direction = ""
iteration = 0
right_bound = 0.1
left_bound = -0.1
old_direction = 0
my_serial.send_motor_command(1, 'F', 70)
while True:
    
    #_, frame = image.read()
    
    frame = picam2.capture_array()
    
    direction = LD.detectLane(frame)
    print(direction)
    # if direction != old_direction:
    #     iteration = 0
    #     old_direction = direction
    #     #my_serial.send_motor_command(2, 'F', 0)
    # else:
    #     iteration+=1
    #     # print(0)
    #     if iteration >= 8:
    #         if direction == "Straight":
    #             # print(1)
    #             my_serial.send_motor_command(2, 'F', 0)
    #         elif direction == "Left Curve":
    #             # print(2)
    #             my_serial.send_motor_command(2, 'F', 100)
    #         elif direction == "Right Curve":
    #             # print(3)
    #             my_serial.send_motor_command(2, 'B', 100)
    # #LD.detectLane(frame)
    
    turn = int((direction + old_direction)*1000/2)

    if turn < 0:
        turn *= -1
    turn += 30

    if turn > 100:
        turn = 100

    if direction > 0.065:
        my_serial.send_motor_command(2, 'F', turn)
    elif direction < -0.065:
        my_serial.send_motor_command(2, 'B', turn)
    else:
        my_serial.send_motor_command(2, 'F', 0)

    old_direction = direction
    
    if cv2.waitKey(1) == 13:
        break

#image.release()
cv2.destroyAllWindows()
