import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
import laneDetectionSend as LD
import sendCommandsArduino as SC
from picamera2 import Picamera2

<<<<<<< HEAD


CWD_PATH = os.getcwd()
def readVideo():
=======
#/usr/bin/python3 ./your_script.py
>>>>>>> 699f398b3e04821bdc02aba912bf92f6b5703889

#CWD_PATH = os.getcwd()

#image = readVideo()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"format": 'SBGGR12_CSI2P', "size": (1000, 1000)}))
picam2.start()
# cv2.imwrite("pista.jpg", picam2.capture_array())

old_deviation = ""
iteration = 0
right_bound = 0.1
left_bound = -0.1
old_deviation = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
result_T = cv2.VideoWriter('trash.avi', fourcc, 20.0, (640,  480))

result_N = cv2.VideoWriter('normal.avi', fourcc, 20.0, (640,  480))

def view_frame(frame):
    cv2.imshow("",frame)

def capture_video(thresh, finalImg):
    result_N.write(finalImg)
    result_T.write(thresh)

def steer(deviation):
    global old_deviation
    deviation -= 0.02
    turn = int((deviation + (deviation - old_deviation))*1000/2)

    print(deviation)

    if turn < 0:
        turn *= -1
    turn += 30

    if turn > 100:
        turn = 100

    if deviation > 0.060:
        SC.send_motor_command(2, 'F', turn)
    elif deviation < -0.060:
        SC.send_motor_command(2, 'B', turn)
    else:
        SC.send_motor_command(2, 'F', 0)

    old_deviation = deviation

# SC.send_motor_command(1, 'F', 70)
while True:
     
    frame = picam2.capture_array()
    
    b, g1,g2, r = cv2.split(frame)
    alteredFrame = cv2.merge([b,g1,g2])
    
    deviation,birdView,thresh,finalImg = LD.detect_lane(alteredFrame)
    
    finalThresh = cv2.merge([thresh, thresh, thresh])
    # capture_video(finalThresh,finalImg)

    view_frame(finalImg)

    steer(deviation)
    
    if cv2.waitKey(1) == 13:
        break

# video.release()
result_T.release()
result_N.release() 
cv2.destroyAllWindows()