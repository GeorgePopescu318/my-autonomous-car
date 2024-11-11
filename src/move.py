import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
import laneDetectionSend as LD



CWD_PATH = os.getcwd()
def readVideo():

    # Read input video from current working directory
    # inpImage = cv2.VideoCapture(os.path.join(CWD_PATH, 'drive.mp4'))
    # frame = cv2.resize(inpImage,(640,480))
    inpImage = cv2.VideoCapture(os.path.join(CWD_PATH, '1.mp4'))

    return inpImage

image = readVideo()

while True:
    _, frame = image.read()

    print(LD.detectLane(frame))

    if cv2.waitKey(1) == 13:
        break

image.release()
cv2.destroyAllWindows()