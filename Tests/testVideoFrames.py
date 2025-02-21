import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
vidcap = cv2.VideoCapture("NewCameraRodNewTrack.mp4")
success, image = vidcap.read()

while success:

    success, frame = vidcap.read()
    cv2.imshow("Original", frame)
    if cv2.waitKey(10) == 27:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break