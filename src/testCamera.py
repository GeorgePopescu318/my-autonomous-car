import numpy as np
import cv2 as cv
from picamera2 import Picamera2

cap = cv.VideoCapture(0)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"format": 'SBGGR12_CSI2P', "size": (1000, 1000)}))
picam2.start()
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
 
while cap.isOpened():
    frame = picam2.capture_array()
    frame = cv.flip(frame, 0)
 
    # write the flipped frame
    out.write(frame)
 
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
 
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()