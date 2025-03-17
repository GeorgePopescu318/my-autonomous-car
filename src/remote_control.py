import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder, Quality
from Functionalities.get_middle_of_road import get_middle

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"format": 'SBGGR12_CSI2P', "size": (1000, 1000)}))
picam2.start()
print("start")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
result_T = cv2.VideoWriter('test.mp4', fourcc, 24.0, (640,  480))

#-----------------------Get Middle constants---------------------#
tl = (133,120)
bl = (0 ,415)
tr = (460,120)
br = (638,415)

l_h = 0
l_s = 0
l_v = 200
u_h = 255
u_s = 50
u_v = 255

lower = np.array([l_h,l_s,l_v])
upper = np.array([u_h,u_s,u_v])

pts1 = np.float32([tl, bl, tr, br]) 
pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 

# Matrix to warp the image for birdseye window
matrix = cv2.getPerspectiveTransform(pts1, pts2)
invMatrix = cv2.getPerspectiveTransform(pts2, pts1)

finalMiddlePoints = 7

singleLaneShift = 315
#-----------------------------------------------------------------------
try:
    while True:
        frame = picam2.capture_array()
        b, g1,g2, r = cv2.split(frame)
        alteredFrame = cv2.merge([b,g1,g2])
        afterConversionFrame = cv2.cvtColor(alteredFrame,cv2.COLOR_BGR2RGB)
        trueMiddle = get_middle(afterConversionFrame,False)
        cv2.circle(frame, (trueMiddle[len(trueMiddle)//2][0][0],trueMiddle[len(trueMiddle)//2][0][1]), 5, (0,0,255), -1)
        result_T.write(afterConversionFrame)
except KeyboardInterrupt:
    print("\nCtrl+C detected. Stopping recording...")
finally:
    result_T.release()
    cv2.destroyAllWindows()
    picam2.stop_recording()
    picam2.stop()
    print(f"Recording complete. Video saved as ")