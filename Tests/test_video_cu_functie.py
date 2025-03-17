import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
# sys.path.append("Functionalities")
parent_dir = os.path.abspath("../Functionalities")
sys.path.append(parent_dir)
# from get_middle_of_road import Middle_of_road
import get_middle_of_road as gmor # type: ignore


vidcap = cv2.VideoCapture('1.mp4')
success, frame = vidcap.read()


# -----------------------Get Middle constants---------------------#
gmor = gmor.Middle_of_road()
# ----------------------------------------------------------- ------------
while vidcap.isOpened():
    success, frame = vidcap.read()
    # cv2.imshow("b",frame)
    frame = cv2.resize(frame, (640,480))
    # b, g1,g2, r = cv2.split(frame)
    # alteredFrame = cv2.merge([b,g1,g2])
    # afterConversionFrame = cv2.cvtColor(alteredFrame,cv2.COLOR_BGR2RGB)
    trueMiddle = gmor.get_middle(frame,True)
    if len(trueMiddle) != 0:
        cv2.circle(frame, (int(trueMiddle[len(trueMiddle)//2][0][0]),int(trueMiddle[len(trueMiddle)//2][0][1])), 5, (0,0,255), -1)
    cv2.imshow("a",frame)
    if cv2.waitKey(10) == 27:
    # cv2.imwrite("bird_for_track.jpg",frame)
        vidcap.release()
        cv2.destroyAllWindows()
        break