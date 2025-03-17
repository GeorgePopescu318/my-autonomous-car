import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder, Quality
from get_middle_of_road import get_middle
import sendCommandsArduino as sca
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"format": 'SBGGR12_CSI2P', "size": (1000, 1000)}))
picam2.start()
print("start")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
result_T = cv2.VideoWriter('test.mp4', fourcc, 24.0, (640,  480))

maxForwardL = 255
baseForwardL = 215
baseForwardR = 255

lastMiddlePoint = 0
realMiddlePoint = 305 #old 320
lowPassFilter = 5

kP = 1.1

kD = 0.1
lastError = 0

lastValueLeft = 0
lastValueRight = 0
try:
    while True:
        frame = picam2.capture_array()
        b, g1,g2, r = cv2.split(frame)
        alteredFrame = cv2.merge([b,g1,g2])
        afterConversionFrame = cv2.cvtColor(alteredFrame,cv2.COLOR_BGR2RGB)
        trueMiddle = get_middle(afterConversionFrame,True)
        if len(trueMiddle)!= 0:
            cv2.circle(afterConversionFrame, (trueMiddle[len(trueMiddle)//2][0][0],trueMiddle[len(trueMiddle)//2][0][1]), 5, (0,0,255), -1)
            cv2.circle(afterConversionFrame, (realMiddlePoint,trueMiddle[len(trueMiddle)//2][0][1]), 5, (255,0,0), -1)
            lastMiddlePoint = trueMiddle[len(trueMiddle)//2+1][0][0]
            print(realMiddlePoint - trueMiddle[len(trueMiddle)//2][0][0])
        else:
            print(realMiddlePoint - lastMiddlePoint)

        error = realMiddlePoint - lastMiddlePoint

        finalError = kP * error + (kD * (np.abs(error - lastError)))


        if np.abs(finalError) > lowPassFilter:
            if finalError < 0:
                if baseForwardR - np.abs((finalError)) > 0:
                    scaledValue = baseForwardR - np.abs((finalError))
                    sca.send_motor_command("R","F",scaledValue)
                else:
                    scaledValue = np.abs(baseForwardR - np.abs((finalError)))
                    sca.send_motor_command("R","B",scaledValue)
                sca.send_motor_command("L","F",maxForwardL)

                lastValueLeft = baseForwardL
                lastValueRight = baseForwardR - scaledValue
            elif finalError > 0:
                if baseForwardL - np.abs((finalError)) > 0:
                    scaledValue = baseForwardL - np.abs((finalError))
                    sca.send_motor_command("L","F", scaledValue)
                else:
                    scaledValue = np.abs(baseForwardL - np.abs((finalError)))
                    sca.send_motor_command("L","B", scaledValue)
                sca.send_motor_command("R","F",baseForwardR)

                lastValueLeft = baseForwardL - scaledValue
                lastValueRight = baseForwardR
        elif lastValueLeft != baseForwardL and lastValueRight != baseForwardR:
            sca.send_motor_command("L","F",baseForwardL)
            sca.send_motor_command("R","F",baseForwardR)

            lastValueLeft = baseForwardL
            lastValueRight = baseForwardR

        # print(trueMiddle[len(trueMiddle)//2][0][1] - frame.shape[1])
        result_T.write(afterConversionFrame)

        lastError = error

except KeyboardInterrupt:
    print("\nCtrl+C detected. Stopping recording...")
finally:
    result_T.release()
    cv2.destroyAllWindows()
    picam2.stop_recording()
    picam2.stop()
    sca.send_motor_command("B","F",0)
    print(f"Recording complete. Video saved as ")