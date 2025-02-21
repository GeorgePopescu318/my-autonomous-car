#!/usr/bin/env python3
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder, Quality
import time
import cv2
def main():
    # Create a Picamera2 instance.
    
    # Configure the camera.
    # When using the MJPEG encoder, try configuring the main stream with a format of "MJPEG".
    # (Note: Available formats may vary with your hardware and driver.)
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(raw={"format": 'SBGGR12_CSI2P', "size": (1000, 1000)}))
    picam2.start()
    print("start")

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    result_T = cv2.VideoWriter('test.mp4', fourcc, 20.0, (640,  480))

    

    try:
        while True:
            frame = picam2.capture_array()
            b, g1,g2, r = cv2.split(frame)
            alteredFrame = cv2.merge([b,g1,g2])
            test = cv2.cvtColor(alteredFrame,cv2.COLOR_BGR2RGB)
            result_T.write(test)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping recording...")
    finally:
        result_T.release()
        cv2.destroyAllWindows()
        picam2.stop_recording()
        picam2.stop()
        print(f"Recording complete. Video saved as ")

if __name__ == '__main__':
    main()
