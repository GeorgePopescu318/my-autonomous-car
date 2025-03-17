#!/usr/bin/python3

import socket
import time
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start()
time.sleep(2)
picam2.capture_file("test.jpg")