from picamera2 import Picamera2

# Create the Picamera2 instance
picam2 = Picamera2()

# Configure the camera for still capture (which does not allocate a preview DRM plane)
picam2.configure(picam2.create_still_configuration(main={"size": (1000, 1000), "format": "RGB888"}))
picam2.set_controls({"ScalerCrop": (0, 0, 3280, 2464)})
# Start the camera, capture the image, and stop the camera
picam2.start()
picam2.capture_file("test.jpg")
picam2.stop()