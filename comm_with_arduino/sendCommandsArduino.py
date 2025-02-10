import serial
import time

# Configure the serial connection (update '/dev/ttyACM0' as needed)
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0)  # Remove timeout
time.sleep(2)  # Wait for the connection to initialize

# Track the previous command
old_command = None

def send_motor_command(motorId, direction, speedPercent):
    global old_command

    # Build the new command
    new_command = f"{motorId},{direction},{speedPercent}\n"

    # Only send if different from the last command
    if new_command != old_command:
        ser.write(new_command.encode('utf-8'))
        print(f"Sent command: {new_command.strip()}")
        old_command = new_command  # Update the old command
