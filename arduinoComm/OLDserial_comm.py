import serial
import time

# Configure the serial connection (update '/dev/ttyACM0' as needed)
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Wait for the connection to initialize

def send_motor_command(motor_id, direction, speed_percent):

    if motor_id not in [1, 2]:
        print("Invalid value")
        return 0
    if direction not in ['F', 'B']:
        print("Invalid value")
        return 0
    if not (0 <= speed_percent <= 100):
        print("Invalid value")
        return 0

    command = f"{motor_id},{direction},{speed_percent}\n"
    ser.write(command.encode('utf-8'))
    print(f"Sent command: {command.strip()}")
    
try:
    while True:
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino Response: {response}")

        motor = int(input("Enter Motor ID (1 or 2): "))
        direction = input("Enter Direction (F/B): ").upper()
        speed = int(input("Enter Speed (0-100): "))
        
        send_motor_command(motor, direction, speed)
        time.sleep(0.1)  # Short delay between comman
except KeyboardInterrupt:
    print("\nExiting program.")
finally:
    ser.close()
