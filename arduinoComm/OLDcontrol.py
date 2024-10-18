from uinput.keyboard import Key, Listener
import sendCommandsArduino as my_serial
inMoving = False
inSteering = False
def on_press(key):
    global inMoving
    global inSteering

    if key == Key.shift:
        inMoving = False
        inSteering = False

    if key == Key.space: 

        my_serial.send_motor_command(1,'B',0)
        my_serial.send_motor_command(2,'B',0)

        inMoving = True
        inSteering = True

    if key == Key.right and inSteering == False:
        inSteering = True
        my_serial.send_motor_command(2,'B',100)
        
    if key == Key.left and inSteering == False:
       inSteering = True
       my_serial.send_motor_command(2,'F',100)

    if key == Key.up and inMoving == False:
        inMoving = True
        my_serial.send_motor_command(1,'F',100)

    if key == Key.down and inSteering == False:
        inMoving = True
        my_serial.send_motor_command(1,'B',100)

def on_release(key):
    global inMoving
    global inSteering

    if key == Key.right or key == Key.left:
        inSteering = False
        my_serial.send_motor_command(2,'B',0)

    if key == Key.up or key == Key.down:
        inMoving = False
        my_serial.send_motor_command(1,'B',0) 

    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
