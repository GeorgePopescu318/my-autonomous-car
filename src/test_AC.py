import sendCommandsArduino as sca
import time

baseForwardR = 215
baseForwardL = 255
# for i in range(0,100,10):
sca.send_motor_command("L","F",baseForwardR)
# sca.send_motor_command("L","B",baseForwardR)
time.sleep(1)
sca.send_motor_command("B","F",0)