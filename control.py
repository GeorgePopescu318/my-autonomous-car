import pygame
import sendCommandsArduino as my_serial
import threading

# Initialize Pygame
pygame.init()

# Set up the Pygame window (you can minimize it if needed)
screen = pygame.display.set_mode((100, 100))
pygame.display.set_caption('Raspberry Pi Car Control')

# Flags to check if movement or steering is in progress
inMoving = False
inSteering = False

# Function to handle key press events
def handle_key_event():
    global inMoving, inSteering
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    my_serial.send_motor_command(1, 'B', 0)
                    my_serial.send_motor_command(2, 'B', 0)
                    inMoving = True
                    inSteering = True

                elif event.key == pygame.K_RIGHT and not inSteering:
                    inSteering = True
                    my_serial.send_motor_command(2, 'B', 100)  # Turn right

                elif event.key == pygame.K_LEFT and not inSteering:
                    inSteering = True
                    my_serial.send_motor_command(2, 'F', 100)  # Turn left

                elif event.key == pygame.K_UP and not inMoving:
                    inMoving = True
                    my_serial.send_motor_command(1, 'F', 100)  # Move forward

                elif event.key == pygame.K_DOWN and not inMoving:
                    inMoving = True
                    my_serial.send_motor_command(1, 'B', 100)  # Move backward

                elif event.key == pygame.K_ESCAPE:
                    running = False  # Exit the loop and quit

            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_RIGHT, pygame.K_LEFT]:
                    inSteering = False
                    my_serial.send_motor_command(2, 'B', 0)  # Stop steering

                elif event.key in [pygame.K_UP, pygame.K_DOWN]:
                    inMoving = False
                    my_serial.send_motor_command(1, 'B', 0)  # Stop moving

    pygame.quit()

# Run the key event handler in the main thread
handle_key_event()
