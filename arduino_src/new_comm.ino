#include <Arduino.h>
#include <Servo.h>
Servo myservo;

const int servoPin = 11;

// Motor A Pins
const int enA = 9;
const int in1 = 7;
const int in2 = 8;

// Motor B Pins
const int enB = 3;
const int in3 = 6;
const int in4 = 5;

int servo_position = 0;

// Serial Buffer Settings
const byte MAX_BUFFER_SIZE = 32;
char inputBuffer[MAX_BUFFER_SIZE];
byte bufferIndex = 0;

void setup() {
  // Initialize motor control pins as outputs
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  // Attach servo
  myservo.attach(servoPin);

  // Initialize serial communication
  Serial.begin(115200);
}

void loop() {
  // Check if data is available on the serial port
  if (Serial.available() > 0) {
    char inChar = Serial.read();
    if (inChar == '\n') {
      inputBuffer[bufferIndex] = '\0'; // Null-terminate the string
      processCommand(inputBuffer);
      bufferIndex = 0; // Reset buffer for the next command
      inputBuffer[0] = '\0';
    } else {
      // Add character to buffer if there's space
      if (bufferIndex < MAX_BUFFER_SIZE - 1) {
        inputBuffer[bufferIndex++] = inChar;
      }
    }
  }
}

// Function to process incoming serial commands
void processCommand(char command[32]) {
  int motorID = 0;
  char direction = ' ';
  int speedPercent = 0;

  // Parse the command: <MotorID>,<Direction>,<Speed>
  int parsed = sscanf(command, "%d,%c,%d", &motorID, &direction, &speedPercent);

  if (parsed != 3) {
    return;
  }

  // Validate parsed values
  if ((motorID != 1 && motorID != 2) ||
      (direction != 'F' && direction != 'B') ||
      (speedPercent < -50 || speedPercent > 50)) { // Adjusted for servo range (-50 to 50)
    return;
  }

  // Control the appropriate motor
  if (motorID == 1) {
    int pwmValue = map(abs(speedPercent), 0, 100, 0, 255);
    controlMotor(in1, in2, enA, direction, pwmValue);
  } else if (motorID == 2) {
    servo_position = mapServoPosition(speedPercent);
    controlServo(servo_position);
  }

}

// Function to control a motor based on direction and speed
void controlMotor(int inPin1, int inPin2, int enPin, char direction, int pwm) {
  if (pwm == 0) {
    // Stop the motor
    digitalWrite(inPin1, LOW);
    digitalWrite(inPin2, LOW);
    analogWrite(enPin, 0);
  } else {
    // Set motor direction
    if (direction == 'F') {
      digitalWrite(inPin1, HIGH);
      digitalWrite(inPin2, LOW);
    } else if (direction == 'B') {
      digitalWrite(inPin1, LOW);
      digitalWrite(inPin2, HIGH);
    }
    // Set motor speed
    analogWrite(enPin, pwm);
  }
}

// Function to control the servo
void controlServo(int position) {
  myservo.write(position);
}

// Function to map speed to servo position
int mapServoPosition(int speedPercent) {
  // Map -50 to 50 to 60 to 120 degrees
  return map(speedPercent, -50, 50, 60, 120);
}
