#include <Arduino.h>
#include <ctype.h>  // For toupper()

// --- Motor Pin Definitions ---

// Left Motor (Motor A) Pins
const int enA = 9;
const int in1 = 7;
const int in2 = 8;

// Right Motor (Motor B) Pins
const int enB = 3;
const int in3 = 6;
const int in4 = 5;

// --- Serial Buffer Settings ---
const byte MAX_BUFFER_SIZE = 32;
char inputBuffer[MAX_BUFFER_SIZE];
byte bufferIndex = 0;

void setup() {
  // Initialize motor control pins as outputs
  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  
  pinMode(enB, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  // Ensure motor driver inputs start low
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  // Initialize serial communication
  Serial.begin(115200);
}

void loop() {
  // Read serial input one character at a time
  if (Serial.available() > 0) {
    char inChar = Serial.read();
    if (inChar == '\n') {
      // End of command: null-terminate the string and process it
      inputBuffer[bufferIndex] = '\0';
      processCommand(inputBuffer);
      // Reset the buffer for the next command
      bufferIndex = 0;
      inputBuffer[0] = '\0';
    } else {
      // Append character if space remains
      if (bufferIndex < MAX_BUFFER_SIZE - 1) {
        inputBuffer[bufferIndex++] = inChar;
      }
    }
  }
}

// Process commands of the form: <MotorSelection>,<Direction>,<Speed>
// MotorSelection: 'L' (left), 'R' (right), or 'B' (both)
// Direction: 'F' (forward) or 'B' (backward)
// Speed: an integer from 0 to 100
void processCommand(char command[32]) {
  char motorSelection;
  char direction;
  int speedPercent;
  
  // Parse the command (spaces are allowed around the commas)
  int parsed = sscanf(command, " %c , %c , %d", &motorSelection, &direction, &speedPercent);
  if (parsed != 3) {
    // Invalid command format; exit function.
    return;
  }
  
  // Convert the characters to uppercase to accept lower- or uppercase letters
  motorSelection = toupper(motorSelection);
  direction = toupper(direction);
  
  // Validate motor selection and direction
  if ((motorSelection != 'L' && motorSelection != 'R' && motorSelection != 'B') ||
      (direction != 'F' && direction != 'B')) {
    return;
  }

  int pwmValue = speedPercent;
  
  // Control the selected motor(s)
  if (motorSelection == 'L' || motorSelection == 'B') {
    controlMotor(in1, in2, enA, direction, pwmValue);
  }
  if (motorSelection == 'R' || motorSelection == 'B') {
    controlMotor(in3, in4, enB, direction, pwmValue);
  }
}

// Controls a motor given two input pins, an enable pin, a direction, and a PWM speed value
void controlMotor(int inPin1, int inPin2, int enPin, char direction, int pwm) {
  if (pwm == 0) {
    // Stop the motor
    digitalWrite(inPin1, LOW);
    digitalWrite(inPin2, LOW);
    analogWrite(enPin, 0);
  } else {
    // Set the motor direction: 'F' for forward, 'B' for backward
    if (direction == 'F') {
      digitalWrite(inPin1, HIGH);
      digitalWrite(inPin2, LOW);
    } else if (direction == 'B') {
      digitalWrite(inPin1, LOW);
      digitalWrite(inPin2, HIGH);
    }
    // Set the motor speed using PWM
    analogWrite(enPin, pwm);
  }
}
