#include <AFMotor.h>

// 모터 객체 선언 (M1, M2 포트 사용)
AF_DCMotor motor1(1);
AF_DCMotor motor2(2);

// 명령어 버퍼
String command = "";
char commandType = 'S';  // F: Forward, B: Backward, L: Left, R: Right, S: Stop
int speedValue = 0;
const unsigned long COMMAND_DURATION_MS = 1000;
bool commandActive = false;
unsigned long commandStartTime = 0;

void setup() {
  // 시리얼 통신 초기화
  Serial.begin(9600);
  
  // 모터 초기화
  motor1.setSpeed(0);
  motor2.setSpeed(0);
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  
  Serial.println("RC Car Ready!");
}

void loop() {
  // 시리얼 데이터가 있으면 읽기
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();  // 공백 제거
    
    if (command.length() > 0) {
      // 명령어 형식: [F/B/L/R/S][속도]
      // 예: F255 (전진, 속도 255), B200 (후진, 속도 200), L150 (좌회전), S (정지)
      commandType = command.charAt(0);
      
      // 속도 값 파싱
      if (command.length() > 1) {
        String params = command.substring(1);
        speedValue = params.toInt();
        // 속도 범위 제한 (0-255)
        speedValue = constrain(speedValue, 0, 255);
      } else {
        speedValue = 0;
      }

      // 명령어 실행
      executeCommand(commandType, speedValue);

      if (commandType == 'S') {
        commandActive = false;
      } else {
        commandActive = true;
        commandStartTime = millis();
      }

      // 디버깅: 받은 명령어 출력
      Serial.print("Command: ");
      Serial.print(commandType);
      Serial.print(", Speed: ");
      Serial.println(speedValue);
    }
  }

  if (commandActive && millis() - commandStartTime >= COMMAND_DURATION_MS) {
    executeCommand('S', 0);
    commandActive = false;
    Serial.print("Auto stop after ");
    Serial.print(COMMAND_DURATION_MS / 1000.0, 2);
    Serial.println(" seconds");
  }
}

void executeCommand(char cmd, int speed) {
  switch(cmd) {
    case 'F':  // Forward (전진)
      motor1.setSpeed(speed);
      motor2.setSpeed(speed);
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Moving Forward");
      break;
      
    case 'B':  // Backward (후진)
      motor1.setSpeed(speed);
      motor2.setSpeed(speed);
      motor1.run(BACKWARD);
      motor2.run(BACKWARD);
      Serial.println("Moving Backward");
      break;
      
    case 'L':  // Left (좌회전)
      // 왼쪽 모터 느리게, 오른쪽 모터 빠르게
      motor1.setSpeed(speed / 2);
      motor2.setSpeed(speed);
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Turning Left");
      break;
      
    case 'R':  // Right (우회전)
      // 왼쪽 모터 빠르게, 오른쪽 모터 느리게
      motor1.setSpeed(speed);
      motor2.setSpeed(speed / 2);
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Turning Right");
      break;
      
    case 'S':  // Stop (정지)
      motor1.setSpeed(0);
      motor2.setSpeed(0);
      motor1.run(RELEASE);
      motor2.run(RELEASE);
      Serial.println("Stopped");
      break;
      
    default:
      Serial.println("Unknown command");
      break;
  }
}


