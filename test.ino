#include <AFMotor.h>

// 모터 객체 선언 (M1, M2 포트 사용)
AF_DCMotor motor1(1);
AF_DCMotor motor2(2);

// 명령어 버퍼
String command = "";
char commandType = 'S';  // F: Forward, L: Left+Gas, R: Right+Gas, S: Stop, X: Brake
int speedValue = 0;
const unsigned long COMMAND_DURATION_MS = 300;  // 최소 명령 지속 시간: 100ms (0.1초)
bool commandActive = false;
unsigned long commandStartTime = 0;

// 기본 속도 설정
const int DEFAULT_SPEED = 250;
const int TURN_SPEED_HIGH = 220;
const int TURN_SPEED_LOW = 50;

void setup() {
  // 시리얼 통신 초기화
  Serial.begin(9600);
  
  // 모터 초기화
  motor1.setSpeed(0);
  motor2.setSpeed(0);
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  
  Serial.println("RC Car Ready! (CarRacing Compatible)");
  Serial.println("Actions: 0=Stop, 1=Right+Gas, 2=Left+Gas, 3=Gas, 4=Brake");
}

void loop() {
  // 시리얼 데이터가 있으면 읽기
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();  // 공백 제거
    
    if (command.length() > 0) {
      // 이산 액션 명령어 지원 (0-4)
      // 형식 1: "A0", "A1" (A 접두사)
      // 형식 2: "0", "1" (숫자만)
      if (command.charAt(0) == 'A') {
        // 이산 액션 모드 (A 접두사)
        int actionNum = command.substring(1).toInt();
        if (actionNum >= 0 && actionNum <= 4) {
          executeDiscreteAction(actionNum);
          // 정지 액션(0, 4)은 자동 정지 타이머 비활성화
          if (actionNum == 0 || actionNum == 4) {
            commandActive = false;
          } else {
            commandActive = true;
            commandStartTime = millis();
          }
        } else {
          Serial.println("Invalid action number (0-4)");
        }
      } else if (isDigit(command.charAt(0))) {
        // 숫자만 오는 경우 (0-4 직접 입력)
        int actionNum = command.toInt();
        if (actionNum >= 0 && actionNum <= 4) {
          executeDiscreteAction(actionNum);
          // 정지 액션(0, 4)은 자동 정지 타이머 비활성화
          if (actionNum == 0 || actionNum == 4) {
            commandActive = false;
          } else {
            commandActive = true;
            commandStartTime = millis();
          }
        } else {
          Serial.println("Invalid action number (0-4)");
        }
      } else {
        // 기존 명령어 형식: [F/L/R/S/X][속도]
        commandType = command.charAt(0);
        
        // 속도 값 파싱
        if (command.length() > 1) {
          String params = command.substring(1);
          speedValue = params.toInt();
          speedValue = constrain(speedValue, 0, 255);
        } else {
          speedValue = DEFAULT_SPEED;
        }

        executeCommand(commandType, speedValue);

        if (commandType == 'S' || commandType == 'X') {
          commandActive = false;
        } else {
          commandActive = true;
          commandStartTime = millis();
        }
      }

      // 디버깅 출력
      Serial.print("Cmd: ");
      Serial.println(command);
    }
  }

  // 자동 정지 (안전 기능)
  if (commandActive && millis() - commandStartTime >= COMMAND_DURATION_MS) {
    executeCommand('S', 0);
    commandActive = false;
  }
}

// 이산 액션 실행 (CarRacing-v3 호환)
// RC Car에서는 action 0(coast)과 action 4(brake) 모두 정지로 처리
void executeDiscreteAction(int action) {
  switch(action) {
    case 0:  // 정지 (do nothing / coast) → RC Car: 정지
    case 4:  // 브레이크 (brake) → RC Car: 정지 (동일하게 처리)
      motor1.setSpeed(0);
      motor2.setSpeed(0);
      motor1.run(RELEASE);
      motor2.run(RELEASE);
      if (action == 0) {
        Serial.println("Action 0: Stop (Coast)");
      } else {
        Serial.println("Action 4: Stop (Brake)");
      }
      break;
      
    case 1:  // 우회전 + 가스 (steer right + gas)
      motor1.setSpeed(TURN_SPEED_HIGH);  // 왼쪽 바퀴 빠르게
      motor2.setSpeed(TURN_SPEED_LOW);   // 오른쪽 바퀴 느리게
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Action 1: Right + Gas");
      break;
      
    case 2:  // 좌회전 + 가스 (steer left + gas)
      motor1.setSpeed(TURN_SPEED_LOW);   // 왼쪽 바퀴 느리게
      motor2.setSpeed(TURN_SPEED_HIGH);  // 오른쪽 바퀴 빠르게
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Action 2: Left + Gas");
      break;
      
    case 3:  // 직진 가스 (gas only)
      motor1.setSpeed(DEFAULT_SPEED);
      motor2.setSpeed(DEFAULT_SPEED);
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Action 3: Gas (Forward)");
      break;
      
    default:
      Serial.println("Unknown action");
      break;
  }
}

// 레거시 명령어 실행 (후진 제거됨)
void executeCommand(char cmd, int speed) {
  switch(cmd) {
    case 'F':  // Forward (직진 + 가스)
      motor1.setSpeed(speed);
      motor2.setSpeed(speed);
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Forward");
      break;
      
    case 'L':  // Left + Gas (좌회전 + 가스)
      motor1.setSpeed(speed / 2);
      motor2.setSpeed(speed);
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Left + Gas");
      break;
      
    case 'R':  // Right + Gas (우회전 + 가스)
      motor1.setSpeed(speed);
      motor2.setSpeed(speed / 2);
      motor1.run(FORWARD);
      motor2.run(FORWARD);
      Serial.println("Right + Gas");
      break;
      
    case 'S':  // Stop (코스팅/정지)
      motor1.setSpeed(0);
      motor2.setSpeed(0);
      motor1.run(RELEASE);
      motor2.run(RELEASE);
      Serial.println("Stop");
      break;
      
    case 'X':  // Brake → Stop (RC Car에서는 동일)
      motor1.setSpeed(0);
      motor2.setSpeed(0);
      motor1.run(RELEASE);
      motor2.run(RELEASE);
      Serial.println("Brake -> Stop");
      break;
      
    default:
      Serial.println("Unknown command");
      break;
  }
}
