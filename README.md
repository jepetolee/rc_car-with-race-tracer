# RC Car 제어 시스템

Arduino와 Python을 사용한 시리얼 통신 기반 RC 카 제어 시스템

## 시스템 구성

- **Arduino**: AFMotor 라이브러리를 사용하여 2개의 DC 모터 제어
- **Python**: pyserial을 사용하여 시리얼 통신으로 명령 전송

## 하드웨어 연결

- Motor 1: M1 포트 (왼쪽 모터)
- Motor 2: M2 포트 (오른쪽 모터)
- 시리얼 통신: USB 케이블로 Arduino와 컴퓨터 연결

## 명령어 체계

Arduino는 다음과 같은 명령어를 받아 처리합니다:

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `F[속도]` | 전진 | `F255` (최대 속도로 전진) |
| `B[속도]` | 후진 | `B200` (속도 200으로 후진) |
| `L[속도]` | 좌회전 | `L150` (속도 150으로 좌회전) |
| `R[속도]` | 우회전 | `R180` (속도 180으로 우회전) |
| `S` | 정지 | `S` (모터 정지) |

- 속도 범위: 0-255 (PWM 값)
- 명령어는 개행문자(`\n`)로 종료

## 설치 방법

### 1. Arduino 설정

1. Arduino IDE 설치
2. AFMotor 라이브러리 설치:
   - Arduino IDE > 스케치 > 라이브러리 포함하기 > 라이브러리 관리
   - "Adafruit Motor Shield" 검색 후 설치
3. `test.ino` 파일을 Arduino에 업로드
4. 시리얼 모니터로 "RC Car Ready!" 메시지 확인

### 2. Python 환경 설정

```bash
# pyserial 설치
pip install pyserial

# 스크립트 실행 권한 부여 (Linux)
chmod +x rc_car_controller.py
```

### 3. 시리얼 포트 확인

**Linux:**
```bash
ls /dev/tty* | grep -E "(USB|ACM)"
# 일반적으로 /dev/ttyUSB0 또는 /dev/ttyACM0

# 권한 설정 (필요시)
sudo chmod 666 /dev/ttyUSB0
# 또는 사용자를 dialout 그룹에 추가
sudo usermod -a -G dialout $USER
```

**Windows:**
- 장치 관리자에서 COM 포트 확인 (예: COM3, COM4)

## 사용 방법

### 1. 인터랙티브 모드 (키보드 제어)

```bash
python rc_car_controller.py --port /dev/ttyUSB0 --mode interactive
```

키보드 명령어:
- `w`: 전진
- `s`: 후진
- `a`: 좌회전
- `d`: 우회전
- `x`: 정지
- `q`: 종료
- `speed [값]`: 속도 변경 (예: `speed 150`)

### 2. 데모 모드 (자동 테스트)

```bash
python rc_car_controller.py --port /dev/ttyUSB0 --mode demo
```

자동으로 전진, 후진, 좌회전, 우회전을 순차적으로 테스트합니다.

### 3. Python 코드에서 직접 사용

```python
from rc_car_controller import RCCarController

# RC Car 초기화
car = RCCarController(port='/dev/ttyUSB0', baudrate=9600)

# 명령 실행
car.forward(200)      # 속도 200으로 전진
time.sleep(2)         # 2초 대기
car.stop()            # 정지

car.left(150)         # 속도 150으로 좌회전
time.sleep(1)
car.stop()

# 종료
car.close()
```

## 모터 제어 로직

### 전진 (Forward)
- Motor 1: FORWARD, 속도 100%
- Motor 2: FORWARD, 속도 100%

### 후진 (Backward)
- Motor 1: BACKWARD, 속도 100%
- Motor 2: BACKWARD, 속도 100%

### 좌회전 (Left)
- Motor 1: FORWARD, 속도 50%
- Motor 2: FORWARD, 속도 100%

### 우회전 (Right)
- Motor 1: FORWARD, 속도 100%
- Motor 2: FORWARD, 속도 50%

### 정지 (Stop)
- Motor 1: RELEASE
- Motor 2: RELEASE

## 문제 해결

### 1. 시리얼 포트를 찾을 수 없음
```bash
# Linux에서 포트 확인
ls /dev/tty* | grep -E "(USB|ACM)"

# 권한 문제 해결
sudo chmod 666 /dev/ttyUSB0
```

### 2. Arduino가 응답하지 않음
- Arduino IDE의 시리얼 모니터가 닫혀있는지 확인
- Arduino를 리셋 후 2초 대기
- 보드레이트 확인 (9600)

### 3. 모터가 움직이지 않음
- 모터 드라이버 전원 연결 확인
- 모터 포트 번호 확인 (M1, M2)
- 배터리 전압 확인

### 4. Python 모듈 없음
```bash
pip install pyserial
```

## 확장 아이디어

1. **속도 제어 개선**: 점진적 가속/감속 구현
2. **센서 통합**: 초음파 센서로 장애물 회피
3. **원격 제어**: WiFi/Bluetooth 모듈 추가
4. **자율 주행**: 카메라 + AI 모델 통합
5. **웹 인터페이스**: Flask/Django로 웹 컨트롤러 제작

## 라이센스

이 프로젝트는 교육 목적으로 자유롭게 사용할 수 있습니다.


