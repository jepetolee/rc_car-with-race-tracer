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

## Raspberry Pi Camera

### Installation

The camera functionality requires the `picamera2` module, which is the modern camera library for Raspberry Pi OS (Bullseye and later).

**Install picamera2:**
```bash
sudo apt-get update
sudo apt-get install python3-picamera2
```

**Install OpenCV for image processing:**
```bash
pip install opencv-python numpy
```

**Important: Virtual Environment Configuration**

`picamera2` is installed as a system package (via `apt`), not via `pip`. If you're using a virtual environment, you need to configure it to access system site packages:

**Option 1: Recreate virtual environment with system site packages (Recommended)**
```bash
# Deactivate current virtual environment
deactivate

# Remove old virtual environment (if needed)
rm -rf venv  # or your venv directory name

# Create new virtual environment with system site packages
python3 -m venv --system-site-packages venv

# Activate the new virtual environment
source venv/bin/activate

# Verify picamera2 is accessible
python3 -c "import picamera2; print('picamera2 imported successfully')"
```

**Option 2: Use system Python directly (outside virtual environment)**
```bash
# Deactivate virtual environment
deactivate

# Run scripts with system Python
python3 raspberry_pi_camera.py --mode preview
```

**Option 3: Enable system site packages in existing virtual environment**
```bash
# Edit your virtual environment's pyvenv.cfg file
# Change: include-system-site-packages = false
# To:     include-system-site-packages = true

# Or recreate as shown in Option 1
```

**Note:** This codebase uses `picamera2`, which is the standard camera library for modern Raspberry Pi OS. If you're using an older Raspberry Pi OS that only supports the legacy `picamera` module, you would need to use an older version of the code or upgrade your OS.

### Camera Test Script

A standalone camera test script (`raspberry_pi_camera.py`) is provided to test and display the Raspberry Pi camera feed.

#### 1. Video Preview Mode

Display real-time video feed from the camera:

```bash
python raspberry_pi_camera.py --mode preview
```

Options:
- `--resolution WIDTH,HEIGHT`: Set camera resolution (default: 320,320)
- `--framerate FPS`: Set frame rate (default: 30)
- `--grayscale`: Enable grayscale mode
- `--duration SECONDS`: Preview duration (default: infinite)
- `--show-processed`: Also display processed 16x16 image

Controls:
- Press `q` to quit
- Press `s` to save a snapshot

Example:
```bash
# Preview with grayscale and processed image
python raspberry_pi_camera.py --mode preview --grayscale --show-processed

# Preview for 10 seconds
python raspberry_pi_camera.py --mode preview --duration 10
```

#### 2. Single Image Capture

Capture a single image:

```bash
python raspberry_pi_camera.py --mode capture --output image.jpg
```

#### 3. Processed Image Test

Test the image processing pipeline (similar to `rc_car_interface.py`):

```bash
python raspberry_pi_camera.py --mode test --output processed.jpg
```

This mode:
- Captures an image
- Converts to grayscale
- Applies threshold
- Resizes to 16x16
- Displays and optionally saves the result

### Camera Integration

The camera is integrated into the RC Car system through `rc_car_interface.py`:

```python
from rc_car_interface import RC_Car_Interface

# Initialize interface (includes camera)
rc_car = RC_Car_Interface()

# Get processed image (16x16 grayscale)
image = rc_car.get_image_from_camera()

# Use image for AI/ML processing
```

The camera interface:
- Resolution: 320x320
- Grayscale mode enabled
- Output: 16x16 processed binary image
- Compatible with reinforcement learning environments

### Troubleshooting

**Camera not detected:**
```bash
# Check if camera is enabled
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Check camera module
vcgencmd get_camera
# Should return: supported=1 detected=1
```

**Permission errors:**
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Log out and log back in
```

**Import errors:**

*"picamera2 module not found" in virtual environment:*
- `picamera2` is a system package, not installable via `pip`
- If using a virtual environment, recreate it with `--system-site-packages` flag:
  ```bash
  deactivate
  rm -rf venv
  python3 -m venv --system-site-packages venv
  source venv/bin/activate
  ```
- Or use system Python directly (outside virtual environment):
  ```bash
  deactivate
  python3 raspberry_pi_camera.py --mode preview
  ```
- Verify installation: `python3 -c "import picamera2; print(picamera2.__file__)"`

*Other import issues:*
- Ensure you're running on Raspberry Pi (picamera2 only works on Raspberry Pi)
- Check Python version: `python3 --version` (should be 3.7+)
- Install picamera2: `sudo apt install python3-picamera2`
- Ensure camera is enabled: `sudo raspi-config` > Interface Options > Camera > Enable

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


