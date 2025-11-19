#!/usr/bin/env python3
"""
RC Car Controller via Serial Communication
Arduino와 시리얼 통신으로 RC 카를 제어하는 Python 스크립트

명령어 형식:
- F[속도]: 전진 (예: F255)
- B[속도]: 후진 (예: B200)
- L[속도]: 좌회전 (예: L150)
- R[속도]: 우회전 (예: R180)
- S: 정지
"""

import serial
import time
import sys


class RCCarController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        """
        RC 카 제어기 초기화
        
        Args:
            port: 시리얼 포트 (Linux: /dev/ttyUSB0, /dev/ttyACM0, Windows: COM3)
            baudrate: 보드레이트 (기본: 9600)
        """
        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Arduino 리셋 대기
            print(f"Connected to {port} at {baudrate} baud")
            
            # Arduino로부터 초기 메시지 읽기
            if self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8').strip()
                print(f"Arduino: {response}")
        except serial.SerialException as e:
            print(f"Error: 시리얼 포트를 열 수 없습니다 - {e}")
            print("사용 가능한 포트를 확인하세요:")
            print("  Linux: /dev/ttyUSB0, /dev/ttyACM0")
            print("  Windows: COM3, COM4, etc.")
            sys.exit(1)
    
    def send_command(self, command):
        """
        Arduino로 명령 전송r
        
        Args:
            command: 명령 문자열 (예: "F255", "S")
        """
        try:
            self.serial.write(f"{command}\n".encode('utf-8'))
            time.sleep(0.1)  # 명령 처리 대기
            
            # Arduino로부터 응답 읽기
            while self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8').strip()
                if response:
                    print(f"Arduino: {response}")
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def forward(self, speed=200):
        """전진 (속도: 0-255)"""
        speed = max(0, min(255, speed))  # 속도 범위 제한
        print(f"전진 - 속도: {speed}")
        self.send_command(f"F{speed}")
    
    def backward(self, speed=200):
        """후진 (속도: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"후진 - 속도: {speed}")
        self.send_command(f"B{speed}")
    
    def left(self, speed=200):
        """좌회전 (속도: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"좌회전 - 속도: {speed}")
        self.send_command(f"L{speed}")
    
    def right(self, speed=200):
        """우회전 (속도: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"우회전 - 속도: {speed}")
        self.send_command(f"R{speed}")
    
    def stop(self):
        """정지"""
        print("정지")
        self.send_command("S")
    
    def close(self):
        """시리얼 포트 닫기"""
        if self.serial.is_open:
            self.stop()  # 종료 전 정지
            self.serial.close()
            print("Connection closed")


def interactive_mode(controller):
    """인터랙티브 모드 - 키보드로 제어"""
    print("\n=== RC Car 인터랙티브 제어 모드 ===")
    print("명령어:")
    print("  w: 전진")
    print("  s: 후진")
    print("  a: 좌회전")
    print("  d: 우회전")
    print("  x: 정지")
    print("  q: 종료")
    print("===================================\n")
    
    speed = 200  # 기본 속도
    
    try:
        while True:
            cmd = input("명령 입력 (w/s/a/d/x/q): ").strip().lower()
            
            if cmd == 'w':
                controller.forward(speed)
            elif cmd == 's':
                controller.backward(speed)
            elif cmd == 'a':
                controller.left(speed)
            elif cmd == 'd':
                controller.right(speed)
            elif cmd == 'x':
                controller.stop()
            elif cmd == 'q':
                print("종료합니다...")
                break
            elif cmd.startswith('speed '):
                # 속도 변경 (예: speed 150)
                try:
                    new_speed = int(cmd.split()[1])
                    speed = max(0, min(255, new_speed))
                    print(f"속도 변경: {speed}")
                except:
                    print("잘못된 속도 값입니다.")
            else:
                print("알 수 없는 명령입니다.")
    except KeyboardInterrupt:
        print("\n종료합니다...")


def demo_mode(controller):
    """데모 모드 - 자동 주행 테스트"""
    print("\n=== RC Car 데모 모드 ===")
    
    print("1. 전진 (3초)")
    controller.forward(200)
    time.sleep(3)
    
    print("2. 정지 (1초)")
    controller.stop()
    time.sleep(1)
    
    print("3. 후진 (2초)")
    controller.backward(150)
    time.sleep(2)
    
    print("4. 정지 (1초)")
    controller.stop()
    time.sleep(1)
    
    print("5. 좌회전 (2초)")
    controller.left(180)
    time.sleep(2)
    
    print("6. 정지 (1초)")
    controller.stop()
    time.sleep(1)
    
    print("7. 우회전 (2초)")
    controller.right(180)
    time.sleep(2)
    
    print("8. 정지")
    controller.stop()
    
    print("\n데모 완료!")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RC Car Controller')
    parser.add_argument('--port', default='/dev/ttyUSB0', 
                        help='시리얼 포트 (기본: /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=9600,
                        help='보드레이트 (기본: 9600)')
    parser.add_argument('--mode', choices=['interactive', 'demo'], 
                        default='interactive',
                        help='실행 모드: interactive(키보드 제어) 또는 demo(자동 테스트)')
    
    args = parser.parse_args()
    
    # RC Car 제어기 초기화
    controller = RCCarController(port=args.port, baudrate=args.baudrate)
    
    try:
        if args.mode == 'interactive':
            interactive_mode(controller)
        elif args.mode == 'demo':
            demo_mode(controller)
    finally:
        controller.close()


if __name__ == "__main__":
    main()

