#!/usr/bin/env python3
"""
RC Car 간단 테스트 스크립트
빠른 동작 테스트를 위한 간단한 스크립트
"""

import serial
import time
import sys


def test_rc_car(port='/dev/ttyUSB0'):
    """RC 카 기본 동작 테스트"""
    
    print(f"연결 시도: {port}")
    
    try:
        # 시리얼 포트 열기
        ser = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)  # Arduino 초기화 대기
        
        # 초기 메시지 읽기
        if ser.in_waiting:
            print(ser.readline().decode('utf-8').strip())
        
        print("\n=== RC Car 테스트 시작 ===\n")
        
        # 테스트 1: 전진
        print("테스트 1: 전진 (속도 200, 2초)")
        ser.write(b"F200\n")
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"  Arduino: {ser.readline().decode('utf-8').strip()}")
        time.sleep(2)
        
        # 정지
        print("정지 (1초)")
        ser.write(b"S\n")
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"  Arduino: {ser.readline().decode('utf-8').strip()}")
        time.sleep(1)
        
        # 테스트 2: 후진
        print("\n테스트 2: 후진 (속도 150, 2초)")
        ser.write(b"B150\n")
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"  Arduino: {ser.readline().decode('utf-8').strip()}")
        time.sleep(2)
        
        # 정지
        print("정지 (1초)")
        ser.write(b"S\n")
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"  Arduino: {ser.readline().decode('utf-8').strip()}")
        time.sleep(1)
        
        # 테스트 3: 좌회전
        print("\n테스트 3: 좌회전 (속도 180, 1.5초)")
        ser.write(b"L180\n")
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"  Arduino: {ser.readline().decode('utf-8').strip()}")
        time.sleep(1.5)
        
        # 정지
        print("정지 (1초)")
        ser.write(b"S\n")
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"  Arduino: {ser.readline().decode('utf-8').strip()}")
        time.sleep(1)
        
        # 테스트 4: 우회전
        print("\n테스트 4: 우회전 (속도 180, 1.5초)")
        ser.write(b"R180\n")
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"  Arduino: {ser.readline().decode('utf-8').strip()}")
        time.sleep(1.5)
        
        # 최종 정지
        print("\n최종 정지")
        ser.write(b"S\n")
        time.sleep(0.5)
        while ser.in_waiting:
            print(f"  Arduino: {ser.readline().decode('utf-8').strip()}")
        
        print("\n=== 테스트 완료 ===")
        
        # 포트 닫기
        ser.close()
        
    except serial.SerialException as e:
        print(f"에러: {e}")
        print("\n시리얼 포트를 확인하세요:")
        print("  Linux: /dev/ttyUSB0, /dev/ttyACM0")
        print("  Windows: COM3, COM4, etc.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n테스트 중단됨")
        ser.write(b"S\n")  # 정지
        ser.close()
        sys.exit(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RC Car 간단 테스트')
    parser.add_argument('--port', default='/dev/ttyUSB0', 
                        help='시리얼 포트 (기본: /dev/ttyUSB0)')
    
    args = parser.parse_args()
    
    test_rc_car(args.port)


