#!/usr/bin/env python3
"""
QR 코드 감지 테스트 스크립트

카메라에서 QR 코드를 감지하고 감지 시 4초간 정지하는 기능을 테스트합니다.

사용법:
    python test_qr_detection.py
    python test_qr_detection.py --duration 60  # 60초 동안 테스트
    python test_qr_detection.py --no-hardware  # 하드웨어 제어 없이 감지만 테스트
"""

import argparse
import sys
import time
from datetime import datetime

try:
    from rc_car_interface import RC_Car_Interface
    HAS_HARDWARE = True
except ImportError as e:
    print(f"⚠️  rc_car_interface를 임포트할 수 없습니다: {e}")
    print("   라즈베리 파이 환경이 아니거나 모듈이 설치되지 않았습니다.")
    HAS_HARDWARE = False


def test_qr_detection_only(duration=30):
    """
    QR 코드 감지만 테스트 (하드웨어 제어 없음)
    
    Args:
        duration: 테스트 지속 시간 (초)
    """
    if not HAS_HARDWARE:
        print("❌ 하드웨어가 사용 불가능합니다.")
        return
    
    print("=" * 60)
    print("QR 코드 감지 테스트 (하드웨어 제어 없음)")
    print("=" * 60)
    print(f"테스트 지속 시간: {duration}초")
    print("QR 코드를 카메라 앞에 보여주세요.")
    print("'q' 키를 누르면 조기 종료할 수 있습니다.")
    print("=" * 60)
    print()
    
    try:
        rc_car = RC_Car_Interface()
        print("✅ 카메라 초기화 완료")
        
        start_time = time.time()
        detection_count = 0
        last_qr_data = None
        
        print("\nQR 코드 감지 대기 중...")
        print("(Ctrl+C로 종료)")
        
        while time.time() - start_time < duration:
            detected, qr_data = rc_car.check_and_stop_on_qr()
            
            if detected and qr_data and qr_data != last_qr_data:
                detection_count += 1
                print(f"\n✅ [{detection_count}] QR 코드 감지!")
                print(f"   데이터: '{qr_data}'")
                print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                last_qr_data = qr_data
                # 1초 후 다시 체크 (같은 QR 코드 중복 감지 방지)
                time.sleep(1.0)
            else:
                time.sleep(0.1)  # 0.1초마다 체크
        
        print(f"\n✅ 테스트 완료!")
        print(f"   총 감지 횟수: {detection_count}회")
        print(f"   테스트 시간: {duration}초")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'rc_car' in locals():
            try:
                rc_car.close()
                print("✅ 카메라 종료 완료")
            except:
                pass


def test_qr_with_hardware_control(duration=60):
    """
    QR 코드 감지 및 하드웨어 제어 테스트
    
    Args:
        duration: 테스트 지속 시간 (초)
    """
    if not HAS_HARDWARE:
        print("❌ 하드웨어가 사용 불가능합니다.")
        return
    
    try:
        from rc_car_controller import RCCarController
    except ImportError:
        print("❌ rc_car_controller를 임포트할 수 없습니다.")
        print("   하드웨어 제어 없이 감지만 테스트합니다.")
        test_qr_detection_only(duration)
        return
    
    print("=" * 60)
    print("QR 코드 감지 및 하드웨어 제어 테스트")
    print("=" * 60)
    print(f"테스트 지속 시간: {duration}초")
    print("주의: 차량이 이동 중일 때만 QR 코드 감지 시 정지합니다.")
    print("QR 코드를 카메라 앞에 보여주세요.")
    print("=" * 60)
    print()
    
    try:
        rc_car = RC_Car_Interface()
        print("✅ 카메라 초기화 완료")
        
        # 하드웨어 컨트롤러 초기화
        try:
            controller = RCCarController(port='/dev/ttyACM0', delay=0.1)
            print("✅ 하드웨어 컨트롤러 연결 완료")
        except Exception as e:
            print(f"⚠️  하드웨어 컨트롤러 연결 실패: {e}")
            print("   감지만 테스트합니다.")
            controller = None
        
        start_time = time.time()
        detection_count = 0
        last_qr_data = None
        
        # 테스트: 차량을 가볍게 움직여서 QR 코드 감지 기능 테스트
        print("\n차량을 가볍게 움직여서 QR 코드 감지 테스트...")
        print("(Ctrl+C로 종료)")
        
        # 5초마다 차량을 가볍게 움직임 (테스트용)
        last_move_time = time.time()
        is_moving = False
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # 5초마다 차량을 가볍게 움직임/정지 (테스트용)
            if current_time - last_move_time > 5.0:
                if controller and not is_moving:
                    # 가볍게 전진 (테스트용)
                    controller.execute_discrete_action(3)  # Gas
                    is_moving = True
                    print("🚗 차량 가벼운 전진 시작 (테스트용)")
                elif controller and is_moving:
                    # 정지
                    controller.execute_discrete_action(0)  # Stop
                    is_moving = False
                    print("🛑 차량 정지 (테스트용)")
                last_move_time = current_time
            
            # QR 코드 체크
            detected, qr_data = rc_car.check_and_stop_on_qr()
            
            if detected and qr_data and qr_data != last_qr_data:
                detection_count += 1
                print(f"\n✅ [{detection_count}] QR 코드 감지!")
                print(f"   데이터: '{qr_data}'")
                print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if is_moving:
                    print(f"   차량 정지 중 (4초)...")
                last_qr_data = qr_data
                # 5초 후 다시 체크 (같은 QR 코드 중복 감지 방지)
                time.sleep(5.0)
                is_moving = False  # QR 감지 후 정지 상태 유지
            else:
                time.sleep(0.1)  # 0.1초마다 체크
        
        # 테스트 종료 시 차량 정지
        if controller:
            controller.execute_discrete_action(0)  # Stop
            print("\n🛑 테스트 종료 - 차량 정지")
        
        print(f"\n✅ 테스트 완료!")
        print(f"   총 감지 횟수: {detection_count}회")
        print(f"   테스트 시간: {duration}초")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        if controller:
            try:
                controller.execute_discrete_action(0)  # Stop
                print("🛑 차량 정지 완료")
            except:
                pass
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        if controller:
            try:
                controller.execute_discrete_action(0)  # Stop
            except:
                pass
    finally:
        if controller:
            try:
                controller.close()
                print("✅ 하드웨어 컨트롤러 종료 완료")
            except:
                pass
        if 'rc_car' in locals():
            try:
                rc_car.close()
                print("✅ 카메라 종료 완료")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(
        description='QR 코드 감지 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 테스트 (30초, 하드웨어 제어 없음)
  python test_qr_detection.py
  
  # 60초 동안 테스트
  python test_qr_detection.py --duration 60
  
  # 하드웨어 제어 없이 감지만 테스트
  python test_qr_detection.py --no-hardware
  
  # 하드웨어 제어 포함 테스트
  python test_qr_detection.py --with-hardware --duration 60
        """
    )
    
    parser.add_argument('--duration', type=int, default=30,
                        help='테스트 지속 시간 (초, 기본: 30)')
    parser.add_argument('--no-hardware', action='store_true',
                        help='하드웨어 제어 없이 감지만 테스트')
    parser.add_argument('--with-hardware', action='store_true',
                        help='하드웨어 제어 포함 테스트 (기본: 감지만)')
    
    args = parser.parse_args()
    
    if args.with_hardware:
        test_qr_with_hardware_control(args.duration)
    else:
        test_qr_detection_only(args.duration)


if __name__ == "__main__":
    main()

