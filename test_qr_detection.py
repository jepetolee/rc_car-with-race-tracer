#!/usr/bin/env python3
"""
CNN 기반 QR 코드 감지 테스트 스크립트

훈련된 CNN 모델을 사용하여 QR 코드를 감지하고 감지 시 4초간 정지하는 기능을 테스트합니다.

사용법:
    python test_qr_detection.py --model trained_models/qr_cnn_best.pth
    python test_qr_detection.py --model trained_models/qr_cnn_best.pth --duration 60
    python test_qr_detection.py --model trained_models/qr_cnn_best.pth --no-hardware
"""

import argparse
import sys
import time
import cv2
import torch
from datetime import datetime

try:
    from rc_car_interface import RC_Car_Interface
    HAS_HARDWARE = True
except ImportError as e:
    print(f"⚠️  rc_car_interface를 임포트할 수 없습니다: {e}")
    print("   라즈베리 파이 환경이 아니거나 모듈이 설치되지 않았습니다.")
    HAS_HARDWARE = False

from detect_qr_with_cnn import QRCNNDetector


def test_qr_detection_only(model_path, model_type='standard', duration=30, threshold=0.5):
    """
    CNN 기반 QR 코드 감지만 테스트 (하드웨어 제어 없음)
    
    Args:
        model_path: 훈련된 CNN 모델 경로
        model_type: 모델 타입 ('standard' 또는 'small')
        duration: 테스트 지속 시간 (초)
        threshold: 감지 임계값
    """
    if not HAS_HARDWARE:
        print("❌ 하드웨어가 사용 불가능합니다.")
        return
    
    print("=" * 60)
    print("CNN 기반 QR 코드 감지 테스트 (하드웨어 제어 없음)")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"테스트 지속 시간: {duration}초")
    print(f"임계값: {threshold}")
    print("QR 코드를 카메라 앞에 보여주세요.")
    print("'q' 키를 누르면 조기 종료할 수 있습니다.")
    print("=" * 60)
    print()
    
    try:
        rc_car = RC_Car_Interface()
        print("✅ 카메라 초기화 완료")
        
        # CNN 감지기 초기화
        detector = QRCNNDetector(model_path, model_type=model_type)
        
        start_time = time.time()
        detection_count = 0
        frame_count = 0
        
        print("\nQR 코드 감지 대기 중...")
        print("(Ctrl+C로 종료)")
        print()
        
        while time.time() - start_time < duration:
            # 원본 이미지 캡처
            img = rc_car.get_raw_image()
            
            # CNN으로 QR 코드 감지
            has_qr, confidence = detector.detect(img, threshold=threshold)
            
            frame_count += 1
            
            # 결과 표시
            display_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
            
            if has_qr:
                detection_count += 1
                detector.detection_count = detection_count
                detector.last_detection_time = time.time()
                status_text = f"QR 감지! (신뢰도: {confidence:.2f})"
                color = (0, 255, 0)  # 초록색
                print(f"\n✅ [{detection_count}] QR 코드 감지! (신뢰도: {confidence:.2f})")
                print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                status_text = f"QR 없음 (신뢰도: {confidence:.2f})"
                color = (255, 255, 255)  # 흰색
            
            cv2.putText(display_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_img, f"프레임: {frame_count} | 감지: {detection_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CNN QR Detection Test', display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # CPU 사용량 감소
        
        # 정리
        cv2.destroyAllWindows()
        
        print(f"\n✅ 테스트 완료!")
        print(f"   총 프레임: {frame_count}")
        print(f"   QR 감지 횟수: {detection_count}회")
        print(f"   테스트 시간: {duration}초")
        
        rc_car.close()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        cv2.destroyAllWindows()
        if 'rc_car' in locals():
            try:
                rc_car.close()
            except:
                pass
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        if 'rc_car' in locals():
            try:
                rc_car.close()
            except:
                pass


def test_qr_with_hardware_control(model_path, model_type='standard', duration=60, threshold=0.5, stop_duration=4.0):
    """
    CNN 기반 QR 코드 감지 및 하드웨어 제어 테스트
    
    Args:
        model_path: 훈련된 CNN 모델 경로
        model_type: 모델 타입 ('standard' 또는 'small')
        duration: 테스트 지속 시간 (초)
        threshold: 감지 임계값
        stop_duration: QR 감지 시 정지 시간 (초)
    """
    if not HAS_HARDWARE:
        print("❌ 하드웨어가 사용 불가능합니다.")
        return
    
    try:
        from rc_car_controller import RCCarController
    except ImportError:
        print("❌ rc_car_controller를 임포트할 수 없습니다.")
        print("   하드웨어 제어 없이 감지만 테스트합니다.")
        test_qr_detection_only(model_path, model_type, duration, threshold)
        return
    
    print("=" * 60)
    print("CNN 기반 QR 코드 감지 및 하드웨어 제어 테스트")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"테스트 지속 시간: {duration}초")
    print(f"임계값: {threshold}")
    print(f"정지 시간: {stop_duration}초")
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
        
        # CNN 감지기 초기화
        detector = QRCNNDetector(model_path, model_type=model_type)
        
        start_time = time.time()
        detection_count = 0
        frame_count = 0
        is_stopped = False
        stop_until = 0
        
        print("\nQR 코드 감지 대기 중...")
        print("(Ctrl+C로 종료)")
        print()
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # 정지 시간 체크
            if is_stopped and current_time >= stop_until:
                is_stopped = False
                print("🔄 정지 해제")
            
            # 원본 이미지 캡처
            img = rc_car.get_raw_image()
            
            # CNN으로 QR 코드 감지
            has_qr, confidence = detector.detect(img, threshold=threshold)
            
            frame_count += 1
            
            # QR 감지 시 차량 정지
            if has_qr and not is_stopped:
                detection_count += 1
                detector.detection_count = detection_count
                detector.last_detection_time = current_time
                
                print(f"\n✅ [{detection_count}] QR 코드 감지! (신뢰도: {confidence:.2f})")
                print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                if controller and rc_car.is_moving:
                    print(f"🛑 차량 정지 중 ({stop_duration}초)...")
                    controller.execute_discrete_action(0)  # Stop
                    is_stopped = True
                    stop_until = current_time + stop_duration
            
            # 결과 표시
            display_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
            
            if has_qr:
                status_text = f"QR 감지! (신뢰도: {confidence:.2f})"
                color = (0, 255, 0)  # 초록색
            else:
                status_text = f"QR 없음 (신뢰도: {confidence:.2f})"
                color = (255, 255, 255)  # 흰색
            
            if is_stopped:
                status_text += " [정지 중]"
            
            cv2.putText(display_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_img, f"프레임: {frame_count} | 감지: {detection_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CNN QR Detection with Control', display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # CPU 사용량 감소
        
        # 테스트 종료 시 차량 정지
        if controller:
            controller.execute_discrete_action(0)  # Stop
            print("\n🛑 테스트 종료 - 차량 정지")
        
        # 정리
        cv2.destroyAllWindows()
        
        print(f"\n✅ 테스트 완료!")
        print(f"   총 프레임: {frame_count}")
        print(f"   QR 감지 횟수: {detection_count}회")
        print(f"   테스트 시간: {duration}초")
        
        if controller:
            controller.close()
        rc_car.close()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        if controller:
            try:
                controller.execute_discrete_action(0)  # Stop
                print("🛑 차량 정지 완료")
            except:
                pass
        cv2.destroyAllWindows()
        if 'rc_car' in locals():
            try:
                rc_car.close()
            except:
                pass
        if 'controller' in locals() and controller:
            try:
                controller.close()
            except:
                pass
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        if 'controller' in locals() and controller:
            try:
                controller.execute_discrete_action(0)  # Stop
                controller.close()
            except:
                pass
        if 'rc_car' in locals():
            try:
                rc_car.close()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(
        description='CNN 기반 QR 코드 감지 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 테스트 (30초, 하드웨어 제어 없음)
  python test_qr_detection.py --model trained_models/qr_cnn_best.pth
  
  # 60초 동안 테스트
  python test_qr_detection.py --model trained_models/qr_cnn_best.pth --duration 60
  
  # 하드웨어 제어 없이 감지만 테스트
  python test_qr_detection.py --model trained_models/qr_cnn_best.pth --no-hardware
  
  # 하드웨어 제어 포함 테스트
  python test_qr_detection.py --model trained_models/qr_cnn_best.pth --with-hardware --duration 60
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='훈련된 CNN 모델 경로 (필수)')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'small'],
                        help='모델 타입 (기본: standard)')
    parser.add_argument('--duration', type=int, default=30,
                        help='테스트 지속 시간 (초, 기본: 30)')
    parser.add_argument('--no-hardware', action='store_true',
                        help='하드웨어 제어 없이 감지만 테스트')
    parser.add_argument('--with-hardware', action='store_true',
                        help='하드웨어 제어 포함 테스트 (기본: 감지만)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='감지 임계값 (기본: 0.5)')
    parser.add_argument('--stop-duration', type=float, default=4.0,
                        help='QR 감지 시 정지 시간 (초, 기본: 4.0)')
    
    args = parser.parse_args()
    
    if args.with_hardware:
        test_qr_with_hardware_control(
            args.model,
            model_type=args.model_type,
            duration=args.duration,
            threshold=args.threshold,
            stop_duration=args.stop_duration
        )
    else:
        test_qr_detection_only(
            args.model,
            model_type=args.model_type,
            duration=args.duration,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()

