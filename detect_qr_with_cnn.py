#!/usr/bin/env python3
"""
CNN 모델을 사용한 QR 코드 감지 및 차량 제어 스크립트

훈련된 CNN 모델을 사용하여 QR 코드를 감지하고, 감지 시 차량을 정지시킵니다.

사용법:
    python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth
    python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --no-hardware  # 하드웨어 제어 없이 감지만
"""

import argparse
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from rc_car_interface import RC_Car_Interface
    HAS_CAMERA = True
except ImportError as e:
    print(f"❌ 카메라 모듈을 임포트할 수 없습니다: {e}")
    HAS_CAMERA = False
    sys.exit(1)

from qr_cnn_model import create_model


class QRCNNDetector:
    """
    CNN 모델을 사용한 QR 코드 감지기
    """
    
    def __init__(self, model_path, model_type='standard', device=None):
        """
        Args:
            model_path: 훈련된 모델 경로
            model_type: 모델 타입 ('standard' 또는 'small')
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"디바이스: {self.device}")
        
        # 모델 로드
        print(f"모델 로드 중: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 모델 타입 확인
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        
        # 모델 생성 및 로드
        self.model = create_model(model_type=model_type, input_size=320, num_classes=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 모델 로드 완료 (타입: {model_type})")
        
        # 통계
        self.detection_count = 0
        self.last_detection_time = 0
    
    def preprocess_image(self, img):
        """
        이미지 전처리
        
        Args:
            img: 원본 이미지 (numpy array, grayscale)
        
        Returns:
            전처리된 텐서 (1, 1, 320, 320)
        """
        # 크기 조정 (320x320)
        if img.shape != (320, 320):
            img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # (H, W) -> (1, 1, H, W)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        
        # 텐서로 변환
        img_tensor = torch.from_numpy(img).to(self.device)
        
        return img_tensor
    
    def detect(self, img, threshold=0.5, return_probs=False):
        """
        QR 코드 감지
        
        Args:
            img: 원본 이미지 (numpy array, grayscale)
            threshold: 확률 임계값 (기본: 0.5)
            return_probs: True면 확률 분포도 반환 (기본: False)
        
        Returns:
            (has_qr: bool, confidence: float) 또는 (has_qr: bool, confidence: float, probs: tuple)
            - has_qr: QR 코드가 있는지 여부
            - confidence: 가장 높은 확률 (신뢰도)
            - probs: (qr_absent_prob, qr_present_prob) - return_probs=True일 때만
        """
        # 전처리
        img_tensor = self.preprocess_image(img)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # 확률 분포 추출
            probs = probabilities[0].cpu().numpy()
            qr_absent_prob = probs[0]  # QR 없음 확률
            qr_present_prob = probs[1]  # QR 있음 확률
            
            has_qr = (predicted.item() == 1) and (confidence.item() >= threshold)
            conf = confidence.item()
        
        if return_probs:
            return has_qr, conf, (qr_absent_prob, qr_present_prob)
        return has_qr, conf
    
    def get_stats(self):
        """통계 반환"""
        return {
            'detection_count': self.detection_count,
            'last_detection_time': self.last_detection_time
        }


def test_detection_only(model_path, model_type='standard', duration=60, threshold=0.5):
    """
    QR 코드 감지만 테스트 (하드웨어 제어 없음)
    """
    if not HAS_CAMERA:
        print("❌ 카메라를 사용할 수 없습니다.")
        return
    
    print("=" * 60)
    print("CNN 기반 QR 코드 감지 테스트")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"테스트 지속 시간: {duration}초")
    print(f"임계값: {threshold}")
    print("=" * 60)
    print()
    
    try:
        # 카메라 초기화
        rc_car = RC_Car_Interface()
        print("✅ 카메라 초기화 완료")
        
        # 감지기 초기화
        detector = QRCNNDetector(model_path, model_type=model_type)
        
        print("\nQR 코드 감지 대기 중...")
        print("(Ctrl+C로 종료)")
        print()
        
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        
        while time.time() - start_time < duration:
            # 원본 이미지 캡처
            img = rc_car.camera.capture_array()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # QR 코드 감지
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
            else:
                status_text = f"QR 없음 (신뢰도: {confidence:.2f})"
                color = (255, 255, 255)  # 흰색
            
            cv2.putText(display_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_img, f"프레임: {frame_count} | 감지: {detection_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('CNN QR Detection', display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # CPU 사용량 감소
        
        # 정리
        cv2.destroyAllWindows()
        
        print(f"\n✅ 테스트 완료!")
        print(f"   총 프레임: {frame_count}")
        print(f"   QR 감지 횟수: {detection_count}")
        
        rc_car.close()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        cv2.destroyAllWindows()
        if 'rc_car' in locals():
            rc_car.close()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        if 'rc_car' in locals():
            rc_car.close()


def test_with_hardware_control(model_path, model_type='standard', duration=60, threshold=0.5, stop_duration=4.0):
    """
    QR 코드 감지 및 하드웨어 제어 테스트
    
    Args:
        model_path: 모델 경로
        model_type: 모델 타입
        duration: 테스트 지속 시간 (초)
        threshold: 감지 임계값
        stop_duration: QR 감지 시 정지 시간 (초)
    """
    if not HAS_CAMERA:
        print("❌ 카메라를 사용할 수 없습니다.")
        return
    
    try:
        from rc_car_controller import RCCarController
    except ImportError:
        print("❌ rc_car_controller를 임포트할 수 없습니다.")
        print("   하드웨어 제어 없이 감지만 테스트합니다.")
        test_detection_only(model_path, model_type, duration, threshold)
        return
    
    print("=" * 60)
    print("CNN 기반 QR 코드 감지 및 하드웨어 제어 테스트")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"테스트 지속 시간: {duration}초")
    print(f"임계값: {threshold}")
    print(f"정지 시간: {stop_duration}초")
    print("=" * 60)
    print()
    
    try:
        # 카메라 초기화
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
        
        # 감지기 초기화
        detector = QRCNNDetector(model_path, model_type=model_type)
        
        print("\nQR 코드 감지 대기 중...")
        print("(Ctrl+C로 종료)")
        print()
        
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        is_stopped = False
        stop_until = 0
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # 정지 시간 체크
            if is_stopped and current_time >= stop_until:
                is_stopped = False
                print("🔄 정지 해제")
            
            # 원본 이미지 캡처
            img = rc_car.camera.capture_array()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # QR 코드 감지
            has_qr, confidence = detector.detect(img, threshold=threshold)
            
            frame_count += 1
            
            # QR 감지 시 차량 정지
            if has_qr and not is_stopped:
                detection_count += 1
                detector.detection_count = detection_count
                detector.last_detection_time = current_time
                
                print(f"\n✅ [{detection_count}] QR 코드 감지! (신뢰도: {confidence:.2f})")
                
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
        print(f"   QR 감지 횟수: {detection_count}")
        
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
            rc_car.close()
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
            rc_car.close()


def main():
    parser = argparse.ArgumentParser(
        description='CNN 기반 QR 코드 감지 및 차량 제어',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 하드웨어 제어 없이 감지만 테스트
  python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --no-hardware
  
  # 하드웨어 제어 포함 테스트
  python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --with-hardware
  
  # 임계값 조정
  python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --threshold 0.7
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='훈련된 모델 경로')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'small'],
                        help='모델 타입 (기본: standard)')
    parser.add_argument('--no-hardware', action='store_true',
                        help='하드웨어 제어 없이 감지만 테스트')
    parser.add_argument('--with-hardware', action='store_true',
                        help='하드웨어 제어 포함 테스트')
    parser.add_argument('--duration', type=int, default=60,
                        help='테스트 지속 시간 (초, 기본: 60)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='감지 임계값 (기본: 0.5)')
    parser.add_argument('--stop-duration', type=float, default=4.0,
                        help='QR 감지 시 정지 시간 (초, 기본: 4.0)')
    
    args = parser.parse_args()
    
    if args.with_hardware:
        test_with_hardware_control(
            args.model,
            model_type=args.model_type,
            duration=args.duration,
            threshold=args.threshold,
            stop_duration=args.stop_duration
        )
    else:
        test_detection_only(
            args.model,
            model_type=args.model_type,
            duration=args.duration,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()

