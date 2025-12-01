#!/usr/bin/env python3
"""
QR 코드 데이터 수집 스크립트

아두이노 카메라로 이미지를 수집하고, 사용자가 QR 코드가 있는지 없는지 라벨을 입력합니다.
수집한 데이터는 CNN 모델 훈련에 사용됩니다.

사용법:
    python collect_qr_data.py --output-dir qr_dataset
    python collect_qr_data.py --output-dir qr_dataset --auto-label  # 자동 라벨링 (OpenCV QR 감지기 사용)
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
import json

try:
    from rc_car_interface import RC_Car_Interface
    HAS_CAMERA = True
except ImportError as e:
    print(f"❌ 카메라 모듈을 임포트할 수 없습니다: {e}")
    HAS_CAMERA = False
    sys.exit(1)


class QRDataCollector:
    def __init__(self, output_dir="qr_dataset", auto_label=False):
        """
        QR 코드 데이터 수집기
        
        Args:
            output_dir: 데이터 저장 디렉토리
            auto_label: True면 OpenCV QR 감지기를 사용하여 자동 라벨링
        """
        self.output_dir = output_dir
        self.auto_label = auto_label
        self.qr_detector = cv2.QRCodeDetector() if auto_label else None
        
        # 디렉토리 생성
        self.qr_dir = os.path.join(output_dir, "qr_present")
        self.no_qr_dir = os.path.join(output_dir, "qr_absent")
        os.makedirs(self.qr_dir, exist_ok=True)
        os.makedirs(self.no_qr_dir, exist_ok=True)
        
        # 통계
        self.stats = {
            "qr_present": 0,
            "qr_absent": 0,
            "total": 0
        }
        
        # 메타데이터 저장
        self.metadata = []
    
    def get_raw_image(self, rc_car):
        """
        원본 320x320 이미지를 가져옵니다 (전처리 없이)
        """
        return rc_car.get_raw_image()
    
    def auto_detect_qr(self, img):
        """
        OpenCV QR 감지기를 사용하여 자동으로 QR 코드를 감지합니다.
        (참고용으로만 사용, 실제로는 CNN 모델이 더 정확할 수 있습니다)
        """
        if self.qr_detector is None:
            return False
        data, points, _ = self.qr_detector.detectAndDecode(img)
        return bool(data)
    
    def save_image(self, img, label):
        """
        이미지를 저장하고 통계를 업데이트합니다.
        
        Args:
            img: 저장할 이미지 (numpy array)
            label: 1 (QR 있음) 또는 0 (QR 없음)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        if label == 1:
            filename = f"qr_{timestamp}.png"
            filepath = os.path.join(self.qr_dir, filename)
            self.stats["qr_present"] += 1
        else:
            filename = f"no_qr_{timestamp}.png"
            filepath = os.path.join(self.no_qr_dir, filename)
            self.stats["qr_absent"] += 1
        
        cv2.imwrite(filepath, img)
        self.stats["total"] += 1
        
        # 메타데이터 저장
        self.metadata.append({
            "filename": filename,
            "label": int(label),
            "timestamp": timestamp,
            "shape": list(img.shape)
        })
        
        return filepath
    
    def save_metadata(self):
        """메타데이터를 JSON 파일로 저장"""
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump({
                "stats": self.stats,
                "images": self.metadata,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)
        print(f"\n💾 메타데이터 저장: {metadata_file}")
    
    def collect_interactive(self):
        """
        대화형 데이터 수집 모드
        """
        if not HAS_CAMERA:
            print("❌ 카메라를 사용할 수 없습니다.")
            return
        
        print("=" * 60)
        print("QR 코드 데이터 수집 시작")
        print("=" * 60)
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"QR 있음: {self.qr_dir}")
        print(f"QR 없음: {self.no_qr_dir}")
        print()
        print("사용법:")
        print("  - 'q' 또는 '1': QR 코드 있음으로 저장")
        print("  - 'n' 또는 '0': QR 코드 없음으로 저장")
        print("  - 's': 통계 보기")
        print("  - 'x' 또는 ESC: 종료")
        print("=" * 60)
        print()
        
        try:
            rc_car = RC_Car_Interface()
            print("✅ 카메라 초기화 완료\n")
            
            # 디스플레이 사용 가능 여부 확인 (환경 변수로 먼저 확인)
            headless = False
            termios_settings = None
            
            # DISPLAY 환경 변수 확인
            if not os.environ.get('DISPLAY'):
                headless = True
                print("⚠️  DISPLAY 환경 변수가 설정되지 않았습니다. 헤드리스 모드로 실행합니다.")
                print("   키보드 입력으로만 제어할 수 있습니다.\n")
            else:
                print("이미지 캡처 대기 중...")
                print("(첫 이미지가 표시되면 키를 입력하세요)\n")
            
            # 헤드리스 모드에서 키보드 입력을 위해 termios 설정
            if headless:
                try:
                    import termios
                    import tty
                    termios_settings = termios.tcgetattr(sys.stdin)
                    tty.setraw(sys.stdin.fileno())
                except Exception:
                    print("⚠️  키보드 입력 설정 실패. Enter 키로만 제어할 수 있습니다.\n")
            
            while True:
                # 원본 이미지 캡처
                img = self.get_raw_image(rc_car)
                
                key = None
                
                if not headless:
                    try:
                        # 이미지 표시 (확대하여 보기 쉽게)
                        display_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
                        
                        # 통계 정보 표시
                        stats_text = f"QR 있음: {self.stats['qr_present']} | QR 없음: {self.stats['qr_absent']} | 총: {self.stats['total']}"
                        cv2.putText(display_img, stats_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_img, "q/1: QR있음 | n/0: QR없음 | s: 통계 | x: 종료", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.imshow('QR Data Collection', display_img)
                        key = cv2.waitKey(100) & 0xFF
                    except (cv2.error, Exception) as e:
                        # 디스플레이 오류 발생 시 헤드리스 모드로 전환
                        if not headless:
                            headless = True
                            print(f"\n⚠️  디스플레이 오류: {str(e)[:100]}")
                            print("⚠️  헤드리스 모드로 전환합니다...")
                            try:
                                cv2.destroyAllWindows()
                            except:
                                pass
                            # 헤드리스 모드에서 키보드 입력을 위해 termios 설정
                            if termios_settings is None:
                                try:
                                    import termios
                                    import tty
                                    termios_settings = termios.tcgetattr(sys.stdin)
                                    tty.setraw(sys.stdin.fileno())
                                except Exception:
                                    pass
                        key = None
                
                if headless:
                    # 헤드리스 모드: 키보드 입력 확인 (논블로킹)
                    try:
                        import select
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key_char = sys.stdin.read(1)
                            key = ord(key_char) if key_char else None
                    except Exception:
                        # select가 실패하면 키 입력 없음
                        key = None
                
                if key:
                    if key == ord('q') or key == ord('1'):
                        # QR 코드 있음
                        filepath = self.save_image(img, 1)
                        print(f"✅ QR 있음 저장: {filepath}")
                    elif key == ord('n') or key == ord('0'):
                        # QR 코드 없음
                        filepath = self.save_image(img, 0)
                        print(f"✅ QR 없음 저장: {filepath}")
                    elif key == ord('s'):
                        # 통계 출력
                        print(f"\n📊 현재 통계:")
                        print(f"   QR 있음: {self.stats['qr_present']}")
                        print(f"   QR 없음: {self.stats['qr_absent']}")
                        print(f"   총: {self.stats['total']}")
                        print()
                    elif key == ord('x') or key == 27:  # ESC
                        break
                
                if not headless:
                    time.sleep(0.1)  # CPU 사용량 감소
                else:
                    time.sleep(0.5)  # 헤드리스 모드에서는 조금 더 긴 간격
            
            # 정리
            if not headless:
                cv2.destroyAllWindows()
            if termios_settings is not None:
                try:
                    import termios
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios_settings)
                except Exception:
                    pass
            self.save_metadata()
            
            print(f"\n✅ 데이터 수집 완료!")
            print(f"   QR 있음: {self.stats['qr_present']}장")
            print(f"   QR 없음: {self.stats['qr_absent']}장")
            print(f"   총: {self.stats['total']}장")
            
            rc_car.close()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자에 의해 중단되었습니다.")
            if 'headless' in locals() and not headless:
                cv2.destroyAllWindows()
            if 'termios_settings' in locals() and termios_settings is not None:
                try:
                    import termios
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios_settings)
                except Exception:
                    pass
            self.save_metadata()
            if 'rc_car' in locals():
                rc_car.close()
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            if 'headless' in locals() and not headless:
                cv2.destroyAllWindows()
            if 'termios_settings' in locals() and termios_settings is not None:
                try:
                    import termios
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios_settings)
                except Exception:
                    pass
            self.save_metadata()
            if 'rc_car' in locals():
                rc_car.close()
    
    def collect_auto(self, num_images=100, interval=0.5):
        """
        자동 데이터 수집 모드 (OpenCV QR 감지기 사용)
        
        Args:
            num_images: 수집할 이미지 수
            interval: 이미지 간 간격 (초)
        """
        if not HAS_CAMERA:
            print("❌ 카메라를 사용할 수 없습니다.")
            return
        
        print("=" * 60)
        print("QR 코드 데이터 수집 (자동 모드)")
        print("=" * 60)
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"수집할 이미지 수: {num_images}")
        print(f"이미지 간 간격: {interval}초")
        print("=" * 60)
        print()
        
        try:
            rc_car = RC_Car_Interface()
            print("✅ 카메라 초기화 완료\n")
            
            print("자동 수집 시작...")
            print("(Ctrl+C로 중단 가능)\n")
            
            for i in range(num_images):
                # 원본 이미지 캡처
                img = self.get_raw_image(rc_car)
                
                # 자동 라벨링
                has_qr = self.auto_detect_qr(img)
                label = 1 if has_qr else 0
                
                # 저장
                filepath = self.save_image(img, label)
                
                status = "QR 있음" if has_qr else "QR 없음"
                print(f"[{i+1}/{num_images}] {status}: {os.path.basename(filepath)}")
                
                time.sleep(interval)
            
            # 정리
            self.save_metadata()
            
            print(f"\n✅ 데이터 수집 완료!")
            print(f"   QR 있음: {self.stats['qr_present']}장")
            print(f"   QR 없음: {self.stats['qr_absent']}장")
            print(f"   총: {self.stats['total']}장")
            
            rc_car.close()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자에 의해 중단되었습니다.")
            self.save_metadata()
            if 'rc_car' in locals():
                rc_car.close()
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            self.save_metadata()
            if 'rc_car' in locals():
                rc_car.close()


def main():
    parser = argparse.ArgumentParser(
        description='QR 코드 데이터 수집 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 대화형 모드 (사용자가 직접 라벨 입력)
  python collect_qr_data.py --output-dir qr_dataset
  
  # 자동 모드 (OpenCV QR 감지기 사용)
  python collect_qr_data.py --output-dir qr_dataset --auto-label --num-images 200
        """
    )
    
    parser.add_argument('--output-dir', type=str, default='qr_dataset',
                        help='데이터 저장 디렉토리 (기본: qr_dataset)')
    parser.add_argument('--auto-label', action='store_true',
                        help='자동 라벨링 모드 (OpenCV QR 감지기 사용)')
    parser.add_argument('--num-images', type=int, default=100,
                        help='자동 모드에서 수집할 이미지 수 (기본: 100)')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='자동 모드에서 이미지 간 간격(초) (기본: 0.5)')
    
    args = parser.parse_args()
    
    collector = QRDataCollector(
        output_dir=args.output_dir,
        auto_label=args.auto_label
    )
    
    if args.auto_label:
        collector.collect_auto(num_images=args.num_images, interval=args.interval)
    else:
        collector.collect_interactive()


if __name__ == "__main__":
    main()

