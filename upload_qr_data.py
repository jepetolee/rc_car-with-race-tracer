#!/usr/bin/env python3
"""
QR 데이터를 서버로 스트리밍 전송하는 클라이언트

수집한 QR 데이터를 실시간으로 서버에 전송합니다.

사용법:
    python upload_qr_data.py --server http://192.168.1.100:5000 --data-dir qr_dataset
    python upload_qr_data.py --server 192.168.1.100:5000 --data-dir qr_dataset --stream  # 실시간 스트리밍
"""

import argparse
import os
import sys
import time
import json
import cv2
import numpy as np
import requests
from pathlib import Path
from datetime import datetime
import base64

try:
    from rc_car_interface import RC_Car_Interface
    HAS_CAMERA = True
except ImportError as e:
    print(f"⚠️  카메라 모듈을 임포트할 수 없습니다: {e}")
    HAS_CAMERA = False


class QRDataUploader:
    """QR 데이터를 서버로 업로드하는 클라이언트"""
    
    def __init__(self, server_url='http://localhost:5000'):
        """
        Args:
            server_url: 서버 URL
        """
        if not server_url.startswith('http://') and not server_url.startswith('https://'):
            server_url = 'http://' + server_url
        self.server_url = server_url.rstrip('/')
    
    def health_check(self):
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.server_url}/api/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 서버 연결 실패: {e}")
            return None
    
    def upload_image_batch(self, images, labels, metadata=None):
        """
        이미지 배치를 서버로 업로드
        
        Args:
            images: 이미지 리스트 (numpy arrays)
            labels: 라벨 리스트 (0 또는 1)
            metadata: 메타데이터 (선택)
        
        Returns:
            업로드 결과
        """
        try:
            # 이미지를 base64로 인코딩
            encoded_images = []
            for img in images:
                # PNG로 인코딩
                _, buffer = cv2.imencode('.png', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                encoded_images.append(img_base64)
            
            # 데이터 준비
            data = {
                'images': encoded_images,
                'labels': labels,
                'metadata': metadata or {}
            }
            
            # 업로드
            response = requests.post(
                f"{self.server_url}/api/upload_qr_data",
                json=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 업로드 실패: {e}")
            return None
    
    def upload_from_directory(self, data_dir):
        """
        디렉토리에서 수집한 데이터를 서버로 업로드
        
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        qr_dir = os.path.join(data_dir, "qr_present")
        no_qr_dir = os.path.join(data_dir, "qr_absent")
        
        if not os.path.exists(qr_dir) and not os.path.exists(no_qr_dir):
            print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
            return False
        
        # 서버 상태 확인
        print("🔍 서버 연결 확인 중...")
        health = self.health_check()
        if not health:
            print("❌ 서버에 연결할 수 없습니다.")
            return False
        print(f"✅ 서버 연결 확인: {health.get('status', 'unknown')}")
        print()
        
        # 이미지 수집
        images = []
        labels = []
        
        # QR 있음 이미지
        if os.path.exists(qr_dir):
            for filename in os.listdir(qr_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(qr_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(1)
        
        # QR 없음 이미지
        if os.path.exists(no_qr_dir):
            for filename in os.listdir(no_qr_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(no_qr_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(0)
        
        if len(images) == 0:
            print("❌ 업로드할 이미지가 없습니다.")
            return False
        
        print(f"📊 총 {len(images)}장의 이미지 발견")
        print(f"   QR 있음: {sum(labels)}장")
        print(f"   QR 없음: {len(labels) - sum(labels)}장")
        print()
        
        # 배치 단위로 업로드
        batch_size = 10
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        print(f"📤 배치 단위로 업로드 시작 (배치 크기: {batch_size})...")
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"📤 배치 {batch_num}/{total_batches} 업로드 중... ({len(batch_images)}장)", end='', flush=True)
            
            result = self.upload_image_batch(
                batch_images,
                batch_labels,
                metadata={'batch_index': batch_num, 'total_batches': total_batches}
            )
            
            if result:
                print(f" ✅")
            else:
                print(f" ❌")
                return False
        
        print()
        print(f"✅ 모든 데이터 업로드 완료!")
        return True
    
    def stream_realtime(self, interval=1.0, duration=60, threshold=0.5):
        """
        실시간으로 이미지를 캡처하여 서버로 스트리밍
        
        Args:
            interval: 이미지 캡처 간격 (초)
            duration: 스트리밍 지속 시간 (초, 0이면 무한)
            threshold: QR 감지 임계값 (자동 라벨링용)
        """
        if not HAS_CAMERA:
            print("❌ 카메라를 사용할 수 없습니다.")
            return False
        
        # 서버 상태 확인
        print("🔍 서버 연결 확인 중...")
        health = self.health_check()
        if not health:
            print("❌ 서버에 연결할 수 없습니다.")
            return False
        print(f"✅ 서버 연결 확인: {health.get('status', 'unknown')}")
        print()
        
        print("=" * 60)
        print("실시간 QR 데이터 스트리밍 시작")
        print("=" * 60)
        print(f"서버: {self.server_url}")
        print(f"캡처 간격: {interval}초")
        print(f"지속 시간: {duration}초" if duration > 0 else "지속 시간: 무한")
        print("=" * 60)
        print()
        
        try:
            rc_car = RC_Car_Interface()
            print("✅ 카메라 초기화 완료")
            
            # QR 감지기 (자동 라벨링용)
            qr_detector = cv2.QRCodeDetector()
            
            start_time = time.time()
            frame_count = 0
            uploaded_count = 0
            
            print("\n스트리밍 시작...")
            print("(Ctrl+C로 중단)")
            print()
            
            while True:
                current_time = time.time()
                
                # 지속 시간 체크
                if duration > 0 and (current_time - start_time) >= duration:
                    break
                
                # 이미지 캡처
                img = rc_car.get_raw_image()
                
                # 자동 라벨링 (OpenCV QR 감지기 사용)
                data, points, _ = qr_detector.detectAndDecode(img)
                has_qr = bool(data)
                label = 1 if has_qr else 0
                
                # 서버로 전송
                result = self.upload_image_batch(
                    [img],
                    [label],
                    metadata={
                        'timestamp': datetime.now().isoformat(),
                        'frame_count': frame_count,
                        'auto_labeled': True
                    }
                )
                
                frame_count += 1
                if result:
                    uploaded_count += 1
                    status = "QR 있음" if has_qr else "QR 없음"
                    print(f"[{frame_count}] {status} 업로드 완료 (총 {uploaded_count}장)", end='\r', flush=True)
                
                time.sleep(interval)
            
            print()
            print(f"\n✅ 스트리밍 완료!")
            print(f"   총 프레임: {frame_count}")
            print(f"   업로드 성공: {uploaded_count}장")
            
            rc_car.close()
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자에 의해 중단되었습니다.")
            if 'rc_car' in locals():
                rc_car.close()
            return False
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            if 'rc_car' in locals():
                rc_car.close()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='QR 데이터를 서버로 업로드',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 디렉토리에서 수집한 데이터 업로드
  python upload_qr_data.py --server 192.168.1.100:5000 --data-dir qr_dataset
  
  # 실시간 스트리밍
  python upload_qr_data.py --server 192.168.1.100:5000 --stream --duration 300
        """
    )
    
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                        help='서버 URL (기본: http://localhost:5000)')
    parser.add_argument('--data-dir', type=str,
                        help='업로드할 데이터 디렉토리')
    parser.add_argument('--stream', action='store_true',
                        help='실시간 스트리밍 모드')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='스트리밍 모드에서 이미지 캡처 간격(초) (기본: 1.0)')
    parser.add_argument('--duration', type=int, default=60,
                        help='스트리밍 모드에서 지속 시간(초, 0=무한) (기본: 60)')
    
    args = parser.parse_args()
    
    uploader = QRDataUploader(server_url=args.server)
    
    if args.stream:
        # 실시간 스트리밍
        uploader.stream_realtime(
            interval=args.interval,
            duration=args.duration
        )
    elif args.data_dir:
        # 디렉토리에서 업로드
        uploader.upload_from_directory(args.data_dir)
    else:
        parser.print_help()
        print("\n❌ --data-dir 또는 --stream 옵션을 지정해야 합니다.")


if __name__ == "__main__":
    main()

