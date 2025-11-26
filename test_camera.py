#!/usr/bin/env python3
"""
카메라 테스트 스크립트
라즈베리 파이 카메라가 제대로 작동하는지 확인
"""

import cv2
import numpy as np
import sys
import time

try:
    from rc_car_interface import RC_Car_Interface
    HAS_CAMERA = True
except ImportError as e:
    print(f"❌ 카메라 모듈을 임포트할 수 없습니다: {e}")
    HAS_CAMERA = False
    sys.exit(1)


def test_camera():
    """카메라 테스트"""
    print("📷 카메라 테스트 시작...")
    print("=" * 60)
    
    try:
        # 카메라 인터페이스 생성
        print("1. 카메라 초기화 중...")
        rc_car = RC_Car_Interface()
        print("✅ 카메라 초기화 완료")
        
        print("\n2. 이미지 캡처 테스트...")
        print("   (원본 320x320 → 전처리 16x16)")
        print("   'q' 키를 누르면 종료합니다.\n")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # 이미지 캡처
            img = rc_car.get_image_from_camera()
            
            # 원본 이미지 크기 확인
            original_size = img.shape if hasattr(img, 'shape') else "Unknown"
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # 이미지 정보 출력
            if frame_count % 10 == 0:  # 10프레임마다 출력
                print(f"프레임 {frame_count}: 크기={original_size}, FPS={fps:.2f}")
            
            # 16x16 이미지를 320x320으로 확대하여 표시
            display_img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
            
            # 텍스트 추가
            cv2.putText(display_img, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, "Press 'q' to quit", (10, 290),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 이미지 표시
            cv2.imshow('RC Car Camera Test (16x16 -> 320x320)', display_img)
            
            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 정리
        print(f"\n✅ 테스트 완료:")
        print(f"   총 프레임: {frame_count}")
        print(f"   평균 FPS: {fps:.2f}")
        print(f"   실행 시간: {elapsed:.2f}초")
        
        cv2.destroyAllWindows()
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
        if 'rc_car' in locals():
            rc_car.close()
        sys.exit(1)


def test_single_image():
    """단일 이미지 캡처 테스트"""
    print("📷 단일 이미지 캡처 테스트...")
    print("=" * 60)
    
    try:
        rc_car = RC_Car_Interface()
        print("✅ 카메라 초기화 완료")
        
        print("\n이미지 캡처 중...")
        img = rc_car.get_image_from_camera()
        
        print(f"✅ 이미지 캡처 완료")
        print(f"   크기: {img.shape}")
        print(f"   데이터 타입: {img.dtype}")
        print(f"   값 범위: {img.min()} ~ {img.max()}")
        print(f"   평균 밝기: {img.mean():.2f}")
        
        # 이미지 저장
        save_path = 'test_camera_output.png'
        # 16x16을 320x320으로 확대하여 저장
        display_img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_path, display_img)
        print(f"\n💾 이미지 저장: {save_path}")
        
        # 이미지 표시
        cv2.imshow('RC Car Camera Test (16x16 -> 320x320)', display_img)
        print("\n이미지를 확인하세요. 아무 키나 누르면 종료합니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        rc_car.close()
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        if 'rc_car' in locals():
            rc_car.close()
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='라즈베리 파이 카메라 테스트')
    parser.add_argument('--single', action='store_true',
                        help='단일 이미지만 캡처 (기본: 실시간 스트림)')
    
    args = parser.parse_args()
    
    if not HAS_CAMERA:
        print("❌ 카메라를 사용할 수 없습니다.")
        sys.exit(1)
    
    if args.single:
        test_single_image()
    else:
        test_camera()


if __name__ == '__main__':
    main()

