#!/usr/bin/env python3
"""
NumPy 문제 진단 스크립트
"""

import os
import sys

# 환경 변수 설정 (메모리 정렬 문제 방지)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

print("=" * 60)
print("NumPy 진단")
print("=" * 60)

print("\n[1] 환경 변수 설정:")
print(f"   OPENBLAS_NUM_THREADS = {os.environ.get('OPENBLAS_NUM_THREADS', 'not set')}")
print(f"   MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS', 'not set')}")

print("\n[2] NumPy 임포트 시도...")
try:
    import numpy as np
    print(f"✅ NumPy 임포트 성공")
    print(f"   버전: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy 임포트 실패: {e}")
    sys.exit(1)

print("\n[3] NumPy 기본 연산 테스트...")
try:
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = a + b
    print(f"✅ NumPy 연산 성공: {c}")
except Exception as e:
    print(f"❌ NumPy 연산 실패: {e}")
    sys.exit(1)

print("\n[4] NumPy 설정 정보:")
try:
    # show_config()는 stdout에 직접 출력하므로 캡처하지 않음
    import sys
    print("   NumPy 빌드 정보 확인 중...")
    # show_config() 호출을 피하고 대신 다른 방법으로 확인
    print(f"   NumPy 데이터 타입: {np.dtype('float32')}")
    print(f"   NumPy 배열 생성 테스트: {np.array([1, 2, 3]).dtype}")
except Exception as e:
    print(f"   ⚠️  설정 정보 확인 실패: {e}")

print("\n" + "=" * 60)
print("✅ NumPy 정상 작동")
print("=" * 60)

