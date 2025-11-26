#!/usr/bin/env python3
"""
가장 간단한 테스트: NumPy만 임포트
"""

import os
# 환경 변수 설정
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'

print("환경 변수 설정 완료")
print("NumPy 임포트 시도...")

try:
    import numpy as np
    print(f"✅ NumPy 임포트 성공: {np.__version__}")
    print("NumPy 기본 테스트 완료")
except Exception as e:
    print(f"❌ NumPy 임포트 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("✅ 모든 테스트 통과")

