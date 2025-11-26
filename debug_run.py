#!/usr/bin/env python3
"""
디버깅 스크립트: Bus error 원인 찾기
단계별로 실행하여 어디서 문제가 발생하는지 확인
"""

import sys
import traceback

print("=" * 60)
print("디버깅 시작")
print("=" * 60)

# 1단계: 기본 임포트
print("\n[1단계] 기본 모듈 임포트...")
try:
    import numpy as np
    print("✅ numpy 임포트 성공")
except Exception as e:
    print(f"❌ numpy 임포트 실패: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ torch 임포트 성공 (버전: {torch.__version__})")
except Exception as e:
    print(f"❌ torch 임포트 실패: {e}")
    sys.exit(1)

# 2단계: 환경 임포트
print("\n[2단계] 환경 모듈 임포트...")
try:
    from rc_car_env import RCCarEnv
    print("✅ rc_car_env 임포트 성공")
except Exception as e:
    print(f"❌ rc_car_env 임포트 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

# 3단계: 카메라 인터페이스 테스트
print("\n[3단계] 카메라 인터페이스 테스트...")
try:
    from rc_car_interface import RC_Car_Interface
    print("✅ rc_car_interface 임포트 성공")
except Exception as e:
    print(f"❌ rc_car_interface 임포트 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("   카메라 초기화 중...")
    rc_car = RC_Car_Interface()
    print("✅ 카메라 초기화 성공")
except Exception as e:
    print(f"❌ 카메라 초기화 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("   이미지 캡처 테스트...")
    img = rc_car.get_image_from_camera()
    print(f"✅ 이미지 캡처 성공: {img.shape}")
except Exception as e:
    print(f"❌ 이미지 캡처 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    rc_car.close()
    print("✅ 카메라 종료 성공")
except Exception as e:
    print(f"⚠️  카메라 종료 실패: {e}")

# 4단계: 환경 생성 테스트
print("\n[4단계] 환경 생성 테스트...")
try:
    env = RCCarEnv(
        max_steps=100,
        use_extended_actions=True,
        use_discrete_actions=True
    )
    print("✅ 환경 생성 성공")
except Exception as e:
    print(f"❌ 환경 생성 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("   환경 리셋 테스트...")
    state = env.reset()
    print(f"✅ 환경 리셋 성공: {state.shape}")
except Exception as e:
    print(f"❌ 환경 리셋 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    env.close()
    print("✅ 환경 종료 성공")
except Exception as e:
    print(f"⚠️  환경 종료 실패: {e}")

# 5단계: PPO Agent 임포트
print("\n[5단계] PPO Agent 임포트...")
try:
    from ppo_agent import PPOAgent
    print("✅ ppo_agent 임포트 성공")
except Exception as e:
    print(f"❌ ppo_agent 임포트 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

# 6단계: PPO Agent 생성
print("\n[6단계] PPO Agent 생성...")
try:
    agent = PPOAgent(
        state_dim=256,
        action_dim=5,
        discrete_action=True,
        use_recurrent=True
    )
    print("✅ PPO Agent 생성 성공")
except Exception as e:
    print(f"❌ PPO Agent 생성 실패: {e}")
    traceback.print_exc()
    sys.exit(1)

# 7단계: 모델 로드 테스트 (선택)
print("\n[7단계] 모델 로드 테스트 (선택)...")
import os
model_path = "./ppo_model.pth"
if os.path.exists(model_path):
    try:
        print(f"   모델 로드 중: {model_path}")
        agent.load(model_path)
        print("✅ 모델 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        traceback.print_exc()
else:
    print(f"⚠️  모델 파일 없음: {model_path}")

print("\n" + "=" * 60)
print("✅ 모든 테스트 통과!")
print("=" * 60)

