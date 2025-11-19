# Box2D 설치 가이드

## 문제
Box2D를 설치하려면 `swig`와 빌드 도구가 필요합니다.

## 해결 방법

### 방법 1: 시스템 패키지 설치 (권장)

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y swig cmake build-essential python3-dev
```

그 다음:
```bash
pip install gymnasium[box2d]
# 또는
pip install gym[box2d]
```

### 방법 2: 사전 빌드된 wheel 사용

일부 경우 사전 빌드된 wheel을 사용할 수 있습니다:
```bash
pip install --only-binary :all: box2d-py
pip install gymnasium[box2d]
```

### 방법 3: Box2D 없이 시뮬레이션 환경 사용

Box2D가 설치되지 않아도 시뮬레이션 환경을 사용할 수 있습니다:
```bash
# CarRacing 대신 자체 시뮬레이션 환경 사용
python train_ppo.py --env-type sim --use-extended-actions
```

## 확인

설치 확인:
```python
import gymnasium as gym
env = gym.make('CarRacing-v3')
print("CarRacing-v3 설치 성공!")
```

