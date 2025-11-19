# RC Car PPO 강화학습 시스템

## 개요

RC Car를 위한 PPO (Proximal Policy Optimization) 강화학습 시스템입니다.

**⚠️ 중요: 실제 하드웨어로 직접 학습하지 마세요!**

**권장 학습 파이프라인:**
1. **CarRacing 환경에서 사전학습** (Gym CarRacing-v2)
2. **시뮬레이션 환경에서 Fine-tuning** (선택사항)
3. **실제 하드웨어에서 추론/테스트만** 수행

실제 라즈베리파이 카메라로 실시간 학습하는 것은 비효율적이고 위험합니다!

## 주요 특징

### 1. 확장된 액션 공간
- **기본 모드**: `[left_speed, right_speed]` (각각 -1.0 ~ 1.0)
- **확장 모드**: `[전진/후진, 좌회전/우회전]` (rc_car_controller.py 스타일)
  - 전진/후진: -1.0(후진) ~ 1.0(전진)
  - 좌회전/우회전: -1.0(좌회전) ~ 1.0(우회전)

### 2. 시뮬레이션 환경
- Pygame 기반 가상 트랙
- 실제 하드웨어 없이 빠른 학습 가능
- 렌더링 옵션 (시각화 가능)

### 3. 실제 하드웨어 환경
- rc_car_interface.py 기반
- 실제 카메라와 모터 제어
- 안전한 학습을 위해 시뮬레이션에서 먼저 학습 권장

## 설치

```bash
pip install torch numpy gym pygame
```

## 사용 방법

### 🎯 권장 학습 파이프라인

#### 1단계: CarRacing 환경에서 사전학습 (권장)

```bash
# CarRacing 환경에서 사전학습 (500K 스텝)
python pretrain_carracing.py --stage pretrain --pretrain-steps 500000

# 또는 직접 train_ppo.py 사용
python train_ppo.py \
    --env-type carracing \
    --use-extended-actions \
    --total-steps 500000 \
    --save-path ppo_pretrained.pth
```

**이유:**
- 실제 하드웨어 없이 빠르게 학습
- CarRacing은 RC Car와 유사한 도메인
- 사전학습된 모델을 실제 환경으로 전이 가능

#### 2단계: 실제 환경으로 전이학습 (선택사항)

```bash
# 사전학습된 모델을 실제 RC Car 환경으로 Fine-tuning
python pretrain_carracing.py \
    --stage transfer \
    --pretrain-save-path ppo_pretrained.pth \
    --transfer-steps 50000
```

⚠️ **주의**: 실제 하드웨어 사용 시 안전을 확인하세요!

#### 3단계: 학습된 모델 테스트

```bash
# 실제 RC Car 환경에서 테스트 (추론만)
python pretrain_carracing.py \
    --stage test \
    --transfer-save-path ppo_transferred.pth
```

### 대안: 시뮬레이션 환경 학습

```bash
# 시뮬레이션 환경에서 학습
python train_ppo.py --env-type sim --use-extended-actions --total-steps 200000
```

### ⛔ 실제 하드웨어에서 학습 금지

**절대 하지 마세요:**
```bash
# ❌ 이렇게 하지 마세요!
python train_ppo.py --env-type real --mode train
```

**실제 하드웨어는 테스트/추론 전용:**
```bash
# ✅ 테스트만 허용
python train_ppo.py --mode test --env-type real --load-path ppo_pretrained.pth
```

## 주요 파라미터

### 환경 파라미터
- `--env-type`: `sim` (시뮬레이션) 또는 `real` (실제 하드웨어)
- `--use-extended-actions`: 확장된 액션 공간 사용
- `--render`: 시뮬레이션 렌더링 활성화

### 학습 파라미터
- `--total-steps`: 총 학습 스텝 수 (기본: 100000)
- `--max-episode-steps`: 에피소드 최대 길이 (기본: 1000)
- `--update-frequency`: 업데이트 주기 (기본: 2048)
- `--update-epochs`: 업데이트 에폭 수 (기본: 10)

### 네트워크 파라미터
- `--hidden-dim`: 히든 레이어 차원 (기본: 256)
- `--lr-actor`: Actor 학습률 (기본: 3e-4)
- `--lr-critic`: Critic 학습률 (기본: 3e-4)
- `--gamma`: 할인율 (기본: 0.99)
- `--gae-lambda`: GAE 람다 (기본: 0.95)
- `--clip-epsilon`: PPO 클립 범위 (기본: 0.2)

## 파일 구조

- `rc_car_env.py`: 실제 하드웨어 환경
- `rc_car_sim_env.py`: 시뮬레이션 환경 (Pygame)
- `ppo_agent.py`: PPO 알고리즘 구현
- `train_ppo.py`: 학습 스크립트
- `rc_car_controller.py`: 실제 하드웨어 제어 (시리얼 통신)
- `rc_car_interface.py`: RC Car 인터페이스 (카메라, 모터)

## 학습 전략

1. **시뮬레이션에서 먼저 학습**
   - 빠른 학습 가능
   - 안전한 실험
   - 다양한 하이퍼파라미터 테스트

2. **실제 하드웨어로 전이**
   - 시뮬레이션에서 학습한 모델 사용
   - Fine-tuning 또는 직접 학습

3. **확장된 액션 공간 활용**
   - 더 직관적인 제어
   - rc_car_controller.py와 호환

## 리워드 함수

시뮬레이션 환경의 리워드 구성:
- 속도 리워드: 전진 시 리워드
- 거리 리워드: 이동 거리에 비례
- 차선 추적 리워드: 이미지 중앙 밝기
- 안정성 리워드: 직진 유지
- 페널티: 정지 시

## 모델 저장/로드

모델은 자동으로 저장됩니다:
- 주기적 저장: `--save-frequency` 스텝마다
- 최종 저장: 학습 종료 시

로드:
```python
agent.load('ppo_model_sim.pth')
```

## 트러블슈팅

### 시뮬레이션 환경이 렌더링되지 않을 때
- `--render` 플래그 추가
- 디스플레이 환경 확인 (X11 forwarding 등)

### 실제 하드웨어 연결 오류
- 시리얼 포트 확인
- 권한 확인 (`sudo` 또는 사용자 그룹 추가)

### 학습이 느릴 때
- 렌더링 비활성화 (`--render` 제거)
- GPU 사용 확인
- 배치 크기 조정

## 참고 자료

- PPO 논문: Proximal Policy Optimization Algorithms
- Gym 환경: https://gym.openai.com/
- Pygame 문서: https://www.pygame.org/docs/

