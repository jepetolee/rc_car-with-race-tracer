# Teacher Forcing + Imitation RL 학습 가이드

## 📋 개요

학습 속도를 보장하고 더 나은 성능을 얻기 위해 **Teacher Forcing (Supervised Learning)**과 **Imitation RL**을 순차적으로 사용하는 것이 권장됩니다.

## 🎯 학습 파이프라인

```
1. Teacher Forcing (Supervised Learning)
   ↓ 빠른 사전 학습 (사람의 행동 패턴 직접 학습)
   ↓
2. Imitation RL (Reinforcement Learning)
   ↓ Fine-tuning (일치율 기반 리워드로 개선)
   ↓
3. 최종 모델
```

## 📚 README 확인 결과

`README_TRAINING_PIPELINE.md`에 이미 파이프라인이 설명되어 있습니다:

### 예시 1: 시뮬레이션 중심 (권장)

```bash
# 1단계: CarRacing 환경에서 사전 학습 (선택사항)
python train_ppo.py --env-type carracing --total-steps 500000 --save-path ppo_carracing.pth

# 2단계: 사람 데모 데이터 수집
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --episodes 5 \
    --output human_demos.pkl

# 3단계: Teacher Forcing 사전 학습
python train_with_teacher_forcing.py \
    --demos human_demos.pkl \
    --pretrain-epochs 100 \
    --pretrain-save pretrained_model.pth

# 4단계: Imitation RL Fine-tuning
python train_imitation_rl.py \
    --demos human_demos.pkl \
    --model pretrained_model.pth \
    --epochs 20 \
    --save imitation_rl_model.pth
```

## ✅ 코드 실행 가능 여부

**네, 지금 바로 실행 가능합니다!**

### 1. Teacher Forcing 실행

```bash
# 기본 실행
python3 train_with_teacher_forcing.py \
    --demos uploaded_data/demos_20251129_164827.pkl \
    --pretrain-epochs 100 \
    --pretrain-save pretrained_model.pth

# 커스텀 파라미터
python3 train_with_teacher_forcing.py \
    --demos uploaded_data/demos_20251129_164827.pkl \
    --pretrain-epochs 50 \
    --pretrain-batch-size 64 \
    --pretrain-lr 3e-4 \
    --pretrain-save pretrained_model.pth
```

**주요 파라미터:**
- `--demos`: 데모 데이터 파일 경로 (필수)
- `--pretrain-epochs`: 사전 학습 에폭 수 (기본: 0, 0이면 생략)
- `--pretrain-batch-size`: 배치 크기 (기본: 64)
- `--pretrain-lr`: 학습률 (기본: 3e-4)
- `--pretrain-save`: 저장 경로 (기본: `pretrained_model.pth`)

### 2. Imitation RL 실행 (Teacher Forcing 모델 사용)

```bash
# Teacher Forcing으로 학습한 모델을 사용하여 Imitation RL
python3 train_imitation_rl.py \
    --demos uploaded_data/demos_20251129_164827.pkl \
    --model pretrained_model.pth \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --save imitation_rl_model.pth
```

### 3. 서버 API를 통한 실행

```bash
# 1. Teacher Forcing 학습 요청
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --train-supervised uploaded_data/demos_XXX.pkl \
    --pretrain-epochs 100

# 2. Imitation RL 학습 요청 (Teacher Forcing 모델 사용)
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --train uploaded_data/demos_XXX.pkl \
    --pretrain-model pretrained_model.pth \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4
```

## 🔄 전체 워크플로우

### 단계별 설명

#### 1단계: Teacher Forcing (Supervised Learning)

**목적**: 사람의 행동 패턴을 빠르게 학습

**방식**:
- 사람이 조작한 (상태, 액션) 쌍을 사용
- Maximum Likelihood Estimation (MLE)
- 실제 액션의 로그 확률을 최대화
- 빠른 수렴 (일반적으로 50-100 에포크면 충분)

**예상 결과**:
- Match Rate: 60-80% (초기)
- 빠른 학습 (수 분 ~ 수십 분)

#### 2단계: Imitation RL (Reinforcement Learning)

**목적**: Teacher Forcing 모델을 Fine-tuning하여 더 나은 성능 달성

**방식**:
- Teacher Forcing으로 학습한 모델을 초기 가중치로 사용
- 모델 액션과 전문가 액션의 일치율을 리워드로 사용
- PPO 알고리즘으로 학습
- 더 정밀한 조정

**예상 결과**:
- Match Rate: 80-95% (개선)
- 더 안정적인 정책

## 📊 비교: Teacher Forcing vs Imitation RL

| 방법 | 학습 속도 | 초기 성능 | 최종 성능 | 데이터 효율 |
|-----|---------|---------|---------|-----------|
| **Teacher Forcing만** | 매우 빠름 | 높음 (60-80%) | 중간 (70-85%) | 높음 |
| **Imitation RL만** | 느림 | 낮음 (35-40%) | 높음 (80-95%) | 중간 |
| **Teacher Forcing → Imitation RL** | 빠름 | 높음 (60-80%) | 매우 높음 (85-95%) | 매우 높음 |

## 🎯 권장 설정

### Teacher Forcing
```bash
python3 train_with_teacher_forcing.py \
    --demos human_demos.pkl \
    --pretrain-epochs 50-100 \
    --pretrain-batch-size 64 \
    --pretrain-lr 3e-4 \
    --pretrain-save pretrained_model.pth
```

### Imitation RL (Teacher Forcing 모델 사용)
```bash
python3 train_imitation_rl.py \
    --demos human_demos.pkl \
    --model pretrained_model.pth \
    --epochs 20-50 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --save imitation_rl_model.pth
```

## 💡 왜 이 조합이 효과적인가?

1. **Teacher Forcing**: 빠르게 기본 패턴 학습 (Supervised Learning의 장점)
2. **Imitation RL**: 세밀한 조정 및 개선 (Reinforcement Learning의 장점)
3. **시너지**: 두 방법의 장점을 결합하여 빠르고 효과적인 학습

---

# Human Demonstration 수집 시 보상 측정 방법

## 📋 개요

`collect_human_demonstrations.py`를 사용하여 사람이 직접 조작한 데이터를 수집할 때, 환경에서 자동으로 보상(reward)을 계산하여 저장합니다.

## 🔍 보상 계산 방식

### 1. 실제 하드웨어 환경 (`rc_car_env.py`)

실제 RC Car 환경에서 보상은 **카메라 이미지**를 기반으로 계산됩니다:

```python
def _compute_reward(self, img, action):
    """
    리워드 계산 (rc_car_env.py)
    
    계산 요소:
    1. 차선 추적 리워드: 이미지 중앙 영역의 밝기 (차선이 있으면 보상)
    2. 속도 유지 리워드: 적당한 속도 유지 (0.5 근처에서 최대)
    3. 안정성 리워드: 이전 이미지와의 유사성 (안정적인 주행)
    4. 방향 일관성 리워드: 직진 선호
    5. 페널티: 너무 느리거나 멈춤
    6. 전진 액션 보너스: 전진 액션에 작은 보너스
    """
```

**구체적인 계산:**

1. **차선 추적 리워드** (최대 0.5)
   ```python
   center_region = img[6:10, 6:10]  # 중앙 4x4 영역
   center_brightness = np.mean(center_region) / 255.0
   lane_reward = center_brightness * 0.5
   ```
   - 중앙이 밝을수록 (차선이 보일수록) 높은 보상

2. **속도 유지 리워드** (최대 0.3)
   ```python
   speed = np.mean([abs(action[0]), abs(action[1])])
   speed_reward = -abs(speed - 0.5) * 0.3
   ```
   - 속도가 0.5 근처일 때 최대 보상

3. **안정성 리워드** (최대 0.2)
   ```python
   stability = 1.0 - np.mean(np.abs(img - last_image)) / 255.0
   stability_reward = stability * 0.2
   ```
   - 이전 프레임과 유사할수록 높은 보상

4. **방향 일관성 리워드** (최대 0.1)
   ```python
   direction_diff = abs(action[0] - action[1])
   direction_reward = (1.0 - direction_diff) * 0.1
   ```
   - 직진할수록 높은 보상

5. **페널티** (-0.5)
   ```python
   if speed < 0.1:
       reward -= 0.5
   ```
   - 너무 느리거나 멈추면 페널티

6. **전진 액션 보너스** (+0.1)
   ```python
   if use_discrete_actions and abs(action[0]) > 0.5:
       reward += 0.1
   ```

**총 보상 범위**: 약 -0.5 ~ 1.2

### 2. CarRacing 환경

CarRacing 환경에서는 Gym이 제공하는 기본 보상을 사용합니다:
- 차선 유지: 양수 보상
- 트랙 이탈: 음수 보상
- 속도: 양수 보상

### 3. 시뮬레이션 환경

시뮬레이션 환경(`rc_car_sim_env.py`)에서도 유사한 방식으로 보상을 계산합니다.

## 📝 데이터 수집 과정

### 수집 흐름

```
1. 사용자가 키보드로 액션 입력 (w/a/s/d/x)
   ↓
2. 현재 상태(카메라 이미지) 저장
   ↓
3. 환경에 액션 전달: env.step(action)
   ↓
4. 환경이 보상 계산: _compute_reward(img, action)
   ↓
5. 다음 상태, 보상, done, info 반환
   ↓
6. 모든 데이터 저장:
   - states: 상태 (이미지)
   - actions: 액션
   - rewards: 보상 (환경에서 계산)
   - dones: 종료 플래그
   - timestamps: 타임스탬프
```

### 코드 예시

```python
# collect_human_demonstrations.py의 collect_episode 메서드

# 환경 스텝 실행
next_state, reward, done, info = self.env.step(action)

# 데이터 저장
episode_data['actions'].append(action)
episode_data['rewards'].append(reward)  # ← 환경에서 계산된 보상
episode_data['dones'].append(done)
```

## 🔍 보상 확인 방법

### 1. 수집 중 확인

데이터 수집 중 에피소드 완료 시 보상 정보가 출력됩니다:

```
에피소드 완료:
  길이: 250 스텝
  총 리워드: 45.320
  평균 리워드: 0.181
```

### 2. 저장된 데이터 확인

```python
import pickle
import numpy as np

# 데이터 로드
with open('human_demos.pkl', 'rb') as f:
    data = pickle.load(f)

# 첫 번째 에피소드의 보상 확인
episode = data['demonstrations'][0]
rewards = episode['rewards']

print(f"에피소드 길이: {len(rewards)}")
print(f"총 보상: {sum(rewards):.2f}")
print(f"평균 보상: {np.mean(rewards):.3f}")
print(f"최대 보상: {max(rewards):.3f}")
print(f"최소 보상: {min(rewards):.3f}")
```

## ⚠️ 중요 사항

### Imitation Learning에서 보상 사용

**중요**: `train_imitation_rl.py`는 **환경의 보상을 사용하지 않습니다**.

- Imitation RL은 모델 액션과 전문가 액션의 **일치율**을 리워드로 사용
- 일치: +1.0
- 불일치: -0.1

따라서 수집된 데이터의 `rewards` 필드는:
- **Teacher Forcing**: 사용하지 않음 (상태-액션 쌍만 사용)
- **Imitation RL**: 사용하지 않음 (일치율 기반 리워드 사용)
- **일반 RL**: 사용 가능 (환경 보상 사용)

### 보상이 필요한 경우

만약 환경 보상을 사용하고 싶다면:
- `train_ppo.py` 사용 (일반 강화학습)
- 또는 `train_with_teacher_forcing.py`의 `--rl-steps` 옵션 사용

## 📊 보상 통계 예시

```python
# 모든 에피소드의 보상 통계
all_rewards = []
for episode in data['demonstrations']:
    all_rewards.extend(episode['rewards'])

print(f"전체 통계:")
print(f"  총 스텝: {len(all_rewards)}")
print(f"  평균 보상: {np.mean(all_rewards):.3f}")
print(f"  표준편차: {np.std(all_rewards):.3f}")
print(f"  최대 보상: {max(all_rewards):.3f}")
print(f"  최소 보상: {min(all_rewards):.3f}")
```

## 🎯 요약

1. **보상은 자동으로 계산됨**: `env.step(action)` 호출 시 환경이 자동 계산
2. **실제 하드웨어**: 카메라 이미지 기반 보상 (차선 추적, 속도, 안정성 등)
3. **CarRacing/시뮬레이션**: 각 환경의 기본 보상 사용
4. **Imitation RL에서는 사용 안 함**: 일치율 기반 리워드 사용
5. **데이터에는 저장됨**: 나중에 분석하거나 다른 용도로 사용 가능

