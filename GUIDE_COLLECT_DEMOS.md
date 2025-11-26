# collect_human_demonstrations.py 사용 가이드

## 📋 개요

`collect_human_demonstrations.py`는 **사람이 직접 RC Car를 조작한 데이터를 수집**하는 스크립트입니다. 
이 데이터는 나중에 Teacher Forcing 사전 학습에 사용됩니다.

## 🔧 코드 구조

### 1. 주요 클래스: `HumanDemonstrationCollector`

```python
class HumanDemonstrationCollector:
    - __init__(): 환경 및 제어기 초기화
    - _create_env(): 환경 생성 (CarRacing/시뮬레이션/실제 하드웨어)
    - collect_episode(): 단일 에피소드 데이터 수집
    - collect_multiple_episodes(): 여러 에피소드 데이터 수집
    - save_demonstrations(): 수집된 데이터를 pickle 파일로 저장
```

### 2. 동작 흐름

```
1. 환경 초기화
   ↓
2. 에피소드 시작
   ↓
3. 매 0.1초마다:
   - 현재 상태(카메라 이미지) 저장
   - 키보드 입력 확인
   - 입력된 액션 실행 (실제 하드웨어 제어)
   - 환경 스텝 실행
   - 상태, 액션, 리워드 저장
   ↓
4. 에피소드 종료 (q 키 또는 최대 스텝 도달)
   ↓
5. 데이터 저장 (pickle 형식)
```

### 3. 수집되는 데이터

각 에피소드마다 다음 정보가 저장됩니다:

```python
episode_data = {
    'states': [],      # 상태 이미지 (16x16 grayscale, 정규화됨)
    'actions': [],     # 이산 액션 (0-4)
    'rewards': [],     # 환경에서 계산된 리워드
    'dones': [],       # 종료 플래그
    'timestamps': []   # 타임스탬프
}
```

## 🎮 사용 방법

### 기본 실행 (실제 하드웨어)

```bash
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --episodes 5 \
    --output human_demos.pkl
```

### 옵션 설명

- `--env-type`: 환경 타입
  - `real`: 실제 하드웨어 (라즈베리 파이 + Arduino)
  - `carracing`: Gym CarRacing 환경
  - `sim`: 시뮬레이션 환경

- `--port`: 시리얼 포트 (real 모드 사용 시)
  - 기본값: `/dev/ttyACM0`

- `--episodes`: 수집할 에피소드 수
  - 기본값: 1

- `--output`: 저장할 파일 경로
  - 기본값: `human_demos.pkl`

- `--delay`: 액션 간 지연 시간 (초)
  - 기본값: 0.1

- `--max-steps`: 최대 스텝 수
  - 기본값: 1000

## ⌨️ 키보드 조작 방법

스크립트 실행 후 다음 키를 사용하여 RC Car를 조작합니다:

| 키 | 액션 | 설명 |
|---|---|---|
| **w** | Action 3 | 전진 (Gas/Forward) |
| **a** | Action 2 | 좌회전 + 가스 (Left + Gas) |
| **d** | Action 1 | 우회전 + 가스 (Right + Gas) |
| **s** | Action 0 | 정지 (Stop/Coast) |
| **x** | Action 4 | 브레이크 (Brake) |
| **q** | - | 에피소드 종료 |

### 조작 팁

1. **키를 누르고 있으면**: 해당 액션이 계속 실행됩니다
2. **키를 떼면**: 이전 액션이 유지됩니다 (또는 정지)
3. **q 키**: 현재 에피소드를 종료하고 다음 에피소드로 진행

## 📝 실제 사용 시나리오

### 시나리오 1: 단일 에피소드 수집

```bash
# 1. 스크립트 실행
python collect_human_demonstrations.py --env-type real --port /dev/ttyACM0

# 2. 화면에 안내 메시지 표시
# ============================================================
# 에피소드 1 데이터 수집 시작
# ============================================================
# 키보드 조작:
#   w: 전진 (Action 3)
#   a: 좌회전+가스 (Action 2)
#   d: 우회전+가스 (Action 1)
#   s: 정지 (Action 0)
#   x: 브레이크 (Action 4)
#   q: 에피소드 종료
# ============================================================
#
# 조작을 시작하세요... (q로 종료)

# 3. 키보드로 RC Car 조작
# - w: 전진
# - a: 좌회전
# - d: 우회전
# - s: 정지
# - q: 종료

# 4. 에피소드 완료 후 자동 저장
# ✅ 데이터 저장 완료: human_demos.pkl
```

### 시나리오 2: 여러 에피소드 수집

```bash
# 5개 에피소드 수집
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --episodes 5 \
    --output my_demos.pkl
```

**실행 과정:**
1. 에피소드 1 시작 → 조작 → 완료
2. "다음 에피소드를 준비하세요... (3초 후 시작)" 메시지
3. 에피소드 2 시작 → 조작 → 완료
4. ... (반복)
5. 모든 에피소드 완료 후 저장

## 🔍 수집된 데이터 확인

```python
import pickle

# 데이터 로드
with open('human_demos.pkl', 'rb') as f:
    data = pickle.load(f)

# 메타데이터 확인
print("환경 타입:", data['metadata']['env_type'])
print("에피소드 수:", data['metadata']['num_episodes'])
print("총 스텝 수:", data['metadata']['total_steps'])

# 첫 번째 에피소드 확인
episode = data['demonstrations'][0]
print("에피소드 길이:", len(episode['states']))
print("첫 번째 액션:", episode['actions'][0])
```

## ⚠️ 주의사항

1. **실제 하드웨어 사용 시**:
   - Arduino가 `/dev/ttyACM0`에 연결되어 있어야 합니다
   - `test.ino`가 Arduino에 업로드되어 있어야 합니다
   - 라즈베리 파이 카메라가 정상 작동해야 합니다

2. **키보드 입력**:
   - 터미널이 포커스되어 있어야 키 입력이 인식됩니다
   - 키를 누르고 있으면 해당 액션이 계속 실행됩니다

3. **에피소드 종료**:
   - `q` 키를 누르면 현재 에피소드가 종료됩니다
   - 최대 스텝 수(기본 1000)에 도달하면 자동 종료됩니다

4. **데이터 저장**:
   - 모든 에피소드가 완료된 후에만 저장됩니다
   - 중간에 Ctrl+C로 중단하면 데이터가 저장되지 않습니다

## 🎯 다음 단계

데이터 수집이 완료되면:

```bash
# Teacher Forcing 사전 학습
python train_with_teacher_forcing.py \
    --demos human_demos.pkl \
    --pretrain-epochs 100
```

이렇게 수집된 데이터로 모델을 사전 학습할 수 있습니다!

