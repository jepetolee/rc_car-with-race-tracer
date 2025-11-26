# RC Car 강화학습 핵심 시스템

이산 액션만 사용하는 5단계 학습 파이프라인

## 시스템 구성

### 1. 강화학습과 시뮬레이션으로 사전 학습
**파일**: `train_ppo.py`

```bash
# CarRacing 환경에서 사전 학습
python train_ppo.py \
    --env-type carracing \
    --total-steps 100000 \
    --save-path ppo_model.pth

# 시뮬레이션 환경에서 학습
python train_ppo.py \
    --env-type sim \
    --total-steps 100000 \
    --save-path ppo_model.pth
```

### 2. 실제 환경에서 라즈베리 파이를 컨트롤해서 사전학습
**파일**: `collect_human_demonstrations.py`, `train_with_teacher_forcing.py`

```bash
# 2-1. 사람이 직접 조작한 데이터 수집
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --output human_demos.pkl \
    --episodes 5

# 2-2. Supervised Learning 사전 학습 (Teacher Forcing)
# 사람이 조작한 (상태, 액션) 쌍으로 Maximum Likelihood Estimation 수행
python train_with_teacher_forcing.py \
    --demos human_demos.pkl \
    --pretrain-epochs 100 \
    --pretrain-save pretrained_model.pth
```

### 3. 반복적으로 사람이 라즈베리 파이 모델의 주행을 평가해 강화학습
**파일**: `train_human_feedback.py`

```bash
python train_human_feedback.py \
    --model pretrained_model.pth \
    --port /dev/ttyACM0 \
    --episodes 10 \
    --save ppo_model_feedback.pth
```

**사용 방법:**
1. 모델이 자동으로 주행
2. 사람이 0.0~1.0 점수로 평가
3. 평가 점수를 리워드로 변환하여 학습
4. 반복

### 4. 추론
**파일**: `run_ai_agent.py`

```bash
# 학습된 모델로 추론 실행
python run_ai_agent.py \
    --model ppo_model_feedback.pth \
    --env-type real \
    --port /dev/ttyACM0 \
    --delay 0.1
```

### 5. Arduino 코드
**파일**: `test.ino`

- Arduino IDE에서 업로드
- 이산 액션 (0-4) 지원
- 0.1초 간격으로 명령 처리

## 이산 액션 정의

- **0**: 정지 (Stop/Coast)
- **1**: 우회전 + 가스 (Right + Gas)
- **2**: 좌회전 + 가스 (Left + Gas)
- **3**: 직진 가스 (Gas/Forward)
- **4**: 브레이크 (Brake)

## 전체 워크플로우

```bash
# 1단계: 시뮬레이션 사전 학습
python train_ppo.py --env-type carracing --total-steps 100000

# 2단계: 사람 조작 데이터 수집
python collect_human_demonstrations.py --env-type real --port /dev/ttyACM0 --episodes 5

# 3단계: Supervised Learning 사전 학습 (Teacher Forcing)
python train_with_teacher_forcing.py --demos human_demos.pkl --pretrain-epochs 100

# 4단계: 사람 평가 기반 강화학습
python train_human_feedback.py --model pretrained_model.pth --port /dev/ttyACM0 --episodes 10

# 5단계: 추론
python run_ai_agent.py --model ppo_model_feedback.pth --env-type real --port /dev/ttyACM0
```

## 필수 파일

- `ppo_agent.py`: PPO 에이전트 구현
- `rc_car_sim_env.py`: 시뮬레이션 환경
- `car_racing_env.py`: CarRacing 환경
- `rc_car_env.py`: 실제 하드웨어 환경
- `rc_car_interface.py`: 라즈베리 파이 카메라 인터페이스
- `rc_car_controller.py`: Arduino 시리얼 통신 제어기
- `train_ppo.py`: 강화학습 학습 스크립트
- `collect_human_demonstrations.py`: 사람 조작 데이터 수집
- `train_with_teacher_forcing.py`: Teacher Forcing 학습
- `train_human_feedback.py`: 사람 평가 기반 강화학습
- `run_ai_agent.py`: 추론 실행
- `test.ino`: Arduino 코드

