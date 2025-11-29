# RC Car 학습 파이프라인 통합 가이드

이 문서는 RC Car를 위한 모든 학습 방법을 통합하여 정리한 가이드입니다.

## 📋 목차

1. [학습 방법 개요](#학습-방법-개요)
2. [권장 학습 파이프라인](#권장-학습-파이프라인)
3. [학습 방법별 상세 가이드](#학습-방법별-상세-가이드)
4. [서버-클라이언트 학습](#서버-클라이언트-학습)
5. [액션 정의](#액션-정의)
6. [문제 해결](#문제-해결)

---

## 학습 방법 개요

이 프로젝트는 5가지 학습 방법을 지원합니다:

| 학습 방법 | 파일 | 설명 | 사용 시점 |
|---------|------|------|----------|
| **A3C** | `train_a3c.py` | 비동기 강화학습, 멀티 프로세스 | 빠른 사전 학습 |
| **PPO (CarRacing)** | `train_ppo.py` | PPO 강화학습, CarRacing 환경 | 시뮬레이션 사전 학습 |
| **PPO (시뮬레이션)** | `train_ppo.py` | PPO 강화학습, Pygame 시뮬레이션 | 하드웨어 없는 학습 |
| **Teacher Forcing** | `train_with_teacher_forcing.py` | 사람 데모로 Supervised Learning | 실제 환경 사전 학습 |
| **Human Feedback** | `train_human_feedback.py` | 사람 평가 기반 강화학습 | Fine-tuning |
| **Imitation RL** | `train_imitation_rl.py` | 사람 데모로 Imitation Learning via RL | 실제 환경 Fine-tuning |

---

## 권장 학습 파이프라인

### 파이프라인 1: 시뮬레이션 중심 (권장)

```
1. CarRacing/시뮬레이션 PPO 사전 학습
   ↓
2. 사람 데모 데이터 수집
   ↓
3. Teacher Forcing (Supervised Learning)
   ↓
4. Imitation RL (선택사항)
   ↓
5. Human Feedback (선택사항)
   ↓
6. 추론/테스트
```

### 파이프라인 2: 실제 환경 중심

```
1. 사람 데모 데이터 수집
   ↓
2. Teacher Forcing (Supervised Learning)
   ↓
3. Imitation RL
   ↓
4. Human Feedback
   ↓
5. 추론/테스트
```

---

## 학습 방법별 상세 가이드

### 1. A3C 학습 (비동기 강화학습)

**파일**: `train_a3c.py`

**설명**: 멀티 프로세스를 사용한 비동기 강화학습으로 빠르게 사전 학습합니다.

**사용법**:

```bash
# A3C 학습 실행 (CarRacing 환경 사용)
python train_a3c.py \
    --num-workers 4 \
    --total-steps 500000 \
    --save-path a3c_model_best.pth
```

**주요 파라미터**:
- `--num-workers`: 워커 프로세스 수 (기본: 4)
- `--total-steps`: 총 학습 스텝 수 (기본: 1000000)
- `--save-path`: 모델 저장 경로 (기본: `a3c_model.pth`)
- `--max-episode-steps`: 에피소드 최대 길이 (기본: 1000)
- `--hidden-dim`: 히든 레이어 차원 (기본: 256)
- `--lr-actor`: Actor 학습률 (기본: 3e-4)
- `--lr-critic`: Critic 학습률 (기본: 3e-4)

**특징**:
- 멀티 프로세스로 빠른 학습 (CarRacing 환경 사용)
- 실제 하드웨어 없이 실행 가능
- 워커 수 조정으로 학습 속도 제어 가능

---

### 2. PPO 학습 (CarRacing/시뮬레이션)

**파일**: `train_ppo.py`

**설명**: CarRacing 또는 Pygame 시뮬레이션 환경에서 PPO 강화학습을 수행합니다.

#### 2-1. CarRacing 환경

```bash
# CarRacing 환경에서 학습
python train_ppo.py \
    --env-type carracing \
    --total-steps 500000 \
    --save-path ppo_carracing.pth \
    --render  # 시각화 (선택사항)
```

#### 2-2. 시뮬레이션 환경

```bash
# Pygame 시뮬레이션에서 학습
python train_ppo.py \
    --env-type sim \
    --total-steps 200000 \
    --save-path ppo_sim.pth
```

**주요 파라미터**:
- `--env-type`: `carracing` 또는 `sim`
- `--total-steps`: 총 학습 스텝 수 (기본: 100000)
- `--max-episode-steps`: 에피소드 최대 길이 (기본: 1000)
- `--update-frequency`: PPO 업데이트 주기 (기본: 2048)
- `--update-epochs`: 업데이트 에폭 수 (기본: 10)
- `--hidden-dim`: 히든 레이어 차원 (기본: 256)
- `--lr-actor`: Actor 학습률 (기본: 3e-4)
- `--lr-critic`: Critic 학습률 (기본: 3e-4)
- `--render`: 시각화 활성화

**특징**:
- 실제 하드웨어 없이 빠른 학습
- CarRacing은 RC Car와 유사한 도메인
- 사전학습된 모델 전이 가능

**⚠️ 주의**: 실제 하드웨어에서 직접 학습하지 마세요! 시뮬레이션에서 먼저 학습 후 전이하세요.

---

### 3. 사람 데모 데이터 수집

**파일**: `collect_human_demonstrations.py`

**설명**: 사람이 직접 RC Car를 조작한 데이터를 수집합니다.

**사용법**:

```bash
# 실제 하드웨어에서 데이터 수집
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --episodes 5 \
    --output human_demos.pkl
```

**키보드 조작**:
- `w`: 전진 (Action 3)
- `a`: 좌회전 + 가스 (Action 2)
- `d`: 우회전 + 가스 (Action 1)
- `s`: 정지 (Action 0)
- `x`: 브레이크 (Action 4)
- `q`: 에피소드 종료

**수집되는 데이터**:
```python
{
    'metadata': {
        'env_type': 'real',
        'num_episodes': 5,
        'total_steps': 250
    },
    'demonstrations': [
        {
            'states': [...],      # 16x16 grayscale 이미지
            'actions': [...],     # 이산 액션 (0-4)
            'rewards': [...],     # 환경 리워드
            'dones': [...],       # 종료 플래그
            'timestamps': [...]   # 타임스탬프
        },
        ...
    ]
}
```

**특징**:
- 실제 환경에서 전문가 행동 학습
- Teacher Forcing 및 Imitation RL에 사용
- 여러 에피소드 수집 가능

---

### 4. Teacher Forcing (Supervised Learning)

**파일**: `train_with_teacher_forcing.py`

**설명**: 사람이 조작한 (상태, 액션) 쌍으로 Maximum Likelihood Estimation을 수행합니다.

**사용법**:

```bash
# Supervised Learning 사전 학습
python train_with_teacher_forcing.py \
    --demos human_demos.pkl \
    --pretrain-epochs 100 \
    --pretrain-save pretrained_model.pth

# 사전 학습 후 강화학습 Fine-tuning
python train_with_teacher_forcing.py \
    --demos human_demos.pkl \
    --pretrain-epochs 100 \
    --pretrain-save pretrained_model.pth \
    --rl-steps 100000 \
    --rl-save fine_tuned_model.pth
```

**주요 파라미터**:
- `--demos`: 데모 데이터 파일 경로
- `--pretrain-epochs`: 사전 학습 에폭 수 (기본: 100)
- `--pretrain-save`: 사전 학습 모델 저장 경로
- `--rl-steps`: 강화학습 Fine-tuning 스텝 수 (선택사항)
- `--rl-save`: Fine-tuning 모델 저장 경로

**특징**:
- 사람의 행동 패턴 직접 학습
- 빠른 수렴 (Supervised Learning)
- 강화학습 Fine-tuning 가능

---

### 5. Imitation RL (Imitation Learning via Reinforcement Learning)

**파일**: `train_imitation_rl.py`

**설명**: 사람 데모 데이터와의 일치율을 리워드로 사용하여 PPO 강화학습을 수행합니다.

**사용법**:

```bash
# Imitation RL 학습
python train_imitation_rl.py \
    --demos human_demos.pkl \
    --model a3c_model_best.pth \  # 사전 학습 모델 (선택사항)
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --save imitation_rl_model.pth
```

**서버 API를 통한 학습**:

```bash
# 클라이언트에서 학습 요청
python client_upload.py \
    --server http://SERVER_IP:5000 \
    --train uploaded_data/demos_XXX.pkl \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4
```

**주요 파라미터**:
- `--demos`: 데모 데이터 파일 경로 (필수)
- `--model`: 사전 학습 모델 경로 (선택사항)
- `--epochs`: 학습 에폭 수 (기본: 100)
- `--batch-size`: 배치 크기 (기본: 64)
- `--learning-rate`: 학습률 (기본: 3e-4)
- `--save`: 모델 저장 경로 (기본: `imitation_rl_model.pth`)
- `--device`: 디바이스 (기본: `cpu`)

**리워드 정의**:
- 액션 일치: `+1.0`
- 액션 불일치: `-0.1`

**특징**:
- Supervised Learning과 Reinforcement Learning 결합
- 시퀀스 모드 지원 (에피소드별 학습)
- Recurrent 모델 지원 (Deep Supervision)
- 에피소드별 일치율 계산

**리워드 사용**: Imitation Learning이므로 환경의 `rewards`는 사용하지 않습니다. 모델 액션과 전문가 액션의 일치율로 리워드를 자동 생성합니다.

---

### 6. Human Feedback (사람 평가 기반 강화학습)

**파일**: `train_human_feedback.py`

**설명**: 사람이 모델의 주행을 평가하여 리워드를 생성하고 강화학습을 수행합니다.

**사용법**:

```bash
# Human Feedback 학습
python train_human_feedback.py \
    --model pretrained_model.pth \
    --port /dev/ttyACM0 \
    --num-episodes 10 \
    --save-path ppo_feedback_model.pth
```

**사용 방법**:
1. 모델이 자동으로 주행
2. 사람이 0.0~1.0 점수로 평가
3. 평가 점수를 리워드로 변환하여 학습
4. 반복

**주요 파라미터**:
- `--model`: 사전 학습 모델 경로
- `--port`: 시리얼 포트 (실제 하드웨어 사용 시)
- `--num-episodes`: 에피소드 수
- `--save-path`: 모델 저장 경로

**특징**:
- 사람의 주관적 평가 활용
- 실제 환경에서 직접 학습
- Fine-tuning에 적합

---

## 서버-클라이언트 학습

서버-클라이언트 아키텍처를 사용하여 라즈베리 파이에서 데이터 수집/추론만 수행하고, 학습은 GPU 서버에서 수행할 수 있습니다.

**아키텍처**:
```
┌─────────────────┐         HTTP REST API         ┌─────────────────┐
│  라즈베리 파이    │ ◄──────────────────────────► │   서버 (GPU)     │
│  (클라이언트)     │                               │   (학습 서버)    │
├─────────────────┤                               ├─────────────────┤
│ - 카메라 수집     │                               │ - 데이터 수신    │
│ - 하드웨어 제어   │                               │ - 모델 학습      │
│ - 추론 실행       │                               │ - 모델 저장      │
│ - 데이터 전송     │                               │ - 모델 제공      │
└─────────────────┘                               └─────────────────┘
```

### 서버 설정

```bash
# 서버 실행
python server_api.py --host 0.0.0.0 --port 5000
```

### 전체 워크플로우

```bash
# 1. 서버 시작 (GPU 서버)
python server_api.py --host 0.0.0.0 --port 5000

# 2. 데이터 수집 (라즈베리 파이)
python collect_human_demonstrations.py --env-type real --episodes 5

# 3. 데이터 업로드 (라즈베리 파이)
python client_upload.py --server http://SERVER_IP:5000 --upload human_demos.pkl

# 4. 학습 요청 (라즈베리 파이 또는 서버)
python client_upload.py \
    --server http://SERVER_IP:5000 \
    --train uploaded_data/demos_XXX.pkl \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4

# 5. 모델 다운로드 (라즈베리 파이)
python client_upload.py --server http://SERVER_IP:5000 --download latest_model.pth

# 6. 추론 실행 (라즈베리 파이)
python run_ai_agent.py --model latest_model.pth --env-type real
```

서버 API 자세한 내용은 `server_api.py` 코드 참고.

---

## 액션 정의

이 프로젝트는 **이산 액션**만 사용합니다:

| 액션 ID | 설명 | RC Car 동작 |
|--------|------|------------|
| **0** | 정지 (Stop/Coast) | 모터 정지 |
| **1** | 우회전 + 가스 (Right + Gas) | 우측 모터 느리게, 좌측 모터 빠르게 |
| **2** | 좌회전 + 가스 (Left + Gas) | 좌측 모터 느리게, 우측 모터 빠르게 |
| **3** | 직진 가스 (Gas/Forward) | 양쪽 모터 동일 속도 전진 |
| **4** | 브레이크 (Brake) | 급정지 |

---

## 전체 학습 파이프라인 예시

### 예시 1: 시뮬레이션 중심 (권장)

```bash
# 1단계: CarRacing 환경에서 사전 학습
python train_ppo.py \
    --env-type carracing \
    --total-steps 500000 \
    --save-path ppo_carracing.pth

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

# 5단계: 추론/테스트
python run_ai_agent.py \
    --model imitation_rl_model.pth \
    --env-type real \
    --port /dev/ttyACM0
```

### 예시 2: 실제 환경 중심

```bash
# 1단계: 사람 데모 데이터 수집
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --episodes 10 \
    --output human_demos.pkl

# 2단계: Teacher Forcing 사전 학습
python train_with_teacher_forcing.py \
    --demos human_demos.pkl \
    --pretrain-epochs 100 \
    --pretrain-save pretrained_model.pth

# 3단계: Imitation RL
python train_imitation_rl.py \
    --demos human_demos.pkl \
    --model pretrained_model.pth \
    --epochs 20 \
    --save imitation_rl_model.pth

# 4단계: Human Feedback (선택사항)
python train_human_feedback.py \
    --model imitation_rl_model.pth \
    --port /dev/ttyACM0 \
    --num-episodes 10 \
    --save-path final_model.pth

# 5단계: 추론/테스트
python run_ai_agent.py \
    --model final_model.pth \
    --env-type real \
    --port /dev/ttyACM0
```

### 예시 3: 서버-클라이언트 방식

```bash
# 서버에서
python server_api.py --host 0.0.0.0 --port 5000

# 클라이언트에서 (라즈베리 파이)
# 1. 데이터 수집
python collect_human_demonstrations.py --env-type real --episodes 5

# 2. 데이터 업로드
python client_upload.py \
    --server http://SERVER_IP:5000 \
    --upload human_demos.pkl

# 3. 학습 요청
python client_upload.py \
    --server http://SERVER_IP:5000 \
    --train uploaded_data/demos_XXX.pkl \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4

# 4. 모델 다운로드
python client_upload.py \
    --server http://SERVER_IP:5000 \
    --download imitation_rl_model.pth

# 5. 추론
python run_ai_agent.py \
    --model imitation_rl_model.pth \
    --env-type real
```

---

## 문제 해결

### 데이터 필터링

`train_imitation_rl.py`는 자동으로 유효하지 않은 에피소드를 필터링합니다:
- `states`나 `actions`가 없는 에피소드
- 빈 에피소드
- 필터링된 에피소드 수가 출력됩니다

### 모델 로드 실패

- 모델 파일 경로 확인
- 모델과 에이전트의 `state_dim` 일치 확인 (자동 감지됨)
- 사전 학습 모델은 `--model` 파라미터로 지정 (기본값 없음)

### 학습 속도가 느릴 때

- GPU 사용 확인
- 배치 크기 조정
- 렌더링 비활성화
- 서버-클라이언트 방식 사용 (GPU 서버 활용)

### 메모리 부족

- 배치 크기 감소 (`--batch-size 32`)
- 에피소드 길이 제한
- 업데이트 주기 감소 (`--update-frequency 1024`)

### 일치율이 낮을 때

- 더 많은 데모 데이터 수집
- 더 많은 에폭 학습
- 학습률 조정
- Teacher Forcing으로 사전 학습 후 Imitation RL

---

## 참고 문서

- `README.md`: 전체 프로젝트 개요 및 하드웨어 설정

---

## 학습 방법 비교

| 방법 | 학습 속도 | 데이터 필요 | 실제 환경 필요 | Fine-tuning | 추천도 |
|-----|---------|-----------|--------------|------------|--------|
| **A3C** | 매우 빠름 | ❌ | ❌ | ⚠️ | ⭐⭐⭐⭐ |
| **PPO (CarRacing)** | 빠름 | ❌ | ❌ | ✅ | ⭐⭐⭐⭐⭐ |
| **PPO (Sim)** | 빠름 | ❌ | ❌ | ✅ | ⭐⭐⭐⭐ |
| **Teacher Forcing** | 매우 빠름 | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| **Imitation RL** | 보통 | ✅ | ❌* | ✅ | ⭐⭐⭐⭐⭐ |
| **Human Feedback** | 느림 | ❌ | ✅ | ✅ | ⭐⭐⭐ |

\* 서버-클라이언트 방식 사용 시 실제 환경 불필요

**권장 순서**: PPO (CarRacing) → Teacher Forcing → Imitation RL → Human Feedback

---

이 가이드를 따라하면 효과적으로 RC Car를 학습시킬 수 있습니다! 🚗

