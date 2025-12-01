# RC Car 자율주행 프로젝트

## 🎥 시연 영상

프로젝트의 실제 동작을 확인할 수 있는 시연 영상입니다.

### 시연 영상: RC Car 자율주행 및 QR 코드 감지

![RC Car 시연 영상](rc_car.gif)

**영상 내용:**
- RC Car의 자율주행 동작
- 선로 추적 및 주행
- QR 코드 감지 및 자동 정지 기능
- CNN 기반 QR 코드 분류 시스템

**시연 시나리오:**
1. RC Car가 선로를 따라 자율주행
2. 선로에 배치된 QR 코드 감지
3. QR 코드 감지 시 자동으로 4초간 정지
4. 정지 후 자동으로 주행 재개

---

## 목차

0. [시연 영상](#-시연-영상)
1. [시스템 개요 및 Arduino↔Raspberry Pi 명령 흐름](#1-시스템-개요-및-arduino↔raspberry-pi-명령-흐름)
2. [데이터 수집과 유틸리티](#2-데이터-수집과-유틸리티)
3. [학습 방법 개요와 주요 파라미터](#3-학습-방법-개요와-주요-파라미터)
4. [권장 학습 파이프라인](#4-권장-학습-파이프라인)
5. [학습 방법별 상세 가이드](#5-학습-방법별-상세-가이드)
6. [사전학습 모델과 현장 Teacher Forcing 운용](#6-사전학습-모델과-현장-teacher-forcing-운용)
7. [서버 기반 학습 제어(REST API + client_upload.py)](#7-서버-기반-학습-제어rest-api--client_uploadpy)
8. [문제 해결, 액션 정의, 참고 자료](#8-문제-해결-액션-정의-참고-자료)
   - [학습된 모델로 시험 주행 (QR 코드 감지 포함)](#83-학습된-모델로-시험-주행)

---

## 1. 시스템 개요 및 Arduino↔Raspberry Pi 명령 흐름

### 1.1 하드웨어 구성
- **Arduino + Adafruit Motor Shield**: 좌/우 DC 모터 제어 (`test.ino` 업로드)
- **Raspberry Pi**: 카메라 + 제어 스크립트 실행
- **USB 시리얼**: Arduino와 Raspberry Pi/PC 간 통신 (9600 baud)
- **카메라**: `picamera2` 기반 320×320 → 16×16 처리

### 1.2 명령어 체계

| 명령 | 의미 | 예시 |
|------|------|------|
| `F[속도]` | 전진 | `F255` |
| `L[속도]` | 좌회전 + 가속 | `L150` |
| `R[속도]` | 우회전 + 가속 | `R150` |
| `S` | **뒤로 가기** | `S` |
| `B[속도]` | 뒤로 가기 | `B200` |
| `stop` (텍스트) | 뒤로 가기 | `stop` |
| `X` | 브레이크 (즉시 정지) | `X` |

> **참고:** 
> - `S` 명령과 `stop` 텍스트는 **뒤로 가기**로 동작합니다
> - 완전 정지가 필요한 경우 `X` (Brake) 명령을 사용하세요
> - `B[속도]` 명령으로 속도 지정 뒤로 가기 가능

이와 별도로 CarRacing 호환을 위해 **이산 액션 0~4**도 지원합니다. `A0` 혹은 숫자 `0`만 보내도 되며, 매핑은 다음과 같습니다: `0/4=정지`, `1=우+가스`, `2=좌+가스`, `3=직진 가스`.

- 속도 범위: 0~255 (PWM)
- 명령은 `\n`으로 종료
- Python 측에서 `pyserial`로 문자열 송신

### 1.3 Python 제어 스크립트
- `rc_car_controller.py --mode interactive`: 키보드 `w/a/s/d/x` 입력을 즉시 송신
- `rc_car_controller.py --mode demo`: 전/후/좌/우/정지 순차 테스트
- `rc_car_interface.py`: 카메라 캡처 + 16×16 전처리 + 추론 루프 보조

### 1.4 카메라 준비
1. `sudo apt-get install python3-picamera2`
2. 가상환경이 필요하면 `python3 -m venv --system-site-packages venv`
3. 미리보기:
   ```bash
   python raspberry_pi_camera.py --mode preview --show-processed
   ```
4. 테스트/캡처 모드: `--mode capture`, `--mode test`
5. 문제 발생 시 `sudo raspi-config`에서 Camera Enable, `vcgencmd get_camera`로 상태 확인

---

## 2. 데이터 수집과 유틸리티

### 2.1 사람 데모 수집 (`collect_human_demonstrations.py`)
```bash
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --episodes 5 \
    --output uploaded_data/human_demos.pkl
```
- 조작키: `w`(직진) / `a`(좌+가속) / `d`(우+가속) / `s`(정지) / `x`(브레이크) / `q`(에피소드 종료)
- 저장 항목: `states`(16×16 이미지), `actions`(0~4), `rewards`, `dones`, `timestamps`

#### 보상 계산 요약 (`rc_car_env.py`)
- 중앙 밝기 기반 차선 추적 보상 (최대 0.5)
- 속도 유지 보상 (0.3)
- 프레임 안정성 (0.2), 방향 일관성 (0.1)
- 너무 느린 경우 -0.5 페널티, 전진 보너스 +0.1
- **Teacher Forcing/Imitation RL은 이 보상을 사용하지 않고 상태-액션 또는 일치율만 사용**하지만 데이터에는 저장되어 후처리에 활용 가능

### 2.2 데모 병합 (`merge_demo_data.py`)
```bash
# 여러 파일 병합
python merge_demo_data.py -i demos_a.pkl demos_b.pkl -o merged.pkl

# 패턴 또는 디렉토리
python merge_demo_data.py -p "uploaded_data/demos_*.pkl" -o merged.pkl
python merge_demo_data.py -d uploaded_data -o merged.pkl
```
- 길이 불일치 자동 보정, 빈 에피소드 필터링, 메타데이터 기록(`merged_from_files`, `total_steps` 등)

### 2.3 데이터 점검
```bash
python check_data_size.py uploaded_data/human_demos.pkl
```
- 총 에피소드/스텝, 상태 차원, 결측 여부 확인

### 2.4 QR 코드 데이터 수집 및 CNN 분류

QR 코드를 CNN으로 분류하여 선로에 QR 코드가 있으면 멈추는 기능을 구현합니다.

#### 2.4.1 QR 데이터 수집 (`collect_qr_data.py`)

```bash
# 대화형 모드 (사용자가 직접 라벨 입력)
python collect_qr_data.py --output-dir qr_dataset

# 자동 모드 (OpenCV QR 감지기로 자동 라벨링)
python collect_qr_data.py --output-dir qr_dataset --auto-label --num-images 200
```

**대화형 모드 조작키:**
- `q` 또는 `1`: QR 코드 있음으로 저장
- `n` 또는 `0`: QR 코드 없음으로 저장
- `s`: 통계 보기
- `x` 또는 ESC: 종료

**데이터 구조:**
```
qr_dataset/
├── qr_present/      # QR 코드가 있는 이미지들
├── qr_absent/       # QR 코드가 없는 이미지들
└── metadata.json    # 메타데이터 (통계 등)
```

#### 2.4.2 CNN 모델 훈련 (`train_qr_cnn.py`)

```bash
# 기본 훈련
python train_qr_cnn.py --data-dir qr_dataset --epochs 50

# 작은 모델로 훈련 (빠른 추론)
python train_qr_cnn.py --data-dir qr_dataset --model-type small --epochs 30

# 학습률 조정
python train_qr_cnn.py --data-dir qr_dataset --lr 0.001 --epochs 50
```

**주요 옵션:**
- `--data-dir`: 데이터 디렉토리 경로 (필수)
- `--model-type`: 모델 타입 (`standard` 또는 `small`, 기본: `standard`)
- `--epochs`: 훈련 에폭 수 (기본: 50)
- `--batch-size`: 배치 크기 (기본: 16)
- `--lr`: 학습률 (기본: 0.001)
- `--val-split`: 검증 데이터 비율 (기본: 0.2)

**Augmentation (자동 적용):**
데이터가 부족한 경우를 대비해 자동으로 augmentation이 적용됩니다:
- 회전: ±15도
- 이동: 10%
- 좌우 반전: 50% 확률
- 상하 반전: 50% 확률
- 밝기/대비 조정: ±20%
- 노이즈 추가: 30% 확률

> **참고:** 128장 정도의 작은 데이터셋에서도 augmentation 덕분에 효과적인 학습이 가능합니다.

**출력:**
- 최고 모델: `trained_models/qr_cnn_{model_type}_best.pth`
- 최종 모델: `trained_models/qr_cnn_{model_type}_{timestamp}.pth`

#### 2.4.3 CNN 기반 QR 감지 및 차량 제어 (`detect_qr_with_cnn.py`)

```bash
# 하드웨어 제어 없이 감지만 테스트
python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --no-hardware

# 하드웨어 제어 포함 (QR 감지 시 차량 정지)
python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --with-hardware

# 임계값 조정
python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --threshold 0.7
```

**주요 옵션:**
- `--model`: 훈련된 모델 경로 (필수)
- `--model-type`: 모델 타입 (`standard` 또는 `small`, 기본: `standard`)
- `--no-hardware`: 하드웨어 제어 없이 감지만 테스트
- `--with-hardware`: 하드웨어 제어 포함 테스트
- `--duration`: 테스트 지속 시간 (초, 기본: 60)
- `--threshold`: 감지 임계값 (기본: 0.5)
- `--stop-duration`: QR 감지 시 정지 시간 (초, 기본: 4.0)

#### 2.4.4 QR 데이터 서버 업로드 (`upload_qr_data.py`)

수집한 QR 데이터를 서버로 스트리밍 전송합니다.

```bash
# 디렉토리에서 수집한 데이터 업로드
python upload_qr_data.py --server 192.168.1.100:5000 --data-dir qr_dataset

# 실시간 스트리밍 (카메라에서 직접 전송)
python upload_qr_data.py --server 192.168.1.100:5000 --stream --duration 300

# 스트리밍 간격 조정
python upload_qr_data.py --server 192.168.1.100:5000 --stream --interval 0.5
```

**주요 옵션:**
- `--server`: 서버 URL (기본: http://localhost:5000)
- `--data-dir`: 업로드할 데이터 디렉토리
- `--stream`: 실시간 스트리밍 모드
- `--interval`: 스트리밍 모드에서 이미지 캡처 간격(초, 기본: 1.0)
- `--duration`: 스트리밍 모드에서 지속 시간(초, 0=무한, 기본: 60)

**서버 API 엔드포인트:**
- `POST /api/upload_qr_data`: QR 데이터 (이미지 배치) 업로드
  - 요청: `images` (base64 인코딩), `labels` (0 또는 1), `metadata` (선택)
  - 응답: `saved_count`, `total_count`

**전체 워크플로우:**
```bash
# 1. 데이터 수집 (라즈베리 파이)
python collect_qr_data.py --output-dir qr_dataset

# 2. 데이터를 서버로 업로드 (라즈베리 파이)
python upload_qr_data.py --server SERVER_IP:5000 --data-dir qr_dataset

# 3. 서버에서 모델 훈련
python train_qr_cnn.py --data-dir qr_dataset --epochs 50

# 4. 훈련된 모델을 라즈베리 파이로 다운로드 후 사용
python detect_qr_with_cnn.py --model qr_cnn_best.pth --with-hardware
```

---

## 3. 학습 방법 개요와 주요 파라미터

| 방법 | 스크립트 | 목적 | 대표 파라미터 |
|------|----------|------|----------------|
| **TRM-DQN (Carracing)** | `train_a3c.py` | Carracing 기반 TRM-DQN | `--state-dim`, `--action-dim`, `--max-episodes`, `--eps-start`, `--eps-decay` |
| **TRM-DQN (Sim/Real)** | `train_ppo.py` | 시뮬/실기 환경 TRM-DQN | `--env-type`, `--max-episodes`, `--eps-*`, `--target-update-interval`, `--save-interval` |
| **Teacher Forcing (TRM-DQN)** | `train_with_teacher_forcing.py` | 데모 기반 Supervised + Offline Q-learning | `--pretrain-epochs`, `--batch-size`, `--offline-steps`, `--learning-rate` |
| **Imitation RL (TRM-DQN Offline)** | `train_imitation_rl.py` | Teacher Forcing 후 Fine-tuning, 오프라인 Q-learning | `--epochs`, `--updates-per-epoch`, `--batch-size`, `--learning-rate`, `--model` |
| **Human Feedback** | `train_human_feedback.py` | 사람 평가 기반 RL | `--model`, `--num-episodes`, `--port`, `--save-path`, `--score-decay` |

추가적으로 `train_with_teacher_forcing.py`의 `--offline-steps` 옵션을 사용하면 Teacher Forcing 이후 곧바로 오프라인 Q-learning을 이어서 실행할 수 있습니다.

---

## 4. 권장 학습 파이프라인

### 4.1 시뮬레이션 중심 (권장)
```
1. train_ppo.py --env-type carracing (또는 train_a3c.py) 로 사전학습
2. collect_human_demonstrations.py 로 실제 데이터 수집
3. train_with_teacher_forcing.py 로 Supervised 사전학습
4. train_imitation_rl.py 로 Fine-tuning (필요 시)
5. train_human_feedback.py 로 추가 보정 (선택)
6. run_ai_agent.py 또는 server_api 추론
```

### 4.2 실제 환경 중심
```
1. 즉시 데모 데이터 수집
2. Teacher Forcing (필수)
3. Imitation RL
4. Human Feedback (사람 평가)
5. 배포/추론
```

각 단계에서 생성되는 모델 파일(`dqn_model_*.pth`, `pretrained_*.pth`, `imitation_dqn_*.pth`)을 명확히 관리하세요.

---

## 5. 학습 방법별 상세 가이드

### 5.1 TRM-DQN Carracing (`train_a3c.py`)
```bash
python train_a3c.py \
    --max-episodes 2000 \
    --max-episode-steps 1000 \
    --save-interval 100
```
- CarRacing Gym 환경을 이용한 TRM-DQN 학습
- epsilon 스케줄(`--eps-*`)과 target network 주기를 조절

### 5.2 TRM-DQN (시뮬/실기, `train_ppo.py`)
```bash
# Carracing
python train_ppo.py --env-type carracing --max-episodes 2000 --save-interval 50

# 시뮬레이터
python train_ppo.py --env-type sim --max-episodes 500 --save-interval 50
```
- `--env-type`으로 carracing/sim/real 선택
- epsilon 스케줄(`--eps-*`)과 target network 갱신 주기를 조절

### 5.3 Teacher Forcing + TRM-DQN (`train_with_teacher_forcing.py`)
```bash
python train_with_teacher_forcing.py \
    --demos uploaded_data/demos.pkl \
    --pretrain-epochs 50 \
    --pretrain-batch-size 64 \
    --pretrain-lr 3e-4 \
    --pretrain-save pretrained_model.pth
```
- TRM 기반 Q-network를 데모로 Supervised pretrain
- `--offline-steps` 설정 시 데모를 리플레이 버퍼에 채워 오프라인 Q-learning 실행
- 주요 파라미터: `--pretrain-epochs`, `--batch-size`, `--learning-rate`, `--offline-steps`

### 5.4 Imitation RL (오프라인 Q-learning, `train_imitation_rl.py`)
```bash
python train_imitation_rl.py \
    --demos uploaded_data/demos.pkl \
    --model pretrained_model.pth \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --save trained_models/imitation_rl_latest.pth
```
- 데모에서 상태·액션·다음 상태를 추출하여 리플레이 버퍼에 적재
- `--updates-per-epoch` 만큼 Q-learning을 반복하며 TRM-DQN을 미세 조정
- 평가 시 데모와의 액션 일치율을 출력

### 5.5 Human Feedback (`train_human_feedback.py`)
```bash
python train_human_feedback.py \
    --model pretrained_model.pth \
    --port /dev/ttyACM0 \
    --num-episodes 10 \
    --save-path trained_models/feedback_model.pth
```
- 모델 주행을 보여주고 사용자가 0.0~1.0 점수를 입력하면 해당 점수를 리워드로 사용
- `--updates-per-episode`로 피드백 후 Q-learning 반복 횟수를 지정
- 실제 하드웨어 Fine-tuning을 위한 절차

---

## 6. 사전학습 모델과 현장 Teacher Forcing 운용

1. **기본 모델**: `trained_models/pretrained_*.pth` (TRM-DQN)  
   - Teacher Forcing/Imitation RL, 서버 API 모두 DQN 체크포인트를 사용
2. **Teacher Forcing CLI**:
   ```bash
   python3 train_with_teacher_forcing.py \
       --demos uploaded_data/demos.pkl \
       --pretrain-epochs 20 \
       --pretrain-save trained_models/pretrained_$(date +%Y%m%d_%H%M%S).pth
   ```
3. **현장 재학습 절차 (TRM-DQN)**:
   - 라즈베리 파이로 데모 수집
   - `client_upload.py --server ... --train-supervised ...` 로 서버에서 학습
   - 결과 모델을 다시 다운로드 후 추론 (`run_ai_agent.py --model ...`)
4. **모델/파라미터 자동 감지**:
   - 모든 스크립트와 서버 엔드포인트가 `state_dim`을 데모에서 자동 추정
   - 학습률, 배치, 에폭, 업데이트 횟수는 CLI/JSON 인자로 조정

---

## 7. 서버 기반 학습 제어(REST API + client_upload.py)

### 7.1 서버 실행
```bash
python server_api.py --host 0.0.0.0 --port 5000
```
- 업로드 폴더: `uploaded_data/`
- 모델 폴더: `trained_models/`
- GPU 서버에서 실행 권장

### 7.2 client_upload.py 워크플로우
```bash
# 서버 상태 확인
python3 client_upload.py --server http://SERVER_IP:5000 --health

# 데이터 업로드
python3 client_upload.py --server http://SERVER_IP:5000 --upload demos.pkl

# Teacher Forcing 학습 요청
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --train-supervised uploaded_data/demos.pkl \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4

# Imitation RL 학습 요청
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --train uploaded_data/demos.pkl \
    --pretrain-model trained_models/pretrained_latest.pth \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 3e-4

# 모델 목록 조회
python3 client_upload.py --server http://SERVER_IP:5000 --list

# 최신 모델 다운로드
python3 client_upload.py --server http://SERVER_IP:5000 --download latest_model.pth
```
- `--train`와 `--train-imitation`은 같은 동작
- Teacher Forcing 호출 시에도 이제 `learning_rate`, `model_path` 전달 가능

### 7.3 학습된 모델 다운로드

학습 완료 후 서버에 저장된 모델을 다운로드하는 방법입니다.

#### 방법 1: client_upload.py 사용 (추천)

```bash
# 1. 사용 가능한 모델 목록 조회
python3 client_upload.py --server http://SERVER_IP:5000 --list

# 출력 예시:
# 📋 사용 가능한 모델 (5개):
#    - pretrained_20251129_190816.pth (12345678 bytes, 2025-11-29T19:08:16)
#    - imitation_rl_20251129_191640.pth (23456789 bytes, 2025-11-29T19:16:40)
#    - dqn_model_20251129_180000.pth (34567890 bytes, 2025-11-29T18:00:00)
#    ...

# 2. 최신 모델 다운로드 (가장 최근에 저장된 모델)
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --download latest_model.pth

# 3. 특정 모델 다운로드 (REST API 직접 호출, 아래 참고)
```

**참고**: 학습 요청 후 응답에서 `model_path`를 확인할 수 있습니다:
```json
{
    "status": "success",
    "model_path": "trained_models/pretrained_20251129_190816.pth",
    "epochs": 20
}
```

#### 방법 2: REST API 직접 호출

```bash
# 모델 목록 조회
curl http://SERVER_IP:5000/api/model/list

# 최신 모델 다운로드
curl -O http://SERVER_IP:5000/api/model/latest

# 또는 파일명 지정
curl http://SERVER_IP:5000/api/model/latest -o my_model.pth

# 특정 모델 다운로드 (파일명으로)
curl -O http://SERVER_IP:5000/api/model/download/pretrained_20251129_190816.pth
```

**응답 예시 (`/api/model/list`)**:
```json
{
    "models": [
        {
            "filename": "pretrained_20251129_190816.pth",
            "size": 12345678,
            "modified": "2025-11-29T19:08:16"
        },
        {
            "filename": "imitation_rl_20251129_191640.pth",
            "size": 23456789,
            "modified": "2025-11-29T19:16:40"
        }
    ]
}
```

#### 전체 워크플로우 예시

```bash
# 1. 데이터 업로드
python3 client_upload.py --server http://SERVER_IP:5000 --upload demos.pkl

# 2. Teacher Forcing 학습 요청
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --train-supervised uploaded_data/demos.pkl \
    --epochs 20 \
    --batch-size 64\
    --pretrain-model ./trained_models/dqn_multi_best_mark1.pth

# 응답에서 model_path 확인:
# "model_path": "trained_models/pretrained_20251129_190816.pth"

# 3. Imitation RL 학습 (Teacher Forcing 모델 사용)
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --train uploaded_data/demos.pkl \
    --pretrain-model trained_models/pretrained_20251129_190816.pth \
    --epochs 100

# 4. 모델 목록 확인
python3 client_upload.py --server http://SERVER_IP:5000 --list

# 5. 최신 모델 다운로드
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --download latest_model.pth

# 또는 특정 모델 다운로드 (curl 사용)
curl -O http://SERVER_IP:5000/api/model/download/imitation_rl_20251129_191640.pth
```

### 7.4 직접 REST 호출
```bash
# Teacher Forcing
curl -X POST http://SERVER_IP:5000/api/train/supervised \
  -H "Content-Type: application/json" \
  -d '{
        "file_path": "uploaded_data/demos.pkl",
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "model_path": "trained_models/pretrained_latest.pth"
      }'

# Imitation RL
curl -X POST http://SERVER_IP:5000/api/train/imitation_rl \
  -H "Content-Type: application/json" \
  -d '{
        "file_path": "uploaded_data/demos.pkl",
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.0003
      }'
```

### 7.5 파라미터 참고

| 엔드포인트 | 메서드 | 필수 | 선택/기본값 |
|------------|--------|------|-------------|
| `/api/train/supervised` | POST | `file_path` | `epochs`(100), `batch_size`(64), `learning_rate`(3e-4), `model_path`(선택) |
| `/api/train/imitation_rl` | POST | `file_path` | `model_path`(선택), `epochs`, `batch_size`, `learning_rate`, `updates_per_epoch`(1000) |
| `/api/upload_data` | POST | 파일 스트림 | 자동으로 `uploaded_data/demos_*.pkl` 저장 |
| `/api/model/list` | GET | 없음 | 사용 가능한 모든 모델 목록 반환 |
| `/api/model/latest` | GET | 없음 | 가장 최근에 저장된 모델 다운로드 |
| `/api/model/download/<filename>` | GET | `filename` | 특정 모델 파일 다운로드 (예: `pretrained_20251129_190816.pth`) |

응답에는 학습된 모델 경로나 Match Rate 등이 포함되며, 실패 시 `traceback`을 함께 제공하므로 `client_upload.py`가 콘솔에 상세 오류를 출력합니다.

---

## 8. 문제 해결, 액션 정의, 참고 자료

### 8.1 액션 정의 (이산 5개)

| ID | 설명 | 모터 상태 |
|----|------|-----------|
| 0 | 정지/Coast | 양쪽 RELEASE |
| 1 | 우회전 + 가속 | 좌측 빠름 / 우측 느림 |
| 2 | 좌회전 + 가속 | 좌측 느림 / 우측 빠름 |
| 3 | 직진 가속 | 양쪽 동일 전진 |
| 4 | 브레이크 | 역방향 또는 급정지 |

### 8.2 시리얼 & 카메라 트러블슈팅
- 포트 확인: `ls /dev/tty* | grep -E "(USB|ACM)"`, 권한: `sudo chmod 666 /dev/ttyUSB0`
- Arduino 응답 X: 시리얼 모니터 종료, 보드 리셋, 보드레이트 9600 확인
- 카메라 인식 X: `sudo raspi-config` > Interface Options > Camera > Enable, `vcgencmd get_camera`

### 8.3 학습된 모델로 시험 주행

학습이 완료된 모델을 사용하여 실제 RC Car를 제어하고 시험 주행하는 방법입니다.

#### 8.3.1 기본 시험 주행

```bash
# 기본 사용법 (실제 하드웨어)
python run_ai_agent.py \
    --model trained_models/imitation_rl_20251129_191640.pth \
    --env-type real \
    --port /dev/ttyACM0 \
    --delay 0.1 \
    --max-steps 1000

# 여러 에피소드 실행
python run_ai_agent.py \
    --model trained_models/imitation_rl_20251129_191640.pth \
    --env-type real \
    --episodes 5 \
    --delay 0.1
```

**주요 옵션:**
- `--model`: 학습된 모델 경로 (필수)
- `--env-type real`: 실제 하드웨어 환경 사용
- `--port /dev/ttyACM0`: Arduino 시리얼 포트
- `--delay 0.1`: 액션 간 지연 시간 (초, 기본: 0.1)
- `--max-steps 1000`: 최대 스텝 수
- `--episodes 5`: 실행할 에피소드 수
- `--qr-cnn-model`: QR CNN 모델 경로 (지정 시 CNN 사용, 미지정 시 OpenCV 사용)

#### 8.3.2 QR 코드 감지 기능

시험 주행 중 QR 코드를 감지하면 자동으로 차량이 **4초간 정지**합니다. 이 기능은 `run_ai_agent.py`에서 자동으로 활성화됩니다 (실제 하드웨어 환경일 때만).

**QR 코드 감지 방식:**
- **CNN 모델 사용 (권장)**: 훈련된 CNN 모델로 더 정확한 QR 코드 감지
- **OpenCV 기본 감지기**: CNN 모델 미지정 시 자동으로 사용

**QR 코드 감지 동작:**
1. 매 스텝마다 카메라 이미지에서 QR 코드 검사
2. QR 코드가 감지되고 차량이 이동 중이면 즉시 정지
3. 4초간 정지 후 자동 제어 재개
4. QR 코드 감지 정보가 로그에 출력됨 (CNN 사용 시 신뢰도 포함)

**CNN 모델을 사용한 시험 주행:**

```bash
# CNN 모델을 사용한 QR 코드 감지 포함 시험 주행
python run_ai_agent.py \
    --model trained_models/imitation_rl_20251129_191640.pth \
    --env-type real \
    --qr-cnn-model trained_models/qr_cnn_best.pth \
    --port /dev/ttyACM0 \
    --delay 0.1 \
    --episodes 5
```

**QR 코드 테스트 (독립 실행):**

```bash
# CNN 모델을 사용한 QR 코드 감지 테스트 (하드웨어 제어 없음)
python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --no-hardware

# 60초 동안 테스트
python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --no-hardware --duration 60

# 하드웨어 제어 포함 테스트 (QR 감지 시 차량 정지)
python detect_qr_with_cnn.py --model trained_models/qr_cnn_best.pth --with-hardware --duration 60

# 작은 모델 사용 및 임계값 조정
python detect_qr_with_cnn.py --model trained_models/qr_cnn_small_best.pth --model-type small --threshold 0.7
```

> **참고:** 
> - `--qr-cnn-model` 옵션을 지정하면 훈련된 CNN 모델을 사용하여 더 정확한 QR 코드 감지가 가능합니다.
> - CNN 모델 미지정 시 OpenCV의 기본 QR 감지기를 사용합니다.
> - CNN 모델은 `train_qr_cnn.py`로 훈련할 수 있으며, augmentation이 적용되어 128장의 데이터로도 효과적인 학습이 가능합니다.

#### 8.3.3 서버에서 다운로드한 모델로 시험 주행

```bash
# 1. 서버에서 모델 다운로드
python client_upload.py --server http://SERVER_IP:5000 --download latest_model.pth

# 2. 다운로드한 모델로 시험 주행
python run_ai_agent.py \
    --model latest_model.pth \
    --env-type real \
    --port /dev/ttyACM0 \
    --delay 0.1
```

#### 8.3.4 주의사항

1. **안전 확인**
   - 시험 주행 전 충분한 공간 확보
   - 차량이 장애물에 부딪히지 않도록 주변 정리
   - 긴급 정지를 위한 키보드 인터럽트 준비 (Ctrl+C)

2. **모델 호환성**
   - 모델이 `state_dim=784` (28×28 이미지를 784차원 벡터로)을 사용하는지 확인
   - TRM-DQN 체크포인트(`ppo_agent.DQNAgent`)만 지원

3. **QR 코드 감지**
   - QR 코드가 카메라 화면 전체에서 감지됨
   - 차량이 정지 상태일 때는 QR 코드 감지 시에도 추가 정지 없음
   - 동일 QR 코드의 중복 감지는 방지됨
   - CNN 모델 사용 시 신뢰도 정보가 함께 출력됨
   - `--qr-cnn-model` 옵션으로 CNN 모델 지정 가능

4. **디버깅**
   - `--quiet` 옵션을 제거하여 상세 로그 확인
   - 각 스텝의 액션, 리워드, 누적 리워드가 출력됨

### 8.4 수동 조작 방법

실제 RC Car를 키보드로 직접 조종하는 방법입니다. 학습 데이터 수집이나 간단한 테스트에 유용합니다.

#### 8.4.1 기본 키보드 조종 (데이터 저장 없음)

**간단한 테스트용 조종:**

```bash
# 기본 실행
python rc_car_controller.py --port /dev/ttyACM0 --mode interactive

# 속도 조절 포함
python rc_car_controller.py \
    --port /dev/ttyACM0 \
    --mode interactive \
    --delay 0.1
```

**조작키:**
- `w`: 전진 (Forward/Gas)
- `a`: 좌회전 + 가속 (Left + Gas)
- `d`: 우회전 + 가속 (Right + Gas)
- `s`: 정지 → **뒤로 가기** (Stop → Backward)
- `x`: 브레이크 (Brake/정지)
- `0-4`: 이산 액션 직접 입력
  - `0`: 정지
  - `1`: 우회전 + 가속
  - `2`: 좌회전 + 가속
  - `3`: 직진 가속
  - `4`: 브레이크
- `q`: 종료
- `speed [숫자]`: 속도 변경 (예: `speed 200`, 범위: 0-255)

**사용 예시:**
```
Enter command (w/a/d/s/x/0-4/q): w    # 전진
Enter command (w/a/d/s/x/0-4/q): a    # 좌회전
Enter command (w/a/d/s/x/0-4/q): s    # 뒤로 가기
Enter command (w/a/d/s/x/0-4/q): x    # 정지
Enter command (w/a/d/s/x/0-4/q): speed 150  # 속도 변경
Enter command (w/a/d/s/x/0-4/q): q    # 종료
```

#### 8.4.2 데모 데이터 수집하면서 조종

**학습용 데이터 수집:**

```bash
# 단일 에피소드 수집
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --output my_demo.pkl

# 여러 에피소드 수집
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --output my_demos.pkl \
    --episodes 5
    --episode-interval 10.0 \
```

**조작키:**
- `w`: 직진 (Action 3)
- `a`: 좌회전 + 가속 (Action 2)
- `d`: 우회전 + 가속 (Action 1)
- `s`: 정지 (Action 0)
- `x`: 브레이크 (Action 4)
- `q`: 에피소드 종료
- Enter: 다음 에피소드 시작

**저장되는 데이터:**
- 카메라 이미지 (states)
- 액션 (0-4)
- 리워드 (자동 계산)
- 타임스탬프

#### 8.4.3 명령어 매핑

| 키/명령 | Arduino 명령 | 동작 | 설명 |
|---------|-------------|------|------|
| `w` | `F[속도]` | 전진 | 직진 가속 |
| `a` | `L[속도]` | 좌회전 + 가속 | 왼쪽으로 회전하며 전진 |
| `d` | `R[속도]` | 우회전 + 가속 | 오른쪽으로 회전하며 전진 |
| `s` | `S` | **뒤로 가기** | 뒤로 이동 (기존 Stop) |
| `x` | `X` | 정지 | 브레이크/정지 |
| `stop` (텍스트) | `B` | 뒤로 가기 | 뒤로 이동 |
| `0-4` | `A[0-4]` | 이산 액션 | CarRacing 호환 액션 |

**참고:** 
- `s` 키와 `S` 명령은 **뒤로 가기**로 동작합니다 (안전을 위해 정지가 필요하면 `x` 키 사용)
- `stop` 텍스트 명령도 뒤로 가기로 처리됩니다
- `X` (Brake) 명령만 완전 정지

### 8.5 유용한 스크립트 모음
- `run_ai_agent.py`: 학습된 모델 추론 및 시험 주행
- `test_qr_detection.py`: QR 코드 감지 기능 테스트
- `upload_patches.py`: patch 단위 업로드
- `train_human_feedback.py`: 사람 평가 기반 학습
- `train_with_teacher_forcing.py`: Teacher Forcing + (선택) RL
- `merge_demo_data.py`: 데모 통합 (삭제하지 말 것)

### 8.4 README 정리 현황
- `README_TRAINING_PIPELINE.md`, `TEACHER_FORCING_IMITATION_RL_GUIDE.md`, `SERVER_TRAINING_GUIDE.md`의 모든 내용은 본 `README.md`에 통합되었습니다.
- 추가 문서가 필요한 경우 이 파일에서 섹션을 찾거나, 특정 스크립트의 docstring을 참고하세요.

---

## 라이선스

교육/연구 목적으로 자유롭게 사용할 수 있습니다. 프로젝트 개선 사항이나 버그 리포트는 이 저장소의 이슈로 남겨주세요.


