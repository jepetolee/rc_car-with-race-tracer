# RC Car 자율주행 프로젝트


## 목차

1. [시스템 개요 및 Arduino↔Raspberry Pi 명령 흐름](#1-시스템-개요-및-arduino↔raspberry-pi-명령-흐름)
2. [데이터 수집과 유틸리티](#2-데이터-수집과-유틸리티)
3. [학습 방법 개요와 주요 파라미터](#3-학습-방법-개요와-주요-파라미터)
4. [권장 학습 파이프라인](#4-권장-학습-파이프라인)
5. [학습 방법별 상세 가이드](#5-학습-방법별-상세-가이드)
6. [사전학습 모델과 현장 Teacher Forcing 운용](#6-사전학습-모델과-현장-teacher-forcing-운용)
7. [서버 기반 학습 제어(REST API + client_upload.py)](#7-서버-기반-학습-제어rest-api--client_uploadpy)
8. [문제 해결, 액션 정의, 참고 자료](#8-문제-해결-액션-정의-참고-자료)

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
| `S` | 정지 (Coast) | `S` |
| `X` | 브레이크 (즉시 정지) | `X` |

> **참고:** 안전상의 이유로 레거시 후진 명령(`B[속도]`)은 펌웨어에서 제거되었습니다. 필요 시 Arduino 스케치를 수정해 다시 활성화할 수 있지만, 기본 분포에서는 전진 기반 조향만 지원합니다.

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

---

## 3. 학습 방법 개요와 주요 파라미터

| 방법 | 스크립트 | 목적 | 대표 파라미터 |
|------|----------|------|----------------|
| **A3C** | `train_a3c.py` | 멀티 프로세스 사전학습 | `--num-workers`, `--total-steps`, `--lr-actor`, `--lr-critic`, `--hidden-dim` |
| **PPO (CarRacing/Sim)** | `train_ppo.py` | 시뮬레이션 기반 PPO | `--env-type`, `--total-steps`, `--update-frequency`, `--update-epochs`, `--use-extended-actions` |
| **TRM + Teacher Forcing** | `train_with_teacher_forcing.py` | 상태·액션 Supervised 학습, TRM(Transformer-based Recurrent Model) 파라미터 공유 | `--pretrain-epochs`, `--pretrain-batch-size`, `--pretrain-lr`, `--n-cycles`, `--n-latent-loops`, `--n-deep-loops` |
| **Imitation RL (TRM-PPO)** | `train_imitation_rl.py` | Teacher Forcing 후 Fine-tuning, 일치율 보상 | `--epochs`, `--batch-size`, `--learning-rate`, `--model`, `--sequence-mode`, `--deep-supervision`, `--n-supervision-steps` |
| **Human Feedback** | `train_human_feedback.py` | 사람 평가 기반 RL | `--model`, `--num-episodes`, `--port`, `--save-path`, `--score-decay` |

추가적으로 `train_with_teacher_forcing.py`의 `--rl-steps` 옵션을 사용하면 Teacher Forcing → PPO Fine-tuning을 단일 스크립트에서 수행할 수 있습니다.

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

각 단계에서 생성되는 모델 파일(`a3c_model_best.pth`, `pretrained_*.pth`, `imitation_rl_*.pth`)을 명확히 관리하세요.

---

## 5. 학습 방법별 상세 가이드

### 5.1 A3C (`train_a3c.py`)
```bash
python train_a3c.py \
    --num-workers 4 \
    --total-steps 500000 \
    --save-path a3c_model_best.pth
```
- CarRacing Gym 환경 사용
- 다중 프로세스로 빠른 수렴
- 주요 옵션: `--entropy-coef`, `--gamma`, `--gae-lambda`

### 5.2 PPO (`train_ppo.py`)
```bash
# CarRacing
python train_ppo.py --env-type carracing --total-steps 500000 --save-path ppo_carracing.pth

# 시뮬레이터
python train_ppo.py --env-type sim --total-steps 200000 --save-path ppo_sim.pth
```
- `--render`로 시각화
- `--use-extended-actions` 활성화 시 RC Car 이산 액션에 맞춰짐

### 5.3 Teacher Forcing + TRM (`train_with_teacher_forcing.py`)
```bash
python train_with_teacher_forcing.py \
    --demos uploaded_data/demos.pkl \
    --pretrain-epochs 50 \
    --pretrain-batch-size 64 \
    --pretrain-lr 3e-4 \
    --pretrain-save pretrained_model.pth
```
- TRM(Transformer Reasoning Module) 기반 recurrent actor-critic
- 훈련 중 Loss/Accuracy/ETA 출력, TensorBoard `runs/teacher_forcing_*`
- 주요 Recurrent 파라미터:
  - `--n-cycles`: reasoning block 반복 횟수
  - `--n-latent-loops`, `--n-deep-loops`: latent 업데이트 제어
  - `--deep-supervision` 사용 시 step-wise backprop 수행
- `--rl-steps`와 `--rl-save`를 지정하면 사전학습 후 PPO Fine-tuning도 동시 실행 가능

### 5.4 Imitation RL (`train_imitation_rl.py`)
```bash
python train_imitation_rl.py \
    --demos uploaded_data/demos.pkl \
    --model pretrained_model.pth \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --save trained_models/imitation_rl_latest.pth
```
- 입력 데이터에서 `state_dim` 자동 감지, 빈 에피소드 필터링
- 기본적으로 `a3c_model_best.pth`를 시도해 로드
- 리워드: 일치 +1.0 / 불일치 -0.1
- 주요 옵션:
  - `--sequence-mode`: 에피소드 단위 학습
  - `--use-recurrent`: TRM Recurrent 모드
  - `--deep-supervision`, `--n-supervision-steps`: Latent carry-over 학습
  - `--max-grad-norm`, `--entropy-coef`, `--value-coef`
- 훈련 로그: 에폭 진행률, 배치별 Match Rate, Loss, ETA

### 5.5 Human Feedback (`train_human_feedback.py`)
```bash
python train_human_feedback.py \
    --model pretrained_model.pth \
    --port /dev/ttyACM0 \
    --num-episodes 10 \
    --save-path trained_models/feedback_model.pth
```
- 모델이 주행하는 동안 사용자가 0.0~1.0 점수를 입력하면 리워드로 사용
- `--score-decay`로 과거 점수 영향 조절
- 실제 환경 Fine-tuning용

---

## 6. 사전학습 모델과 현장 Teacher Forcing 운용

1. **기본 모델**: `a3c_model_best.pth`  
   - `train_imitation_rl.py`와 `server_api.py`에서 기본값으로 로드
2. **Teacher Forcing CLI**:
   ```bash
   python3 train_with_teacher_forcing.py \
       --demos uploaded_data/demos.pkl \
       --pretrain-epochs 20 \
       --pretrain-save trained_models/pretrained_$(date +%Y%m%d_%H%M%S).pth
   ```
3. **현장 재학습 절차**:
   - 라즈베리 파이로 데모 수집
   - `client_upload.py --server ... --train-supervised ...` 로 서버에서 학습
   - 결과 모델을 다시 다운로드 후 추론 (`run_ai_agent.py --model ...`)
4. **모델/파라미터 자동 감지**:
   - `train_with_teacher_forcing.py`와 서버 엔드포인트 모두 `state_dim`을 데모 데이터에서 계산
   - 학습률, 배치, 에폭은 JSON/CLI 인자로 조정

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
    --learning-rate 3e-4 \
    --pretrain-model a3c_model_best.pth

# Imitation RL 학습 요청
python3 client_upload.py \
    --server http://SERVER_IP:5000 \
    --train uploaded_data/demos.pkl \
    --pretrain-model trained_models/pretrained_xxx.pth \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 3e-4

# 모델 다운로드
python3 client_upload.py --server http://SERVER_IP:5000 --download latest_model.pth
```
- `--train`와 `--train-imitation`은 같은 동작
- Teacher Forcing 호출 시에도 이제 `learning_rate`, `model_path` 전달 가능

### 7.3 직접 REST 호출
```bash
# Teacher Forcing
curl -X POST http://SERVER_IP:5000/api/train/supervised \
  -H "Content-Type: application/json" \
  -d '{
        "file_path": "uploaded_data/demos.pkl",
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "model_path": "a3c_model_best.pth"
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

### 7.4 파라미터 참고

| 엔드포인트 | 필수 | 선택/기본값 |
|------------|------|-------------|
| `/api/train/supervised` | `file_path` | `epochs`(100), `batch_size`(64), `learning_rate`(3e-4), `model_path`(없으면 `a3c_model_best.pth` 탐색) |
| `/api/train/imitation_rl` | `file_path` | `model_path`(기본 `a3c_model_best.pth`), `epochs`, `batch_size`, `learning_rate` |
| `/api/upload_data` | 파일 스트림 | 자동으로 `uploaded_data/demos_*.pkl` 저장 |

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

### 8.3 유용한 스크립트 모음
- `run_ai_agent.py`: 학습된 모델 추론
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


