# 서버-클라이언트 아키텍처 사용 가이드

## 개요

라즈베리 파이에서 데이터 수집 및 추론만 수행하고, 학습은 서버(GPU)에서 수행하는 구조

## 아키텍처

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

## 설치

### 서버 측 (GPU 서버)

```bash
# Flask 및 의존성 설치
pip install flask flask-cors requests

# 서버 실행
python server_api.py --host 0.0.0.0 --port 5000
```

### 클라이언트 측 (라즈베리 파이)

```bash
# requests 설치
pip install requests

# 클라이언트 스크립트는 이미 포함됨
```

## 사용 방법

### 1단계: 서버 시작 (GPU 서버)

```bash
# 서버 실행
python server_api.py --host 0.0.0.0 --port 5000

# 서버가 시작되면:
# 🚀 서버 시작: http://0.0.0.0:5000
# 📁 업로드 폴더: uploaded_data
# 📁 모델 폴더: trained_models
```

### 2단계: 데이터 수집 (라즈베리 파이)

```bash
# 라즈베리 파이에서 데이터 수집
python collect_human_demonstrations.py \
    --env-type real \
    --port /dev/ttyACM0 \
    --episodes 5 \
    --output human_demos.pkl
```

### 3단계: 데이터 업로드 (라즈베리 파이)

```bash
# 서버로 데이터 업로드
python client_upload.py \
    --server http://192.168.1.100:5000 \
    --upload human_demos.pkl
```

**출력 예시:**
```
📤 데이터 업로드 중: human_demos.pkl
✅ 업로드 성공:
   파일: demos_20240101_120000.pkl
   에피소드: 5
   스텝: 250
   파일 경로: uploaded_data/demos_20240101_120000.pkl
```

### 4단계: 학습 요청 (라즈베리 파이 또는 서버)

```bash
# 서버에서 학습 시작
python client_upload.py \
    --server http://192.168.1.100:5000 \
    --train uploaded_data/demos_20240101_120000.pkl \
    --epochs 100
```

또는 서버에서 직접:

```bash
# 서버에서 직접 학습
python train_with_teacher_forcing.py \
    --demos uploaded_data/demos_20240101_120000.pkl \
    --pretrain-epochs 100
```

### 5단계: 모델 다운로드 (라즈베리 파이)

```bash
# 최신 모델 다운로드
python client_upload.py \
    --server http://192.168.1.100:5000 \
    --download latest_model.pth
```

### 6단계: 추론 실행 (라즈베리 파이)

```bash
# 다운로드한 모델로 추론
python run_ai_agent.py \
    --model latest_model.pth \
    --env-type real \
    --port /dev/ttyACM0
```

## API 엔드포인트

### 서버 API

- `GET /api/health`: 서버 상태 확인
- `POST /api/upload_data`: 데이터 파일 업로드
- `POST /api/train/supervised`: Supervised Learning 학습
- `POST /api/train/ppo`: PPO 강화학습
- `GET /api/model/latest`: 최신 모델 다운로드
- `GET /api/model/list`: 모델 목록 조회
- `POST /api/inference`: 실시간 추론 (선택)

## 전체 워크플로우

```bash
# 1. 서버 시작 (GPU 서버)
python server_api.py --host 0.0.0.0 --port 5000

# 2. 데이터 수집 (라즈베리 파이)
python collect_human_demonstrations.py --env-type real --episodes 5

# 3. 데이터 업로드 (라즈베리 파이)
python client_upload.py --server http://SERVER_IP:5000 --upload human_demos.pkl

# 4. 학습 요청 (라즈베리 파이 또는 서버)
python client_upload.py --server http://SERVER_IP:5000 --train uploaded_data/demos_XXX.pkl --epochs 100

# 5. 모델 다운로드 (라즈베리 파이)
python client_upload.py --server http://SERVER_IP:5000 --download latest_model.pth

# 6. 추론 실행 (라즈베리 파이)
python run_ai_agent.py --model latest_model.pth --env-type real
```

## 네트워크 설정

### 서버 IP 확인

```bash
# 서버에서 IP 확인
hostname -I
# 또는
ip addr show
```

### 방화벽 설정

```bash
# 서버에서 포트 5000 열기 (Ubuntu/Debian)
sudo ufw allow 5000/tcp

# 또는 iptables
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
```

## 문제 해결

### 서버 연결 실패

```bash
# 서버 상태 확인
python client_upload.py --server http://SERVER_IP:5000 --health

# 네트워크 연결 확인
ping SERVER_IP
curl http://SERVER_IP:5000/api/health
```

### 모델 다운로드 실패

```bash
# 모델 목록 확인
python client_upload.py --server http://SERVER_IP:5000 --list
```

## 보안 고려사항

1. **인증 추가**: 프로덕션 환경에서는 JWT 토큰 등 인증 추가
2. **HTTPS 사용**: 민감한 데이터 전송 시 HTTPS 사용
3. **방화벽 설정**: 필요한 IP만 접근 허용

