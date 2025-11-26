#!/bin/bash
# PyTorch 버전 확인 및 수정 스크립트

echo "=========================================="
echo "PyTorch 문제 진단 및 수정"
echo "=========================================="

echo "현재 PyTorch 버전 확인 시도..."
python -c "import torch; print(f'현재 버전: {torch.__version__}')" 2>/dev/null || echo "PyTorch 임포트 실패 (Bus error)"

echo ""
echo "시스템 정보 확인..."
uname -m
python -c "import platform; print(f'Python: {platform.python_version()}'); print(f'Platform: {platform.platform()}')"

echo ""
echo "=========================================="
echo "해결 방법:"
echo "1. PyTorch 제거:"
echo "   uv pip uninstall -y torch torchvision"
echo ""
echo "2. ARM용 PyTorch 설치 (라즈베리 파이용):"
echo "   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
echo ""
echo "또는"
echo ""
echo "3. 사전 빌드된 wheel 파일 사용:"
echo "   uv pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu"
echo "=========================================="

