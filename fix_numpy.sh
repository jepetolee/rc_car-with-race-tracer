#!/bin/bash
# NumPy 버전 다운그레이드 스크립트

echo "=========================================="
echo "NumPy 버전 다운그레이드"
echo "=========================================="

echo "현재 NumPy 버전 확인..."
python -c "import numpy; print(f'현재 버전: {numpy.__version__}')" 2>/dev/null || echo "NumPy가 설치되어 있지 않습니다."

echo ""
echo "NumPy 제거 중..."
pip uninstall -y numpy

echo ""
echo "NumPy 1.24.3 설치 중 (라즈베리 파이 호환 버전)..."
pip install numpy==1.24.3 --no-cache-dir

echo ""
echo "설치 확인..."
python -c "import numpy; print(f'✅ 설치된 버전: {numpy.__version__}')"

echo ""
echo "=========================================="
echo "완료! 이제 다시 테스트해보세요:"
echo "  python simple_test.py"
echo "=========================================="

