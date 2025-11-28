#!/bin/bash
# A3C 학습 스크립트 (nohup 로그 포함)

PYTHONUNBUFFERED=1 nohup python3 train_a3c.py \
    --use-recurrent \
    --update-frequency 2048 \
    --num-workers 20 \
    --total-steps 1000000 \
    --state-dim 784 \
    --hidden-dim 256 \
    --entropy-coef 0.02 \
    --gae-lambda 0.95 \
    > training_a3c.log 2>&1 &

echo "학습 시작됨! PID: $!"
echo "로그 확인: tail -f training_a3c.log"
