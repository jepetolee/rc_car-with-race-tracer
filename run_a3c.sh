#!/bin/bash
# Multi-Worker TRM-DQN 학습 스크립트 (A3C 스타일, nohup)

PYTHONUNBUFFERED=1 nohup python3 train_a3c.py \
    --state-dim 784 \
    --hidden-dim 256 \
    --latent-dim 256 \
    --max-steps 4000000 \
    --max-episode-steps 1000 \
    --num-workers 24 \
    --sync-interval 1000 \
    --replay-buffer 200000 \
    --batch-size 128 \
    --learning-rate 3e-4 \
    --eps-decay 300000 \
    --target-update-interval 2000 \
    --save-interval-steps 50000 \
    --use-tensorboard \
    > training_a3c.log 2>&1 &

echo "Multi-Worker DQN 학습 시작됨! PID: $!"
echo "워커 수: 16개"
echo "로그 확인: tail -f training_a3c.log"
