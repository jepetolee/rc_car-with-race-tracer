#!/usr/bin/env python3
"""
QR 코드 분류 CNN 모델 훈련 스크립트

수집한 데이터로 CNN 모델을 훈련합니다.

사용법:
    python train_qr_cnn.py --data-dir qr_dataset --epochs 50
    python train_qr_cnn.py --data-dir qr_dataset --model-type small --epochs 30
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm

from qr_cnn_model import create_model


class QRDataset(Dataset):
    """
    QR 코드 데이터셋
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: 데이터 디렉토리 (qr_present, qr_absent 하위 디렉토리 포함)
            transform: 이미지 변환 (augmentation 등)
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 이미지 파일과 라벨 수집
        self.images = []
        self.labels = []
        
        # QR 있음 이미지
        qr_dir = os.path.join(data_dir, "qr_present")
        if os.path.exists(qr_dir):
            for filename in os.listdir(qr_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(qr_dir, filename))
                    self.labels.append(1)  # QR 있음
        
        # QR 없음 이미지
        no_qr_dir = os.path.join(data_dir, "qr_absent")
        if os.path.exists(no_qr_dir):
            for filename in os.listdir(no_qr_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(no_qr_dir, filename))
                    self.labels.append(0)  # QR 없음
        
        print(f"데이터셋 로드 완료:")
        print(f"  QR 있음: {sum(self.labels)}장")
        print(f"  QR 없음: {len(self.labels) - sum(self.labels)}장")
        print(f"  총: {len(self.images)}장")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_path = self.images[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {img_path}")
        
        # 이미지 크기 조정 (필요한 경우)
        if img.shape != (320, 320):
            img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # (H, W) -> (1, H, W) 채널 추가
        img = np.expand_dims(img, axis=0)
        
        # Transform 적용
        if self.transform:
            img = self.transform(torch.from_numpy(img))
        else:
            img = torch.from_numpy(img)
        
        # 라벨
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 훈련"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="훈련 중"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="검증 중"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(
        description='QR 코드 분류 CNN 모델 훈련',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 훈련
  python train_qr_cnn.py --data-dir qr_dataset --epochs 50
  
  # 작은 모델로 훈련
  python train_qr_cnn.py --data-dir qr_dataset --model-type small --epochs 30
  
  # 학습률 조정
  python train_qr_cnn.py --data-dir qr_dataset --lr 0.001 --epochs 50
        """
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='데이터 디렉토리 경로')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'small'],
                        help='모델 타입 (기본: standard)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='훈련 에폭 수 (기본: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='배치 크기 (기본: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률 (기본: 0.001)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='검증 데이터 비율 (기본: 0.2)')
    parser.add_argument('--save-dir', type=str, default='trained_models',
                        help='모델 저장 디렉토리 (기본: trained_models)')
    parser.add_argument('--save-name', type=str, default=None,
                        help='모델 저장 이름 (기본: qr_cnn_{model_type}_{timestamp}.pth)')
    parser.add_argument('--resume', type=str, default=None,
                        help='이전 모델에서 재개할 체크포인트 경로')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터셋 로드
    print("\n데이터셋 로드 중...")
    dataset = QRDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("❌ 데이터가 없습니다. 먼저 collect_qr_data.py로 데이터를 수집하세요.")
        sys.exit(1)
    
    # Train/Val 분할
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n데이터 분할:")
    print(f"  훈련: {train_size}장")
    print(f"  검증: {val_size}장")
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    # 모델 생성
    print(f"\n모델 생성 중... (타입: {args.model_type})")
    model = create_model(model_type=args.model_type, input_size=320, num_classes=2)
    model = model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 재개
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"체크포인트 로드: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"에폭 {start_epoch}부터 재개")
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 모델 저장 이름
    if args.save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_name = f"qr_cnn_{args.model_type}_{timestamp}.pth"
    
    save_path = os.path.join(args.save_dir, args.save_name)
    best_save_path = os.path.join(args.save_dir, f"qr_cnn_{args.model_type}_best.pth")
    
    print(f"\n훈련 시작...")
    print(f"  에폭: {args.epochs}")
    print(f"  배치 크기: {args.batch_size}")
    print(f"  학습률: {args.lr}")
    print(f"  모델 저장 경로: {save_path}")
    print("=" * 60)
    
    # 훈련 루프
    train_history = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n에폭 {epoch+1}/{args.epochs}")
        
        # 훈련
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 결과 출력
        print(f"  훈련 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  검증 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 히스토리 저장
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # 최고 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'model_type': args.model_type,
                'train_history': train_history
            }, best_save_path)
            print(f"  ✅ 최고 모델 저장: {best_save_path} (검증 정확도: {val_acc:.2f}%)")
        
        # 주기적 저장
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'model_type': args.model_type,
                'train_history': train_history
            }, save_path)
            print(f"  💾 체크포인트 저장: {save_path}")
    
    # 최종 모델 저장
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'model_type': args.model_type,
        'train_history': train_history
    }, save_path)
    
    print("\n" + "=" * 60)
    print("✅ 훈련 완료!")
    print(f"  최고 검증 정확도: {best_val_acc:.2f}%")
    print(f"  최종 모델: {save_path}")
    print(f"  최고 모델: {best_save_path}")


if __name__ == "__main__":
    main()

