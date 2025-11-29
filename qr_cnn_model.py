#!/usr/bin/env python3
"""
QR 코드 분류를 위한 CNN 모델 정의

이진 분류 모델: QR 코드가 있는지 없는지를 판단합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QRCNN(nn.Module):
    """
    QR 코드 분류를 위한 간단한 CNN 모델
    
    입력: 320x320 그레이스케일 이미지
    출력: 2개 클래스 (QR 있음, QR 없음)
    """
    
    def __init__(self, input_size=320, num_classes=2):
        super(QRCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  # 320x320 -> 160x160
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 160x160 -> 80x80
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # 80x80 -> 40x40
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 40x40 -> 20x20
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 20x20 -> 10x10
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 10x10 -> 5x5
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 320, 320)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class QRCNNSmall(nn.Module):
    """
    더 작은 CNN 모델 (빠른 추론용)
    
    입력: 160x160 그레이스케일 이미지 (리사이즈 가능)
    출력: 2개 클래스 (QR 있음, QR 없음)
    """
    
    def __init__(self, input_size=160, num_classes=2):
        super(QRCNNSmall, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)  # 160x160 -> 80x80
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 80x80 -> 40x40
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)  # 40x40 -> 20x20
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 20x20 -> 10x10
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 10x10 -> 5x5
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 160, 160)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


def create_model(model_type='standard', input_size=320, num_classes=2):
    """
    모델 생성 헬퍼 함수
    
    Args:
        model_type: 'standard' 또는 'small'
        input_size: 입력 이미지 크기
        num_classes: 분류 클래스 수
    
    Returns:
        모델 인스턴스
    """
    if model_type == 'small':
        return QRCNNSmall(input_size=input_size, num_classes=num_classes)
    else:
        return QRCNN(input_size=input_size, num_classes=num_classes)


if __name__ == "__main__":
    # 모델 테스트
    print("QRCNN 모델 테스트")
    print("=" * 60)
    
    # Standard model
    model = QRCNN(input_size=320, num_classes=2)
    x = torch.randn(1, 1, 320, 320)
    y = model(x)
    print(f"Standard 모델:")
    print(f"  입력 크기: {x.shape}")
    print(f"  출력 크기: {y.shape}")
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # Small model
    model_small = QRCNNSmall(input_size=160, num_classes=2)
    x_small = torch.randn(1, 1, 160, 160)
    y_small = model_small(x_small)
    print(f"\nSmall 모델:")
    print(f"  입력 크기: {x_small.shape}")
    print(f"  출력 크기: {y_small.shape}")
    print(f"  파라미터 수: {sum(p.numel() for p in model_small.parameters()):,}")

