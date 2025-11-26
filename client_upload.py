#!/usr/bin/env python3
"""
클라이언트: 라즈베리 파이에서 데이터를 수집하여 서버로 전송
"""

import argparse
import requests
import os
import sys
from pathlib import Path


class ServerClient:
    """서버와 통신하는 클라이언트"""
    
    def __init__(self, server_url='http://localhost:5000'):
        """
        Args:
            server_url: 서버 URL (예: http://192.168.1.100:5000)
        """
        self.server_url = server_url.rstrip('/')
    
    def health_check(self):
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.server_url}/api/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 서버 연결 실패: {e}")
            return None
    
    def upload_data(self, file_path):
        """
        데이터 파일 업로드
        
        Args:
            file_path: 업로드할 pickle 파일 경로
        
        Returns:
            업로드 결과 (dict)
        """
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                response = requests.post(
                    f"{self.server_url}/api/upload_data",
                    files=files,
                    timeout=60
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"❌ 업로드 실패: {e}")
            return None
    
    def train_supervised(self, file_path, epochs=100, batch_size=64):
        """
        Supervised Learning 학습 요청
        
        Args:
            file_path: 서버에 업로드된 파일 경로
            epochs: 학습 에폭 수
            batch_size: 배치 크기
        
        Returns:
            학습 결과 (dict)
        """
        try:
            data = {
                'file_path': file_path,
                'epochs': epochs,
                'batch_size': batch_size
            }
            response = requests.post(
                f"{self.server_url}/api/train/supervised",
                json=data,
                timeout=3600  # 1시간 타임아웃
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 학습 요청 실패: {e}")
            return None
    
    def download_model(self, save_path='latest_model.pth'):
        """
        최신 모델 다운로드
        
        Args:
            save_path: 저장할 파일 경로
        
        Returns:
            다운로드 성공 여부
        """
        try:
            response = requests.get(
                f"{self.server_url}/api/model/latest",
                timeout=60
            )
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ 모델 다운로드 완료: {save_path}")
            return True
        except Exception as e:
            print(f"❌ 모델 다운로드 실패: {e}")
            return False
    
    def list_models(self):
        """사용 가능한 모델 목록 조회"""
        try:
            response = requests.get(
                f"{self.server_url}/api/model/list",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 모델 목록 조회 실패: {e}")
            return None
    
    def inference(self, state, model_path=None):
        """
        실시간 추론 (선택 사항)
        
        Args:
            state: 256차원 상태 벡터
            model_path: 사용할 모델 경로 (선택)
        
        Returns:
            추론 결과 (action, log_prob, value)
        """
        try:
            data = {'state': state}
            if model_path:
                data['model_path'] = model_path
            
            response = requests.post(
                f"{self.server_url}/api/inference",
                json=data,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='라즈베리 파이에서 서버로 데이터 전송 및 모델 다운로드'
    )
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                        help='서버 URL (기본: http://localhost:5000)')
    parser.add_argument('--upload', type=str,
                        help='업로드할 데이터 파일 경로')
    parser.add_argument('--train', type=str,
                        help='학습할 데이터 파일 경로 (서버에 업로드된 파일)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에폭 수 (기본: 100)')
    parser.add_argument('--download', type=str,
                        help='모델 다운로드 경로 (예: latest_model.pth)')
    parser.add_argument('--list', action='store_true',
                        help='사용 가능한 모델 목록 조회')
    parser.add_argument('--health', action='store_true',
                        help='서버 상태 확인')
    
    args = parser.parse_args()
    
    client = ServerClient(server_url=args.server)
    
    # 서버 상태 확인
    if args.health:
        result = client.health_check()
        if result:
            print(f"✅ 서버 상태: {result}")
        sys.exit(0)
    
    # 데이터 업로드
    if args.upload:
        print(f"📤 데이터 업로드 중: {args.upload}")
        result = client.upload_data(args.upload)
        if result:
            print(f"✅ 업로드 성공:")
            print(f"   파일: {result.get('filename')}")
            print(f"   에피소드: {result.get('num_episodes')}")
            print(f"   스텝: {result.get('total_steps')}")
            print(f"   파일 경로: {result.get('file_path')}")
    
    # 학습 요청
    if args.train:
        print(f"🎓 학습 시작: {args.train}")
        result = client.train_supervised(args.train, epochs=args.epochs)
        if result:
            print(f"✅ 학습 완료:")
            print(f"   모델 경로: {result.get('model_path')}")
    
    # 모델 다운로드
    if args.download:
        print(f"📥 모델 다운로드 중...")
        client.download_model(args.download)
    
    # 모델 목록 조회
    if args.list:
        result = client.list_models()
        if result:
            models = result.get('models', [])
            print(f"📋 사용 가능한 모델 ({len(models)}개):")
            for model in models:
                print(f"   - {model['filename']} ({model['size']} bytes, {model['modified']})")


if __name__ == '__main__':
    main()

