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
            server_url: 서버 URL (예: http://192.168.1.100:5000 또는 192.168.1.100:5000)
        """
        # http:// 프로토콜이 없으면 자동 추가
        if not server_url.startswith('http://') and not server_url.startswith('https://'):
            server_url = 'http://' + server_url
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
    
    def upload_data(self, file_path, chunk_size_mb=10):
        """
        데이터 파일 업로드 (대용량 파일 지원)
        
        Args:
            file_path: 업로드할 pickle 파일 경로
            chunk_size_mb: 청크 크기 (MB, 기본: 10MB)
        
        Returns:
            업로드 결과 (dict)
        """
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return None
        
        # 파일 크기 확인
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"📊 파일 크기: {file_size_mb:.2f} MB")
        
        # 파일이 크면 경고
        if file_size_mb > 50:
            print(f"⚠️  파일이 큽니다 ({file_size_mb:.2f} MB). 업로드에 시간이 걸릴 수 있습니다.")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                
                # 타임아웃 계산 (파일 크기에 따라)
                timeout = max(60, int(file_size_mb * 2))  # 최소 60초, MB당 2초
                print(f"⏱️  타임아웃: {timeout}초")
                
                response = requests.post(
                    f"{self.server_url}/api/upload_data",
                    files=files,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.Timeout:
            print(f"❌ 업로드 타임아웃 (파일이 너무 큽니다)")
            print(f"💡 해결 방법:")
            print(f"   1. 파일을 압축하거나")
            print(f"   2. 서버의 타임아웃 설정을 늘리세요")
            return None
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
    
    def train_imitation_rl(self, file_path, model_path=None, epochs=100, batch_size=64, learning_rate=3e-4):
        """
        Imitation Learning via Reinforcement Learning 학습 요청
        
        Args:
            file_path: 서버에 업로드된 데모 데이터 파일 경로
            model_path: 사전 학습된 모델 경로 (선택)
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            learning_rate: 학습률
        
        Returns:
            학습 결과 (dict)
        """
        try:
            data = {
                'file_path': file_path,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
            if model_path:
                data['model_path'] = model_path
            
            response = requests.post(
                f"{self.server_url}/api/train/imitation_rl",
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
    parser.add_argument('--train-imitation', type=str,
                        help='Imitation RL 학습할 데이터 파일 경로')
    parser.add_argument('--pretrain-model', type=str,
                        help='사전 학습된 모델 경로 (Imitation RL용)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에폭 수 (기본: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='배치 크기 (기본: 64)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='학습률 (기본: 3e-4)')
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
    
    # Supervised Learning 학습 요청
    if args.train:
        print(f"🎓 Supervised Learning 시작: {args.train}")
        result = client.train_supervised(
            args.train,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        if result:
            print(f"✅ 학습 완료:")
            print(f"   모델 경로: {result.get('model_path')}")
    
    # Imitation RL 학습 요청
    if args.train_imitation:
        print(f"🎓 Imitation RL 학습 시작: {args.train_imitation}")
        result = client.train_imitation_rl(
            args.train_imitation,
            model_path=args.pretrain_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        if result:
            print(f"✅ 학습 완료:")
            print(f"   모델 경로: {result.get('model_path')}")
            print(f"   최종 일치율: {result.get('final_match_rate', 0):.2%}")
    
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

