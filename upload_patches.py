#!/usr/bin/env python3
"""
패치 단위로 데이터 업로드
16x16 이미지를 패치로 묶어서 효율적으로 전송
"""

import os
import pickle
import numpy as np
import requests
import argparse
import sys
from pathlib import Path


class PatchUploader:
    """패치 단위로 데이터를 업로드하는 클래스"""
    
    def __init__(self, server_url='http://localhost:5000', patch_size=100):
        """
        Args:
            server_url: 서버 URL
            patch_size: 패치 크기 (한 번에 전송할 이미지 수)
        """
        if not server_url.startswith('http://') and not server_url.startswith('https://'):
            server_url = 'http://' + server_url
        self.server_url = server_url.rstrip('/')
        self.patch_size = patch_size
    
    def load_demo_data(self, file_path):
        """데모 데이터 로드"""
        print(f"📂 데이터 로드: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        demonstrations = data.get('demonstrations', [])
        metadata = data.get('metadata', {})
        
        # 모든 (state, action) 쌍 추출
        states = []
        actions = []
        
        for episode in demonstrations:
            ep_states = episode.get('states', [])
            ep_actions = episode.get('actions', [])
            
            min_len = min(len(ep_states), len(ep_actions))
            states.extend(ep_states[:min_len])
            actions.extend(ep_actions[:min_len])
        
        print(f"✅ {len(states)}개 샘플 로드 완료")
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'metadata': metadata
        }
    
    def upload_patches(self, data, file_name='human_demos.pkl'):
        """
        패치 단위로 데이터 업로드
        
        Args:
            data: {'states': [...], 'actions': [...], 'metadata': {...}}
            file_name: 원본 파일명
        """
        states = data['states']
        actions = data['actions']
        metadata = data['metadata']
        
        total_samples = len(states)
        num_patches = (total_samples + self.patch_size - 1) // self.patch_size
        
        print(f"\n📤 패치 업로드 시작")
        print(f"   총 샘플: {total_samples}")
        print(f"   패치 크기: {self.patch_size}")
        print(f"   패치 수: {num_patches}")
        print()
        
        # 세션 ID 생성 (서버에서 패치들을 묶을 때 사용)
        import uuid
        session_id = str(uuid.uuid4())
        
        uploaded_patches = []
        
        for patch_idx in range(num_patches):
            start_idx = patch_idx * self.patch_size
            end_idx = min(start_idx + self.patch_size, total_samples)
            
            patch_states = states[start_idx:end_idx]
            patch_actions = actions[start_idx:end_idx]
            
            # 패치 데이터 준비
            patch_data = {
                'session_id': session_id,
                'patch_index': patch_idx,
                'total_patches': num_patches,
                'states': patch_states.tolist(),  # numpy array를 list로 변환
                'actions': patch_actions.tolist(),
                'metadata': metadata if patch_idx == 0 else None  # 첫 패치에만 메타데이터
            }
            
            print(f"📦 패치 {patch_idx+1}/{num_patches} 업로드 중... ({end_idx-start_idx}개 샘플)", end='', flush=True)
            
            try:
                response = requests.post(
                    f"{self.server_url}/api/upload_patch",
                    json=patch_data,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                uploaded_patches.append(result)
                print(f" ✅")
            except Exception as e:
                print(f" ❌ 실패: {e}")
                return None
        
        # 모든 패치 업로드 완료 후 최종화 요청
        print(f"\n🔗 패치 병합 요청...")
        try:
            merge_response = requests.post(
                f"{self.server_url}/api/merge_patches",
                json={'session_id': session_id},
                timeout=60
            )
            merge_response.raise_for_status()
            result = merge_response.json()
            print(f"✅ 업로드 완료!")
            print(f"   파일 경로: {result.get('file_path')}")
            print(f"   총 샘플: {result.get('total_samples')}")
            return result
        except Exception as e:
            print(f"❌ 패치 병합 실패: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='패치 단위로 데이터 업로드'
    )
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                        help='서버 URL')
    parser.add_argument('--file', type=str, required=True,
                        help='업로드할 pickle 파일 경로')
    parser.add_argument('--patch-size', type=int, default=100,
                        help='패치 크기 (한 번에 전송할 샘플 수, 기본: 100)')
    
    args = parser.parse_args()
    
    # 파일 확인
    if not os.path.exists(args.file):
        print(f"❌ 파일을 찾을 수 없습니다: {args.file}")
        sys.exit(1)
    
    # 업로더 생성
    uploader = PatchUploader(
        server_url=args.server,
        patch_size=args.patch_size
    )
    
    # 데이터 로드
    data = uploader.load_demo_data(args.file)
    
    # 패치 업로드
    result = uploader.upload_patches(data, os.path.basename(args.file))
    
    if result:
        print(f"\n✅ 업로드 성공!")
    else:
        print(f"\n❌ 업로드 실패!")
        sys.exit(1)


if __name__ == '__main__':
    main()

