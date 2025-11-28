#!/usr/bin/env python3
"""
데이터 파일 크기 및 구조 확인 스크립트
"""

import pickle
import os
import sys
import numpy as np

def check_data_file(file_path):
    """데이터 파일 정보 확인"""
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    # 파일 크기
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"📁 파일: {file_path}")
    print(f"📊 파일 크기: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print()
    
    # 데이터 로드
    print("📂 데이터 로드 중...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data.get('metadata', {})
        demonstrations = data.get('demonstrations', [])
        
        print(f"✅ 데이터 로드 완료")
        print()
        
        # 메타데이터
        print("📋 메타데이터:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        print()
        
        # 에피소드 정보
        print(f"📊 에피소드 정보:")
        print(f"   에피소드 수: {len(demonstrations)}")
        
        total_steps = 0
        total_images = 0
        
        for i, episode in enumerate(demonstrations):
            states = episode.get('states', [])
            actions = episode.get('actions', [])
            
            total_steps += len(states)
            total_images += len(states)
            
            if i < 3:  # 처음 3개만 상세 출력
                print(f"   에피소드 {i+1}: {len(states)} 스텝")
                if len(states) > 0:
                    state = states[0]
                    if isinstance(state, np.ndarray):
                        print(f"      상태 shape: {state.shape}, dtype: {state.dtype}")
        
        print(f"   총 스텝 수: {total_steps}")
        print(f"   총 이미지 수: {total_images}")
        print()
        
        # 이미지 데이터 크기 추정
        if len(demonstrations) > 0 and len(demonstrations[0].get('states', [])) > 0:
            sample_state = demonstrations[0]['states'][0]
            if isinstance(sample_state, np.ndarray):
                state_size = sample_state.nbytes
                estimated_size = state_size * total_images
                estimated_size_mb = estimated_size / (1024 * 1024)
                
                print(f"📸 이미지 데이터:")
                print(f"   이미지 크기: {sample_state.shape}")
                print(f"   이미지당 크기: {state_size} bytes")
                print(f"   예상 총 이미지 데이터: {estimated_size_mb:.2f} MB")
                print()
        
        # 액션 정보
        if len(demonstrations) > 0:
            actions = demonstrations[0].get('actions', [])
            if len(actions) > 0:
                print(f"🎮 액션 정보:")
                print(f"   액션 타입: {type(actions[0])}")
                unique_actions = set()
                for episode in demonstrations:
                    unique_actions.update(episode.get('actions', []))
                print(f"   고유 액션: {sorted(unique_actions)}")
                print()
        
        # 압축 제안
        print(f"💡 최적화 제안:")
        if file_size_mb > 10:
            print(f"   ⚠️  파일이 큽니다 ({file_size_mb:.2f} MB)")
            print(f"   - 이미지를 더 작게 리사이즈 (현재: 16x16)")
            print(f"   - JPEG 압축 사용")
            print(f"   - 샘플 수 줄이기")
        else:
            print(f"   ✅ 파일 크기 적절함")
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python check_data_size.py <데이터_파일.pkl>")
        sys.exit(1)
    
    check_data_file(sys.argv[1])

