#!/usr/bin/env python3
"""
여러 데모 데이터 파일을 하나로 합치는 스크립트
"""

import pickle
import os
import argparse
from datetime import datetime
import glob


def merge_demo_files(input_files, output_file, verbose=True):
    """
    여러 데모 데이터 파일을 하나로 합치기
    
    Args:
        input_files: 입력 파일 경로 리스트
        output_file: 출력 파일 경로
        verbose: 상세 출력 여부
    """
    if verbose:
        print(f"\n{'='*60}")
        print("데모 데이터 합치기")
        print(f"{'='*60}\n")
    
    all_demonstrations = []
    total_episodes = 0
    total_steps = 0
    all_metadata = []
    filtered_count = 0
    
    # 각 파일 로드
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️  파일을 찾을 수 없습니다: {file_path}")
            continue
        
        if verbose:
            print(f"📂 로드 중: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data.get('metadata', {})
            demonstrations = data.get('demonstrations', [])
            
            if verbose:
                file_episodes = len(demonstrations)
                file_steps = sum(len(ep.get('states', [])) for ep in demonstrations)
                print(f"   ✅ {file_episodes}개 에피소드, {file_steps}개 스텝")
            
            # 각 에피소드 처리
            for episode in demonstrations:
                states = episode.get('states', [])
                actions = episode.get('actions', [])
                
                # 유효성 검사
                if not states or not actions or len(states) == 0 or len(actions) == 0:
                    filtered_count += 1
                    continue
                
                # 길이 맞추기
                if len(states) != len(actions):
                    min_len = min(len(states), len(actions))
                    states = states[:min_len]
                    actions = actions[:min_len]
                
                # 유효한 에피소드만 추가
                if len(states) > 0 and len(actions) > 0:
                    # rewards, dones, timestamps도 포함 (있는 경우)
                    merged_episode = {
                        'states': states,
                        'actions': actions
                    }
                    
                    # 선택적 필드 추가
                    if 'rewards' in episode:
                        rewards = episode['rewards']
                        if len(rewards) == len(states):
                            merged_episode['rewards'] = rewards
                    
                    if 'dones' in episode:
                        dones = episode['dones']
                        if len(dones) == len(states):
                            merged_episode['dones'] = dones
                    
                    if 'timestamps' in episode:
                        timestamps = episode['timestamps']
                        if len(timestamps) == len(states):
                            merged_episode['timestamps'] = timestamps
                    
                    all_demonstrations.append(merged_episode)
                    total_steps += len(states)
                    total_episodes += 1
            
            # 메타데이터 저장 (나중에 통합)
            all_metadata.append({
                'source_file': os.path.basename(file_path),
                'metadata': metadata
            })
            
        except Exception as e:
            print(f"❌ 파일 로드 실패: {file_path}")
            print(f"   에러: {e}")
            continue
    
    if len(all_demonstrations) == 0:
        print("❌ 합칠 수 있는 유효한 에피소드가 없습니다.")
        return False
    
    if verbose:
        print(f"\n{'='*60}")
        print("통합 결과")
        print(f"{'='*60}")
        print(f"✅ 총 {total_episodes}개 에피소드")
        print(f"✅ 총 {total_steps:,}개 스텝")
        if filtered_count > 0:
            print(f"⚠️  {filtered_count}개 에피소드 필터링됨")
        print()
    
    # 통합 메타데이터 생성
    # 첫 번째 파일의 메타데이터를 기본으로 사용
    base_metadata = {}
    if all_metadata:
        base_metadata = all_metadata[0]['metadata'].copy()
    
    # 통합된 메타데이터
    merged_metadata = {
        'env_type': base_metadata.get('env_type', 'real'),
        'use_discrete_actions': base_metadata.get('use_discrete_actions', True),
        'use_extended_actions': base_metadata.get('use_extended_actions', False),
        'num_episodes': total_episodes,
        'total_steps': total_steps,
        'timestamp': datetime.now().isoformat(),
        'source_files': [m['source_file'] for m in all_metadata],
        'merged_at': datetime.now().isoformat()
    }
    
    # 통합 데이터 구조
    merged_data = {
        'metadata': merged_metadata,
        'demonstrations': all_demonstrations
    }
    
    # 파일 저장
    try:
        # 출력 디렉토리가 없으면 생성
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(merged_data, f)
        
        if verbose:
            print(f"✅ 통합 파일 저장: {output_file}")
            print(f"   파일 크기: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='여러 데모 데이터 파일을 하나로 합치기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 여러 파일 명시
  python merge_demo_data.py -i file1.pkl file2.pkl file3.pkl -o merged.pkl
  
  # 패턴 사용
  python merge_demo_data.py -p "uploaded_data/demos_*.pkl" -o merged.pkl
  
  # 특정 디렉토리의 모든 .pkl 파일
  python merge_demo_data.py -d uploaded_data -o merged.pkl
        """
    )
    
    # 입력 파일 옵션 (여러 방법 지원)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', nargs='+', 
                            help='입력 파일 경로 리스트')
    input_group.add_argument('-p', '--pattern',
                            help='파일 패턴 (glob, 예: "uploaded_data/*.pkl")')
    input_group.add_argument('-d', '--directory',
                            help='디렉토리 내 모든 .pkl 파일 사용')
    
    parser.add_argument('-o', '--output', required=True,
                       help='출력 파일 경로')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                       help='상세 출력 (기본: True)')
    
    args = parser.parse_args()
    
    # 입력 파일 목록 생성
    input_files = []
    
    if args.input:
        input_files = args.input
    elif args.pattern:
        input_files = sorted(glob.glob(args.pattern))
        if not input_files:
            print(f"❌ 패턴과 일치하는 파일이 없습니다: {args.pattern}")
            return
    elif args.directory:
        if not os.path.isdir(args.directory):
            print(f"❌ 디렉토리가 존재하지 않습니다: {args.directory}")
            return
        input_files = sorted(glob.glob(os.path.join(args.directory, '*.pkl')))
        if not input_files:
            print(f"❌ 디렉토리에 .pkl 파일이 없습니다: {args.directory}")
            return
    
    if len(input_files) == 0:
        print("❌ 입력 파일이 없습니다.")
        return
    
    if args.verbose:
        print(f"📋 입력 파일: {len(input_files)}개")
        for f in input_files:
            print(f"   - {f}")
        print()
    
    # 합치기 실행
    success = merge_demo_files(input_files, args.output, args.verbose)
    
    if success:
        print(f"\n{'='*60}")
        print("✅ 데이터 합치기 완료!")
        print(f"{'='*60}\n")
    else:
        print(f"\n❌ 데이터 합치기 실패\n")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

