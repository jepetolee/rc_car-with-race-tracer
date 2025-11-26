#!/usr/bin/env python3
"""
사람이 직접 조작한 데이터 수집 스크립트
Supervised Learning (Teacher Forcing)을 위한 데모 데이터 수집

사용법:
    python collect_human_demonstrations.py --port /dev/ttyACM0 --output demos.pkl
"""

import argparse
import numpy as np
import pickle
import time
import sys
import os
from datetime import datetime
from collections import deque

# 환경 및 제어기 임포트
from rc_car_sim_env import RCCarSimEnv
from car_racing_env import CarRacingEnvWrapper
from rc_car_controller import RCCarController

# 실제 하드웨어 환경은 선택적 임포트
try:
    from rc_car_env import RCCarEnv
    HAS_REAL_ENV = True
except ImportError:
    HAS_REAL_ENV = False
    RCCarEnv = None


class HumanDemonstrationCollector:
    """
    사람이 직접 조작한 데이터 수집 클래스
    """
    
    def __init__(
        self,
        env_type: str = 'carracing',
        port: str = '/dev/ttyACM0',
        use_discrete_actions: bool = True,  # 이산 액션만 사용
        use_extended_actions: bool = True,
        max_steps: int = 1000,
        action_delay: float = 0.1
    ):
        """
        Args:
            env_type: 환경 타입 ('carracing', 'sim', 'real')
            port: 시리얼 포트 (real 모드 사용 시)
            use_discrete_actions: 이산 액션 사용 여부
            use_extended_actions: 확장된 액션 공간 사용 여부
            max_steps: 최대 스텝 수
            action_delay: 액션 간 지연 시간
        """
        self.env_type = env_type
        self.port = port
        self.use_discrete_actions = use_discrete_actions
        self.use_extended_actions = use_extended_actions
        self.max_steps = max_steps
        self.action_delay = action_delay
        
        # 환경 생성
        self.env = self._create_env()
        
        # 실제 하드웨어 제어기 (real 모드일 때만)
        self.controller = None
        if env_type == 'real':
            try:
                self.controller = RCCarController(port=port, delay=action_delay)
                print(f"✅ 실제 하드웨어 연결: {port}")
            except Exception as e:
                print(f"⚠️  실제 하드웨어 연결 실패: {e}")
                print("시뮬레이션 모드로 전환합니다.")
                self.env_type = 'sim'
                self.env = self._create_env()
        
        # 데이터 저장소
        self.demonstrations = []
        self.current_episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'timestamps': []
        }
    
    def _create_env(self):
        """환경 생성"""
        if self.env_type == 'carracing':
            try:
                env = CarRacingEnvWrapper(
                    max_steps=self.max_steps,
                    use_extended_actions=self.use_extended_actions,
                    use_discrete_actions=self.use_discrete_actions
                )
                print("✅ CarRacing 환경 사용")
                return env
            except ImportError as e:
                print(f"❌ CarRacing 환경을 사용할 수 없습니다: {e}")
                print("시뮬레이션 환경으로 전환합니다.")
                self.env_type = 'sim'
                return self._create_env()
        
        elif self.env_type == 'sim':
            env = RCCarSimEnv(
                max_steps=self.max_steps,
                use_extended_actions=self.use_extended_actions,
                use_discrete_actions=self.use_discrete_actions
            )
            print("✅ 시뮬레이션 환경 사용")
            return env
        
        elif self.env_type == 'real':
            if not HAS_REAL_ENV:
                raise ImportError(
                    "실제 하드웨어 환경을 사용할 수 없습니다.\n"
                    "시뮬레이션 환경을 사용하세요: --env-type sim"
                )
            env = RCCarEnv(
                max_steps=self.max_steps,
                use_extended_actions=self.use_extended_actions,
                use_discrete_actions=self.use_discrete_actions
            )
            print("✅ 실제 하드웨어 환경 사용")
            return env
        
        else:
            raise ValueError(f"알 수 없는 환경 타입: {self.env_type}")
    
    def collect_episode(self, episode_num: int = 1):
        """
        단일 에피소드 데이터 수집
        
        Args:
            episode_num: 에피소드 번호
        
        Returns:
            episode_data: 수집된 에피소드 데이터
        """
        print(f"\n{'='*60}")
        print(f"에피소드 {episode_num} 데이터 수집 시작")
        print(f"{'='*60}")
        print("키보드 조작:")
        print("  w: 전진 (Action 3)")
        print("  a: 좌회전+가스 (Action 2)")
        print("  d: 우회전+가스 (Action 1)")
        print("  s: 정지 (Action 0)")
        print("  x: 브레이크 (Action 4)")
        print("  q: 에피소드 종료")
        print(f"{'='*60}")
        print("💡 중요: 키를 누를 때만 이미지가 캡처됩니다!")
        print("   키를 누르지 않으면 데이터가 수집되지 않습니다.")
        print(f"{'='*60}\n")
        
        # 환경 리셋
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, _ = reset_result  # Gymnasium
        else:
            state = reset_result  # Gym
        
        # 에피소드 데이터 초기화
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'timestamps': []
        }
        
        episode_reward = 0.0
        episode_length = 0
        
        try:
            # 키보드 입력을 위한 설정 (비동기 입력)
            import select
            import tty
            import termios
            
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            print("조작을 시작하세요... (q로 종료)")
            print("💡 팁: 키를 누를 때만 이미지가 캡처됩니다!")
            
            step = 0
            last_action = 0  # 마지막 액션 저장
            
            while step < self.max_steps:
                # 키보드 입력 확인 (논블로킹)
                action = None
                key_pressed = False
                
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    key_pressed = True
                    
                    if key == 'q':
                        print("\n에피소드 종료 요청")
                        break
                    elif key == 'w':
                        action = 3  # 전진
                        print(f"[데이터 {len(episode_data['states'])+1}] Action: Gas (Forward)")
                    elif key == 'a':
                        action = 2  # 좌회전+가스
                        print(f"[데이터 {len(episode_data['states'])+1}] Action: Left + Gas")
                    elif key == 'd':
                        action = 1  # 우회전+가스
                        print(f"[데이터 {len(episode_data['states'])+1}] Action: Right + Gas")
                    elif key == 's':
                        action = 0  # 정지
                        print(f"[데이터 {len(episode_data['states'])+1}] Action: Stop")
                    elif key == 'x':
                        action = 4  # 브레이크
                        print(f"[데이터 {len(episode_data['states'])+1}] Action: Brake")
                
                # 키를 눌렀을 때만 이미지 캡처 및 데이터 저장
                if key_pressed and action is not None:
                    # 현재 상태 저장 (키를 누른 순간의 이미지)
                    state_normalized = state.astype(np.float32) / 255.0
                    episode_data['states'].append(state_normalized.copy())
                    episode_data['timestamps'].append(time.time())
                    
                    # 실제 하드웨어 제어
                    if self.controller is not None:
                        self.controller.execute_discrete_action(action)
                    
                    # 환경 스텝
                    next_state, reward, done, info = self.env.step(action)
                    
                    # 데이터 저장
                    episode_data['actions'].append(action)
                    episode_data['rewards'].append(reward)
                    episode_data['dones'].append(done)
                    
                    episode_reward += reward
                    episode_length += 1
                    step += 1
                    last_action = action
                    
                    # 다음 상태로 업데이트
                    state = next_state
                    
                    # 액션 지연
                    time.sleep(self.action_delay)
                    
                    if done:
                        break
                else:
                    # 키 입력이 없으면 짧은 대기 (CPU 사용량 감소)
                    time.sleep(0.05)
            
            # 터미널 설정 복원
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자에 의해 중단되었습니다.")
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        # 정지 (실제 하드웨어)
        if self.controller is not None:
            self.controller.stop()
        
        print(f"\n에피소드 완료:")
        print(f"  길이: {episode_length} 스텝")
        print(f"  총 리워드: {episode_reward:.3f}")
        print(f"  평균 리워드: {episode_reward/episode_length:.3f}" if episode_length > 0 else "  평균 리워드: 0.000")
        
        return episode_data
    
    def collect_multiple_episodes(self, num_episodes: int = 5):
        """
        여러 에피소드 데이터 수집
        
        Args:
            num_episodes: 수집할 에피소드 수
        
        Returns:
            demonstrations: 수집된 모든 에피소드 데이터
        """
        demonstrations = []
        
        print(f"\n{'='*60}")
        print(f"총 {num_episodes}개 에피소드 데이터 수집")
        print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            episode_data = self.collect_episode(episode + 1)
            demonstrations.append(episode_data)
            
            # 에피소드 간 대기
            if episode < num_episodes - 1:
                print(f"\n다음 에피소드를 준비하세요... (3초 후 시작)")
                time.sleep(3)
        
        return demonstrations
    
    def save_demonstrations(self, demonstrations, filepath: str):
        """
        수집된 데이터를 파일로 저장
        
        Args:
            demonstrations: 수집된 에피소드 데이터 리스트
            filepath: 저장할 파일 경로
        """
        # 메타데이터 추가
        metadata = {
            'env_type': self.env_type,
            'use_discrete_actions': self.use_discrete_actions,
            'use_extended_actions': self.use_extended_actions,
            'max_steps': self.max_steps,
            'action_delay': self.action_delay,
            'num_episodes': len(demonstrations),
            'total_steps': sum(len(ep['states']) for ep in demonstrations),
            'timestamp': datetime.now().isoformat()
        }
        
        data = {
            'metadata': metadata,
            'demonstrations': demonstrations
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✅ 데이터 저장 완료: {filepath}")
        print(f"   에피소드 수: {len(demonstrations)}")
        print(f"   총 스텝 수: {metadata['total_steps']}")
    
    def close(self):
        """리소스 정리"""
        if self.env:
            self.env.close()
        if self.controller:
            self.controller.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='사람이 직접 조작한 데이터 수집 (Supervised Learning용)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # CarRacing 환경에서 데이터 수집
  python collect_human_demonstrations.py --env-type carracing --output demos.pkl
  
  # 실제 하드웨어에서 데이터 수집
  python collect_human_demonstrations.py --env-type real --port /dev/ttyACM0 --output demos.pkl
  
  # 여러 에피소드 수집
  python collect_human_demonstrations.py --episodes 10 --output demos.pkl
        """
    )
    
    # 환경 설정
    parser.add_argument('--env-type', choices=['carracing', 'sim', 'real'],
                        default='carracing',
                        help='환경 타입 (기본: carracing)')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                        help='시리얼 포트 (real 모드 사용 시, 기본: /dev/ttyACM0)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='최대 스텝 수 (기본: 1000)')
    
    # 액션 설정
    parser.add_argument('--delay', type=float, default=0.1,
                        help='액션 간 지연 시간 (초, 기본: 0.1)')
    parser.add_argument('--use-discrete-actions', action='store_true', default=True,
                        help='이산 액션 사용 (기본: True)')
    parser.add_argument('--use-continuous-actions', dest='use_discrete_actions', action='store_false',
                        help='연속 액션 사용')
    parser.add_argument('--use-extended-actions', action='store_true', default=True,
                        help='확장된 액션 공간 사용 (기본: True)')
    
    # 수집 설정
    parser.add_argument('--episodes', type=int, default=1,
                        help='수집할 에피소드 수 (기본: 1)')
    parser.add_argument('--output', type=str, default='human_demos.pkl',
                        help='저장할 파일 경로 (기본: human_demos.pkl)')
    
    args = parser.parse_args()
    
    # 데이터 수집기 생성
    collector = HumanDemonstrationCollector(
        env_type=args.env_type,
        port=args.port,
        use_discrete_actions=args.use_discrete_actions,
        use_extended_actions=args.use_extended_actions,
        max_steps=args.max_steps,
        action_delay=args.delay
    )
    
    try:
        # 데이터 수집
        if args.episodes == 1:
            demonstrations = [collector.collect_episode(1)]
        else:
            demonstrations = collector.collect_multiple_episodes(args.episodes)
        
        # 데이터 저장
        collector.save_demonstrations(demonstrations, args.output)
        
    finally:
        collector.close()


if __name__ == "__main__":
    main()

