#!/usr/bin/env python3
"""
AI 에이전트 실행 스크립트
학습된 PPO 모델을 로드하여 RC Car를 0.1초 간격으로 제어

사용법:
    python run_ai_agent.py --model ppo_model.pth --port /dev/ttyACM0 --delay 0.1
"""

import os
# NumPy/PyTorch 임포트 전에 환경 변수 설정 (Bus error 방지)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# PyTorch 스레드 제한
os.environ['TORCH_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import numpy as np

# PyTorch 임포트 (안전하게)
print("PyTorch 임포트 중...", flush=True)
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} 임포트 성공", flush=True)
except Exception as e:
    print(f"❌ PyTorch 임포트 실패: {e}", flush=True)
    print("라즈베리 파이용 PyTorch를 설치하세요:", flush=True)
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu", flush=True)
    import sys
    sys.exit(1)
import time
import sys
import os
from datetime import datetime

# 환경 및 에이전트 임포트
from rc_car_sim_env import RCCarSimEnv
from car_racing_env import CarRacingEnvWrapper
from ppo_agent import PPOAgent
from rc_car_controller import RCCarController

# 실제 하드웨어 환경은 선택적 임포트
try:
    from rc_car_env import RCCarEnv
    HAS_REAL_ENV = True
except ImportError:
    HAS_REAL_ENV = False
    RCCarEnv = None


class AIAgentRunner:
    """
    AI 에이전트 실행 클래스
    0.1초 간격으로 액션을 실행하며 RC Car를 제어
    """
    
    def __init__(
        self,
        model_path: str,
        env_type: str = 'carracing',
        port: str = '/dev/ttyACM0',
        action_delay: float = 0.1,
        max_steps: int = 1000,
        use_discrete_actions: bool = True,  # 이산 액션만 사용
        use_extended_actions: bool = True,
        device: str = None
    ):
        """
        Args:
            model_path: 학습된 모델 경로
            env_type: 환경 타입 ('carracing', 'sim', 'real')
            port: 시리얼 포트 (실제 하드웨어 사용 시)
            action_delay: 액션 간 지연 시간 (초, 기본: 0.1)
            max_steps: 최대 스텝 수
            use_discrete_actions: 이산 액션 사용 여부
            use_extended_actions: 확장된 액션 공간 사용 여부
            device: 디바이스 (cuda/cpu)
        """
        self.model_path = model_path
        self.env_type = env_type
        self.port = port
        self.action_delay = action_delay
        self.max_steps = max_steps
        self.use_discrete_actions = use_discrete_actions
        self.use_extended_actions = use_extended_actions
        
        # 디바이스 설정 (라즈베리 파이에서는 항상 CPU)
        if device is None:
            # 라즈베리 파이에서는 GPU가 없으므로 항상 CPU 사용
            self.device = 'cpu'
        else:
            self.device = device
        
        print(f"🔧 디바이스: {self.device}")
        print(f"액션 지연 시간: {action_delay:.3f}초")
        print(f"환경 타입: {env_type}")
        
        # 단계별 초기화 (Bus error 방지)
        print("\n[초기화 단계 1/4] 환경 생성 중...")
        try:
            self.env = self._create_env()
            print("✅ 환경 생성 완료")
        except Exception as e:
            print(f"❌ 환경 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 하드웨어 제어기 생성 (real 모드일 때만)
        print("\n[초기화 단계 2/4] 하드웨어 제어기 연결 중...")
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
        
        # 에이전트 생성 및 모델 로드
        print("\n[초기화 단계 3/4] 에이전트 생성 중...")
        try:
            self.agent = self._load_agent()
            print("✅ 에이전트 생성 완료")
        except Exception as e:
            print(f"❌ 에이전트 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("\n[초기화 단계 4/4] 초기화 완료!")
        print("=" * 60)
    
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
    
    def _load_agent(self):
        """에이전트 생성 및 모델 로드"""
        # 에이전트 생성
        agent = PPOAgent(
            state_dim=784,  # 28x28 이미지 = 784 차원 (환경 출력과 일치)
            action_dim=5,  # 이산 액션만 (고정)
            latent_dim=256,
            hidden_dim=256,
            n_cycles=4,
            carry_latent=True,
            device=self.device,
            discrete_action=True,  # 이산 액션만
            num_discrete_actions=5,
            use_recurrent=True
        )
        
        # 모델 로드 (안전한 방식)
        if os.path.exists(self.model_path):
            try:
                print(f"📥 모델 로드 중: {self.model_path}")
                agent.load(self.model_path)
                print(f"✅ 모델 로드 완료: {self.model_path}")
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {e}")
                print("랜덤 정책으로 실행합니다.")
        else:
            print(f"⚠️  모델 파일을 찾을 수 없습니다: {self.model_path}")
            print("랜덤 정책으로 실행합니다.")
        
        return agent
    
    def run_episode(self, render: bool = False, verbose: bool = True):
        """
        단일 에피소드 실행
        
        Args:
            render: 렌더링 여부
            verbose: 상세 출력 여부
        
        Returns:
            episode_reward: 에피소드 총 리워드
            episode_length: 에피소드 길이
        """
        # 환경 리셋
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, _ = reset_result  # Gymnasium
        else:
            state = reset_result  # Gym
        
        # TRM-PPO: 잠재 상태 초기화
        if hasattr(self.agent, 'use_recurrent') and self.agent.use_recurrent:
            self.agent.reset_carry()
        
        episode_reward = 0.0
        episode_length = 0
        
        if verbose:
            print("\n" + "=" * 60)
            print("AI 에이전트 실행 시작")
            print("=" * 60)
            print(f"액션 간격: {self.action_delay:.3f}초")
            print("=" * 60 + "\n")
        
        try:
            for step in range(self.max_steps):
                # 상태 정규화 [0, 255] -> [0, 1]
                state_normalized = state.astype(np.float32) / 255.0
                state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(self.device)
                
                # 액션 선택 (deterministic: 최대 확률 액션)
                if hasattr(self.agent, 'use_recurrent') and self.agent.use_recurrent:
                    action, _, value, _ = self.agent.get_action_with_carry(
                        state_tensor, deterministic=True
                    )
                else:
                    action, _, value = self.agent.actor_critic.get_action(
                        state_tensor, deterministic=True
                    )
                
                # 액션 변환
                if self.use_discrete_actions:
                    if isinstance(action, torch.Tensor):
                        action_np = action.squeeze(0).cpu().detach().numpy()
                        if action_np.ndim == 0:
                            action_np = int(action_np)
                        else:
                            action_np = int(action_np[0]) if len(action_np) > 0 else int(action_np)
                    else:
                        action_np = int(action)
                else:
                    if isinstance(action, torch.Tensor):
                        action_np = action.squeeze(0).cpu().detach().numpy()
                    else:
                        action_np = np.array(action)
                
                # 실제 하드웨어 제어 (real 모드일 때)
                if self.controller is not None and self.use_discrete_actions:
                    self.controller.execute_discrete_action(action_np)
                
                # 환경 스텝 실행
                next_state, reward, done, info = self.env.step(action_np)
                
                episode_reward += reward
                episode_length += 1
                
                # 출력
                if verbose:
                    action_name = {
                        0: "Stop", 1: "Right+Gas", 2: "Left+Gas", 
                        3: "Gas", 4: "Brake"
                    }.get(action_np, f"Action {action_np}") if self.use_discrete_actions else f"Action {action_np}"
                    
                    print(f"[Step {step+1:4d}] "
                          f"Action: {action_name:12s} | "
                          f"Reward: {reward:7.3f} | "
                          f"Total: {episode_reward:7.3f} | "
                          f"Value: {value.item():7.3f}")
                
                # 렌더링
                if render and hasattr(self.env, 'render'):
                    self.env.render()
                
                # 0.1초 지연 (액션 간격)
                time.sleep(self.action_delay)
                
                # 에피소드 종료
                if done:
                    break
                
                state = next_state
        
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        
        if verbose:
            print("\n" + "=" * 60)
            print("에피소드 완료")
            print("=" * 60)
            print(f"총 리워드: {episode_reward:.3f}")
            print(f"에피소드 길이: {episode_length} 스텝")
            print(f"평균 리워드: {episode_reward/episode_length:.3f}" if episode_length > 0 else "평균 리워드: 0.000")
            print("=" * 60 + "\n")
        
        # 정지 (실제 하드웨어)
        if self.controller is not None:
            self.controller.stop()
        
        return episode_reward, episode_length
    
    def run_multiple_episodes(self, num_episodes: int = 5, render: bool = False, verbose: bool = True):
        """
        여러 에피소드 실행
        
        Args:
            num_episodes: 에피소드 수
            render: 렌더링 여부
            verbose: 상세 출력 여부
        
        Returns:
            episode_rewards: 에피소드 리워드 리스트
            episode_lengths: 에피소드 길이 리스트
        """
        episode_rewards = []
        episode_lengths = []
        
        print(f"\n{'='*60}")
        print(f"총 {num_episodes}개 에피소드 실행")
        print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            if verbose:
                print(f"\n>>> 에피소드 {episode + 1}/{num_episodes} <<<")
            
            reward, length = self.run_episode(render=render, verbose=verbose)
            episode_rewards.append(reward)
            episode_lengths.append(length)
            
            # 에피소드 간 짧은 대기
            if episode < num_episodes - 1:
                time.sleep(1.0)
        
        # 통계 출력
        print(f"\n{'='*60}")
        print("전체 통계")
        print(f"{'='*60}")
        print(f"평균 리워드: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"최고 리워드: {np.max(episode_rewards):.3f}")
        print(f"최저 리워드: {np.min(episode_rewards):.3f}")
        print(f"평균 길이: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} 스텝")
        print(f"{'='*60}\n")
        
        return episode_rewards, episode_lengths
    
    def close(self):
        """리소스 정리"""
        if self.env:
            self.env.close()
        if self.controller:
            self.controller.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='AI 에이전트 실행 - 학습된 모델로 RC Car 제어',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # CarRacing 환경에서 실행 (0.1초 간격)
  python run_ai_agent.py --model ppo_model.pth --env-type carracing --delay 0.1
  
  # 실제 하드웨어에서 실행 (0.1초 간격)
  python run_ai_agent.py --model ppo_model.pth --env-type real --port /dev/ttyACM0 --delay 0.1
  
  # 여러 에피소드 실행
  python run_ai_agent.py --model ppo_model.pth --episodes 5 --delay 0.1
        """
    )
    
    # 모델 경로
    parser.add_argument('--model', type=str, required=True,
                        help='학습된 모델 경로 (예: ppo_model.pth)')
    
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
    
    # 실행 설정
    parser.add_argument('--episodes', type=int, default=1,
                        help='실행할 에피소드 수 (기본: 1)')
    parser.add_argument('--render', action='store_true',
                        help='렌더링 활성화 (시뮬레이션/CarRacing 모드)')
    parser.add_argument('--quiet', action='store_true',
                        help='상세 출력 비활성화')
    
    # 디바이스
    parser.add_argument('--device', type=str, default=None,
                        help='디바이스 (cuda/cpu, 기본: 자동 선택)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("AIAgentRunner 생성 시작")
    print("=" * 60)
    
    # AI 에이전트 실행기 생성 (단계별)
    try:
        print("\n[단계 1] AIAgentRunner 인스턴스 생성 중...")
        runner = AIAgentRunner(
            model_path=args.model,
            env_type=args.env_type,
            port=args.port,
            action_delay=args.delay,
            max_steps=args.max_steps,
            use_discrete_actions=args.use_discrete_actions,
            use_extended_actions=args.use_extended_actions,
            device=args.device
        )
        print("✅ AIAgentRunner 생성 완료")
    except Exception as e:
        print(f"\n❌ AIAgentRunner 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # 에피소드 실행
        if args.episodes == 1:
            runner.run_episode(render=args.render, verbose=not args.quiet)
        else:
            runner.run_multiple_episodes(
                num_episodes=args.episodes,
                render=args.render,
                verbose=not args.quiet
            )
    finally:
        runner.close()


if __name__ == "__main__":
    main()

