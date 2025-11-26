#!/usr/bin/env python3
"""
Supervised Learning (Teacher Forcing)을 사용한 사전 학습
사람이 직접 조작한 데이터로 모델을 supervised learning으로 사전 학습한 후 강화학습으로 fine-tuning

Teacher Forcing = Supervised Learning:
- 사람이 조작한 (상태, 액션) 쌍을 사용
- Maximum Likelihood Estimation (MLE)으로 정책 학습
- 실제 액션의 로그 확률을 최대화하는 방식

사용법:
    # 1단계: Supervised Learning 사전 학습
    python train_with_teacher_forcing.py --demos human_demos.pkl --pretrain-epochs 100
    
    # 2단계: 강화학습으로 fine-tuning
    python train_with_teacher_forcing.py --demos human_demos.pkl --pretrain-epochs 100 --rl-steps 100000
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import sys
from datetime import datetime
from collections import deque

# 환경 및 에이전트 임포트
from rc_car_sim_env import RCCarSimEnv
from car_racing_env import CarRacingEnvWrapper
from ppo_agent import PPOAgent
from train_ppo import train_ppo

# TensorBoard 지원
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️  TensorBoard 미설치 - pip install tensorboard 로 설치하면 실시간 모니터링 가능")


class TeacherForcingTrainer:
    """
    Supervised Learning (Teacher Forcing)을 사용한 사전 학습 클래스
    사람이 직접 조작한 (상태, 액션) 쌍으로 정책을 supervised learning으로 학습
    
    학습 방식:
    - Maximum Likelihood Estimation (MLE)
    - Loss = -log P(실제_액션 | 상태)
    - 실제 액션의 로그 확률을 최대화
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        demonstrations: list,
        device: str = 'cuda',
        lr: float = 3e-4
    ):
        """
        Args:
            agent: PPO 에이전트
            demonstrations: 수집된 데모 데이터 리스트
            device: 디바이스
            lr: 학습률
        """
        self.agent = agent
        self.device = device
        self.demonstrations = demonstrations
        
        # 옵티마이저 (Actor만 학습)
        self.optimizer = optim.Adam(
            self.agent.actor_critic.parameters(),
            lr=lr
        )
        
        # 데이터 준비
        self.states, self.actions = self._prepare_data()
        
        print(f"✅ Teacher Forcing 데이터 준비 완료")
        print(f"   총 상태 수: {len(self.states)}")
        print(f"   총 액션 수: {len(self.actions)}")
    
    def _prepare_data(self):
        """데모 데이터를 학습용으로 변환"""
        all_states = []
        all_actions = []
        
        for episode in self.demonstrations:
            states = episode['states']
            actions = episode['actions']
            
            # 상태와 액션을 텐서로 변환
            for state, action in zip(states, actions):
                all_states.append(state)
                all_actions.append(action)
        
        return np.array(all_states), np.array(all_actions)
    
    def train_epoch(self, batch_size: int = 64):
        """
        단일 에폭 학습
        
        Args:
            batch_size: 배치 크기
        
        Returns:
            loss: 평균 손실
        """
        total_loss = 0.0
        num_batches = 0
        
        # 데이터 셔플
        indices = np.random.permutation(len(self.states))
        
        for i in range(0, len(self.states), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = self.states[batch_indices]
            batch_actions = self.actions[batch_indices]
            
            # 텐서로 변환
            states_tensor = torch.FloatTensor(batch_states).to(self.device)
            # 액션은 나중에 discrete/continuous에 따라 변환
            actions_tensor = torch.from_numpy(batch_actions).to(self.device)
            
            # TRM-PPO 모드 확인
            use_recurrent = getattr(self.agent, 'use_recurrent', False)
            
            # 이산 액션 처리
            if self.agent.actor_critic.discrete_action:
                # 이산 액션: LongTensor로 변환
                actions_tensor = actions_tensor.long()
                if actions_tensor.dim() == 1:
                    actions_tensor = actions_tensor.unsqueeze(-1)
            else:
                # 연속 액션: FloatTensor로 변환
                actions_tensor = actions_tensor.float()
                if actions_tensor.dim() == 1:
                    actions_tensor = actions_tensor.unsqueeze(-1)
            
            # 정책 네트워크로 액션 확률 계산
            if use_recurrent:
                # TRM-PPO: evaluate 사용
                log_probs, _, _ = self.agent.actor_critic.evaluate(
                    states_tensor,
                    actions_tensor,
                    n_cycles=self.agent.n_cycles
                )
            else:
                # 기존 PPO
                log_probs, _, _ = self.agent.actor_critic.evaluate(
                    states_tensor,
                    actions_tensor
                )
            
            # Supervised Learning: Negative log likelihood loss (최대 우도 추정)
            # 사람이 조작한 실제 액션의 로그 확률을 최대화
            # Loss = -log P(실제_액션 | 상태) → 최소화하면 P(실제_액션 | 상태) 최대화
            loss = -log_probs.mean()
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def pretrain(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        save_path: str = 'pretrained_model.pth',
        log_dir: str = 'runs',
        verbose: bool = True
    ):
        """
        Supervised Learning (Teacher Forcing) 사전 학습
        
        사람이 조작한 (상태, 액션) 쌍을 사용하여 정책을 supervised learning으로 학습
        
        Args:
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            save_path: 모델 저장 경로
            log_dir: TensorBoard 로그 디렉토리
            verbose: 상세 출력 여부
        
        Returns:
            final_loss: 최종 손실
        """
        print(f"\n{'='*60}")
        print("Supervised Learning (Teacher Forcing) 사전 학습 시작")
        print(f"{'='*60}")
        print(f"에폭 수: {epochs}")
        print(f"배치 크기: {batch_size}")
        print(f"데이터 크기: {len(self.states)}")
        print(f"{'='*60}\n")
        
        # TensorBoard 설정
        writer = None
        if HAS_TENSORBOARD:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(log_dir, f"teacher_forcing_{timestamp}")
            writer = SummaryWriter(log_path)
            print(f"📊 TensorBoard 로그: {log_path}")
            print(f"   실행: tensorboard --logdir={log_dir}\n")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 학습
            loss = self.train_epoch(batch_size)
            
            # 로깅
            if verbose and (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1:4d}/{epochs}] Loss: {loss:.6f}")
            
            if writer:
                writer.add_scalar('Train/Loss', loss, epoch)
            
            # 최고 모델 저장
            if loss < best_loss:
                best_loss = loss
                self.agent.save(save_path)
                if verbose:
                    print(f"  💾 최고 모델 저장: {save_path} (Loss: {loss:.6f})")
        
        if writer:
            writer.close()
        
        print(f"\n{'='*60}")
        print("Supervised Learning (Teacher Forcing) 사전 학습 완료")
        print(f"{'='*60}")
        print(f"최종 손실: {best_loss:.6f}")
        print(f"모델 저장: {save_path}")
        print(f"{'='*60}\n")
        
        return best_loss


def load_demonstrations(filepath: str):
    """
    저장된 데모 데이터 로드
    
    Args:
        filepath: 데모 데이터 파일 경로
    
    Returns:
        data: 로드된 데이터 (metadata, demonstrations)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ 데모 데이터 로드 완료: {filepath}")
    print(f"   에피소드 수: {data['metadata']['num_episodes']}")
    print(f"   총 스텝 수: {data['metadata']['total_steps']}")
    print(f"   환경 타입: {data['metadata']['env_type']}")
    
    return data


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='Supervised Learning (Teacher Forcing)을 사용한 사전 학습 및 강화학습',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 1단계: Supervised Learning 사전 학습만
  python train_with_teacher_forcing.py --demos human_demos.pkl --pretrain-epochs 100
  
  # 2단계: 사전 학습 + 강화학습 fine-tuning
  python train_with_teacher_forcing.py --demos human_demos.pkl --pretrain-epochs 100 --rl-steps 100000
  
  # 3단계: 기존 사전 학습 모델로 강화학습만
  python train_with_teacher_forcing.py --load pretrained_model.pth --rl-steps 100000
        """
    )
    
    # 데이터 설정
    parser.add_argument('--demos', type=str, default=None,
                        help='데모 데이터 파일 경로 (pickle 형식)')
    parser.add_argument('--load', type=str, default=None,
                        help='사전 학습된 모델 경로 (사전 학습 생략 시)')
    
    # Supervised Learning (Teacher Forcing) 설정
    parser.add_argument('--pretrain-epochs', type=int, default=0,
                        help='Supervised Learning 사전 학습 에폭 수 (0이면 생략)')
    parser.add_argument('--pretrain-batch-size', type=int, default=64,
                        help='사전 학습 배치 크기 (기본: 64)')
    parser.add_argument('--pretrain-lr', type=float, default=3e-4,
                        help='사전 학습 학습률 (기본: 3e-4)')
    parser.add_argument('--pretrain-save', type=str, default='pretrained_model.pth',
                        help='사전 학습 모델 저장 경로 (기본: pretrained_model.pth)')
    
    # 강화학습 설정
    parser.add_argument('--rl-steps', type=int, default=0,
                        help='강화학습 스텝 수 (0이면 생략)')
    parser.add_argument('--rl-env-type', choices=['carracing', 'sim', 'real'],
                        default='carracing',
                        help='강화학습 환경 타입 (기본: carracing)')
    parser.add_argument('--rl-port', type=str, default='/dev/ttyACM0',
                        help='시리얼 포트 (real 모드 사용 시)')
    parser.add_argument('--rl-save', type=str, default='ppo_model.pth',
                        help='강화학습 모델 저장 경로 (기본: ppo_model.pth)')
    
    # 네트워크 파라미터
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='히든 레이어 차원 (기본: 256)')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='TRM-PPO 잠재 상태 차원 (기본: 256)')
    parser.add_argument('--n-cycles', type=int, default=4,
                        help='TRM-PPO 재귀 추론 반복 횟수 (기본: 4)')
    
    # 디바이스
    parser.add_argument('--device', type=str, default=None,
                        help='디바이스 (cuda/cpu, 기본: 자동 선택)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"사용 디바이스: {device}")
    
    # 에이전트 생성
    agent = PPOAgent(
        state_dim=256,
        action_dim=5,  # 이산 액션
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_cycles=args.n_cycles,
        carry_latent=True,
        device=device,
        discrete_action=True,
        num_discrete_actions=5,
        use_recurrent=True
    )
    
    # 기존 모델 로드 (있는 경우)
    if args.load:
        if os.path.exists(args.load):
            agent.load(args.load)
            print(f"✅ 모델 로드 완료: {args.load}")
        else:
            print(f"⚠️  모델 파일을 찾을 수 없습니다: {args.load}")
    
    # Supervised Learning (Teacher Forcing) 사전 학습
    if args.pretrain_epochs > 0:
        if args.demos is None:
            print("❌ Supervised Learning을 사용하려면 --demos 옵션이 필요합니다.")
            sys.exit(1)
        
        if not os.path.exists(args.demos):
            print(f"❌ 데모 데이터 파일을 찾을 수 없습니다: {args.demos}")
            sys.exit(1)
        
        # 데모 데이터 로드
        demo_data = load_demonstrations(args.demos)
        demonstrations = demo_data['demonstrations']
        
        # Supervised Learning 학습
        trainer = TeacherForcingTrainer(
            agent=agent,
            demonstrations=demonstrations,
            device=device,
            lr=args.pretrain_lr
        )
        
        trainer.pretrain(
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            save_path=args.pretrain_save,
            verbose=True
        )
    
    # 강화학습 fine-tuning
    if args.rl_steps > 0:
        print(f"\n{'='*60}")
        print("강화학습 Fine-tuning 시작")
        print(f"{'='*60}\n")
        
        # 환경 생성
        if args.rl_env_type == 'carracing':
            try:
                env = CarRacingEnvWrapper(
                    max_steps=1000,
                    use_extended_actions=True,
                    use_discrete_actions=True
                )
            except ImportError as e:
                print(f"❌ CarRacing 환경을 사용할 수 없습니다: {e}")
                sys.exit(1)
        elif args.rl_env_type == 'sim':
            env = RCCarSimEnv(
                max_steps=1000,
                use_extended_actions=True,
                use_discrete_actions=True
            )
        else:  # real
            try:
                from rc_car_env import RCCarEnv
                env = RCCarEnv(
                    max_steps=1000,
                    use_extended_actions=True,
                    use_discrete_actions=True
                )
            except ImportError:
                print("❌ 실제 하드웨어 환경을 사용할 수 없습니다.")
                sys.exit(1)
        
        # 강화학습 실행
        train_ppo(
            env=env,
            agent=agent,
            total_steps=args.rl_steps,
            max_episode_steps=1000,
            update_frequency=2048,
            update_epochs=10,
            save_frequency=10000,
            save_path=args.rl_save,
            use_tensorboard=True,
            log_dir='runs',
            mc_update_on_done=False
        )
        
        env.close()
    
    print("\n✅ 모든 학습 완료!")


if __name__ == "__main__":
    main()

