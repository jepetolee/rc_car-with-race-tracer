#!/usr/bin/env python3
"""
사람 평가 기반 강화학습
사람이 모델의 주행을 평가하여 강화학습 진행

사용법:
    python train_human_feedback.py --model ppo_model.pth --port /dev/ttyACM0
"""

import argparse
import numpy as np
import torch
import time
import sys
import os
from datetime import datetime
from collections import deque

# 환경 및 에이전트 임포트
from rc_car_env import RCCarEnv
from ppo_agent import PPOAgent
from rc_car_controller import RCCarController

# TensorBoard 지원
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class HumanFeedbackTrainer:
    """
    사람 평가 기반 강화학습 클래스
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        port: str = '/dev/ttyACM0',
        max_steps: int = 1000,
        action_delay: float = 0.1
    ):
        """
        Args:
            agent: PPO 에이전트
            port: 시리얼 포트
            max_steps: 최대 스텝 수
            action_delay: 액션 간 지연 시간
        """
        self.agent = agent
        self.port = port
        self.max_steps = max_steps
        self.action_delay = action_delay
        
        # 실제 하드웨어 환경 및 제어기
        self.env = RCCarEnv(
            max_steps=max_steps,
            use_extended_actions=True,
            use_discrete_actions=True
        )
        
        self.controller = RCCarController(port=port, delay=action_delay)
        
        # 평가 데이터 저장소
        self.evaluation_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],  # 사람이 준 평가 점수
            'dones': [],
            'log_probs': [],
            'values': [],
            'latents': []
        }
    
    def run_episode_for_evaluation(self, verbose: bool = True):
        """
        평가를 위한 에피소드 실행
        
        Returns:
            episode_data: 에피소드 데이터
        """
        # 환경 리셋
        state = self.env.reset()
        
        # TRM-PPO: 잠재 상태 초기화
        if hasattr(self.agent, 'use_recurrent') and self.agent.use_recurrent:
            self.agent.reset_carry()
        
        episode_data = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'latents': []
        }
        
        if verbose:
            print("\n" + "="*60)
            print("모델 주행 평가 중...")
            print("="*60)
        
        try:
            for step in range(self.max_steps):
                # 상태 정규화
                state_normalized = state.astype(np.float32) / 255.0
                state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(self.agent.device)
                
                # 액션 선택
                if hasattr(self.agent, 'use_recurrent') and self.agent.use_recurrent:
                    action, log_prob, value, latent_np = self.agent.get_action_with_carry(
                        state_tensor, deterministic=False
                    )
                else:
                    action, log_prob, value = self.agent.actor_critic.get_action(state_tensor)
                    latent_np = None
                
                # 액션 변환
                action_np = int(action.squeeze(0).cpu().detach().numpy().item())
                log_prob_np = log_prob.squeeze(0).cpu().item() if log_prob is not None else 0.0
                value_np = value.squeeze(0).cpu().item()
                
                # 실제 하드웨어 제어
                self.controller.execute_discrete_action(action_np)
                
                # 환경 스텝
                next_state, _, done, info = self.env.step(action_np)
                
                # 데이터 저장
                episode_data['states'].append(state_normalized.copy())
                episode_data['actions'].append(action_np)
                episode_data['log_probs'].append(log_prob_np)
                episode_data['values'].append(value_np)
                if latent_np is not None:
                    episode_data['latents'].append(latent_np)
                
                if verbose and (step + 1) % 50 == 0:
                    action_name = {
                        0: "Stop", 1: "Right+Gas", 2: "Left+Gas",
                        3: "Gas", 4: "Brake"
                    }.get(action_np, f"Action {action_np}")
                    print(f"[Step {step+1:4d}] Action: {action_name}")
                
                time.sleep(self.action_delay)
                
                if done:
                    break
                
                state = next_state
        
        except KeyboardInterrupt:
            print("\n⚠️  사용자에 의해 중단되었습니다.")
        
        # 정지
        self.controller.stop()
        
        return episode_data
    
    def get_human_feedback(self, episode_data: dict):
        """
        사람으로부터 평가 점수 받기
        
        Args:
            episode_data: 에피소드 데이터
        
        Returns:
            feedback_score: 평가 점수 (0.0 ~ 1.0)
        """
        print("\n" + "="*60)
        print("주행 평가")
        print("="*60)
        print("에피소드 길이:", len(episode_data['states']), "스텝")
        print("\n평가 점수를 입력하세요 (0.0 ~ 1.0):")
        print("  0.0: 매우 나쁨")
        print("  0.5: 보통")
        print("  1.0: 매우 좋음")
        print("="*60)
        
        while True:
            try:
                score = float(input("점수 (0.0-1.0): "))
                if 0.0 <= score <= 1.0:
                    return score
                else:
                    print("⚠️  0.0과 1.0 사이의 값을 입력하세요.")
            except ValueError:
                print("⚠️  숫자를 입력하세요.")
            except KeyboardInterrupt:
                print("\n⚠️  평가 취소")
                return None
    
    def train_with_feedback(
        self,
        num_episodes: int = 10,
        feedback_weight: float = 1.0,
        save_path: str = 'ppo_model_feedback.pth',
        log_dir: str = 'runs'
    ):
        """
        사람 평가 기반 강화학습
        
        Args:
            num_episodes: 평가할 에피소드 수
            feedback_weight: 피드백 가중치
            save_path: 모델 저장 경로
            log_dir: TensorBoard 로그 디렉토리
        """
        print(f"\n{'='*60}")
        print("사람 평가 기반 강화학습 시작")
        print(f"{'='*60}")
        print(f"에피소드 수: {num_episodes}")
        print(f"피드백 가중치: {feedback_weight}")
        print(f"{'='*60}\n")
        
        # TensorBoard 설정
        writer = None
        if HAS_TENSORBOARD:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(log_dir, f"human_feedback_{timestamp}")
            writer = SummaryWriter(log_path)
            print(f"📊 TensorBoard 로그: {log_path}\n")
        
        episode_scores = []
        
        for episode in range(num_episodes):
            print(f"\n>>> 에피소드 {episode + 1}/{num_episodes} <<<")
            
            # 에피소드 실행
            episode_data = self.run_episode_for_evaluation(verbose=True)
            
            # 사람 평가 받기
            feedback_score = self.get_human_feedback(episode_data)
            
            if feedback_score is None:
                print("⚠️  평가가 취소되었습니다. 이 에피소드는 건너뜁니다.")
                continue
            
            episode_scores.append(feedback_score)
            
            # 피드백을 리워드로 변환 (정규화)
            # feedback_score를 -1.0 ~ 1.0 범위로 변환
            normalized_reward = (feedback_score - 0.5) * 2.0  # 0.0 -> -1.0, 1.0 -> 1.0
            
            # 모든 스텝에 동일한 피드백 리워드 적용
            rewards = [normalized_reward * feedback_weight] * len(episode_data['states'])
            dones = [False] * (len(episode_data['states']) - 1) + [True]
            
            # 버퍼에 저장
            for i in range(len(episode_data['states'])):
                self.evaluation_buffer['states'].append(episode_data['states'][i])
                self.evaluation_buffer['actions'].append(episode_data['actions'][i])
                self.evaluation_buffer['rewards'].append(rewards[i])
                self.evaluation_buffer['dones'].append(dones[i])
                self.evaluation_buffer['log_probs'].append(episode_data['log_probs'][i])
                self.evaluation_buffer['values'].append(episode_data['values'][i])
                if i < len(episode_data.get('latents', [])):
                    self.evaluation_buffer['latents'].append(episode_data['latents'][i])
            
            # 일정량 쌓이면 업데이트
            if len(self.evaluation_buffer['states']) >= 512:  # 작은 배치로 업데이트
                print(f"\n업데이트 중... (버퍼 크기: {len(self.evaluation_buffer['states'])})")
                loss_info = self._update_from_buffer()
                
                if writer:
                    writer.add_scalar('Train/Loss', loss_info.get('loss', 0), episode)
                    writer.add_scalar('Train/PolicyLoss', loss_info.get('policy_loss', 0), episode)
                    writer.add_scalar('Train/ValueLoss', loss_info.get('value_loss', 0), episode)
                    writer.add_scalar('Feedback/Score', feedback_score, episode)
                    writer.add_scalar('Feedback/AvgScore', np.mean(episode_scores), episode)
                
                # 버퍼 초기화
                self.evaluation_buffer = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'dones': [],
                    'log_probs': [],
                    'values': [],
                    'latents': []
                }
            
            # 모델 저장
            if (episode + 1) % 5 == 0:
                self.agent.save(save_path)
                print(f"💾 모델 저장: {save_path}")
            
            # 통계 출력
            print(f"\n평가 점수: {feedback_score:.3f}")
            print(f"평균 점수: {np.mean(episode_scores):.3f} ± {np.std(episode_scores):.3f}")
            
            # 다음 에피소드 준비
            if episode < num_episodes - 1:
                print("\n다음 에피소드를 준비하세요... (3초 후 시작)")
                time.sleep(3)
        
        # 최종 업데이트
        if len(self.evaluation_buffer['states']) > 0:
            print(f"\n최종 업데이트 중... (버퍼 크기: {len(self.evaluation_buffer['states'])})")
            self._update_from_buffer()
        
        # 최종 저장
        self.agent.save(save_path)
        
        if writer:
            writer.close()
        
        print(f"\n{'='*60}")
        print("사람 평가 기반 강화학습 완료")
        print(f"{'='*60}")
        print(f"평균 평가 점수: {np.mean(episode_scores):.3f} ± {np.std(episode_scores):.3f}")
        print(f"최고 점수: {np.max(episode_scores):.3f}")
        print(f"최저 점수: {np.min(episode_scores):.3f}")
        print(f"모델 저장: {save_path}")
        print(f"{'='*60}\n")
    
    def _update_from_buffer(self):
        """버퍼 데이터로 모델 업데이트"""
        if len(self.evaluation_buffer['states']) == 0:
            return {}
        
        # 버퍼를 텐서로 변환
        states = torch.FloatTensor(np.array(self.evaluation_buffer['states'])).to(self.agent.device)
        actions = torch.LongTensor(np.array(self.evaluation_buffer['actions'])).to(self.agent.device)
        old_log_probs = torch.FloatTensor(np.array(self.evaluation_buffer['log_probs'])).to(self.agent.device)
        old_values = torch.FloatTensor(np.array(self.evaluation_buffer['values'])).to(self.agent.device)
        rewards = np.array(self.evaluation_buffer['rewards'])
        dones = np.array(self.evaluation_buffer['dones'])
        
        # 잠재 상태 텐서
        latents = None
        if hasattr(self.agent, 'use_recurrent') and self.agent.use_recurrent:
            if len(self.evaluation_buffer['latents']) > 0:
                latents = torch.FloatTensor(np.array(self.evaluation_buffer['latents'])).to(self.agent.device)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
        
        # 리턴 계산 (Monte Carlo)
        returns = []
        running_return = 0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                running_return = 0
            running_return = rewards[step] + self.agent.gamma * running_return
            returns.insert(0, running_return)
        
        advantages = [r - v for r, v in zip(returns, old_values.cpu().numpy())]
        advantages = torch.FloatTensor(advantages).to(self.agent.device)
        returns = torch.FloatTensor(returns).to(self.agent.device)
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트 (간단한 버전)
        epochs = 3
        total_loss = 0
        
        for epoch in range(epochs):
            # 현재 정책으로 평가 (이산 액션)
            if hasattr(self.agent, 'use_recurrent') and self.agent.use_recurrent:
                log_probs, values, entropy = self.agent.actor_critic.evaluate(
                    states, actions, latent=latents, n_cycles=self.agent.n_cycles
                )
            else:
                log_probs, values, entropy = self.agent.actor_critic.evaluate(states, actions)
            
            # 정책 비율
            ratio = torch.exp(log_probs - old_log_probs)
            
            # PPO 클리핑 손실
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.agent.clip_epsilon, 1 + self.agent.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 가치 함수 손실
            value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns)
            
            # 엔트로피 손실
            entropy_loss = -entropy.mean()
            
            # 총 손실
            loss = policy_loss + self.agent.value_coef * value_loss + self.agent.entropy_coef * entropy_loss
            
            # 역전파
            self.agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), self.agent.max_grad_norm)
            self.agent.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'loss': total_loss / epochs,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item()
        }
    
    def close(self):
        """리소스 정리"""
        if self.env:
            self.env.close()
        if self.controller:
            self.controller.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='사람 평가 기반 강화학습',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python train_human_feedback.py --model ppo_model.pth --port /dev/ttyACM0 --episodes 10
        """
    )
    
    # 모델 설정
    parser.add_argument('--model', type=str, required=True,
                        help='학습된 모델 경로')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                        help='시리얼 포트 (기본: /dev/ttyACM0)')
    
    # 학습 설정
    parser.add_argument('--episodes', type=int, default=10,
                        help='평가할 에피소드 수 (기본: 10)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='최대 스텝 수 (기본: 1000)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='액션 간 지연 시간 (초, 기본: 0.1)')
    parser.add_argument('--feedback-weight', type=float, default=1.0,
                        help='피드백 가중치 (기본: 1.0)')
    
    # 저장 설정
    parser.add_argument('--save', type=str, default='ppo_model_feedback.pth',
                        help='모델 저장 경로 (기본: ppo_model_feedback.pth)')
    
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
        action_dim=5,  # 이산 액션만
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_cycles=args.n_cycles,
        carry_latent=True,
        device=device,
        discrete_action=True,  # 이산 액션만
        num_discrete_actions=5,
        use_recurrent=True
    )
    
    # 모델 로드
    if os.path.exists(args.model):
        agent.load(args.model)
        print(f"✅ 모델 로드 완료: {args.model}")
    else:
        print(f"⚠️  모델 파일을 찾을 수 없습니다: {args.model}")
        print("랜덤 정책으로 시작합니다.")
    
    # 학습기 생성
    trainer = HumanFeedbackTrainer(
        agent=agent,
        port=args.port,
        max_steps=args.max_steps,
        action_delay=args.delay
    )
    
    try:
        # 사람 평가 기반 강화학습
        trainer.train_with_feedback(
            num_episodes=args.episodes,
            feedback_weight=args.feedback_weight,
            save_path=args.save
        )
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

