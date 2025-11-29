#!/usr/bin/env python3
"""
Imitation Learning via Reinforcement Learning
사용자 데모 데이터와의 일치율을 리워드로 사용하여 강화학습

Supervised Learning이 아닌 강화학습으로:
- 모델이 액션을 선택
- 사용자가 선택한 액션과 비교
- 일치율에 따라 리워드 부여
- PPO로 학습
"""

import os
# 환경 변수 설정
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'

import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import sys

from ppo_agent import PPOAgent
from train_ppo import train_ppo


class ImitationRLTrainer:
    """
    사용자 데모 데이터와의 일치율을 리워드로 사용하는 강화학습
    """
    
    def __init__(
        self,
        demos_path: str,
        model_path: str = None,
        device: str = 'cpu',
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        update_epochs: int = 10
    ):
        """
        Args:
            demos_path: 사용자 데모 데이터 경로 (pickle 파일)
            model_path: 사전 학습된 모델 경로 (선택)
            device: 디바이스
            learning_rate: 학습률
            gamma: 할인 계수
            gae_lambda: GAE 람다
            clip_epsilon: PPO clip epsilon
            value_coef: Value loss 계수
            entropy_coef: Entropy 계수
            max_grad_norm: Gradient clipping
            batch_size: 배치 크기
            update_epochs: 업데이트 에폭 수
        """
        self.demos_path = demos_path
        self.device = device
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        
        # 데모 데이터 로드
        print(f"📂 데모 데이터 로드: {demos_path}")
        with open(demos_path, 'rb') as f:
            data = pickle.load(f)
        
        self.demos = data.get('demonstrations', [])
        if len(self.demos) == 0:
            raise ValueError("데모 데이터가 비어있습니다.")
        
        print(f"✅ {len(self.demos)}개 에피소드 로드 완료")
        
        # 모든 (state, action) 쌍 추출
        self.demo_states = []
        self.demo_actions = []
        
        for episode in self.demos:
            states = episode.get('states', [])
            actions = episode.get('actions', [])
            
            if len(states) != len(actions):
                print(f"⚠️  에피소드 길이 불일치: states={len(states)}, actions={len(actions)}")
                min_len = min(len(states), len(actions))
                states = states[:min_len]
                actions = actions[:min_len]
            
            self.demo_states.extend(states)
            self.demo_actions.extend(actions)
        
        print(f"✅ 총 {len(self.demo_states)}개 (state, action) 쌍")
        
        # 에이전트 생성
        print(f"\n🤖 에이전트 생성...")
        self.agent = PPOAgent(
            state_dim=256,
            action_dim=5,
            discrete_action=True,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            use_recurrent=True,  # Deep Supervision을 위해 Recurrent 활성화
            deep_supervision=True
        )
        
        # 사전 학습된 모델 로드
        if model_path and os.path.exists(model_path):
            print(f"📥 사전 학습된 모델 로드: {model_path}")
            try:
                self.agent.load(model_path)
                print("✅ 모델 로드 완료")
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {e}")
                print("랜덤 초기화로 시작합니다.")
    
    def compute_imitation_reward(self, predicted_action: int, expert_action: int) -> float:
        """
        모델의 액션과 전문가 액션의 일치율에 따른 리워드 계산
        
        Args:
            predicted_action: 모델이 선택한 액션
            expert_action: 사용자가 선택한 액션
        
        Returns:
            reward: 일치하면 1.0, 불일치하면 -0.1
        """
        if predicted_action == expert_action:
            return 1.0  # 완전 일치
        else:
            return -0.1  # 불일치 페널티
    
    def train_step(self, states: np.ndarray, expert_actions: np.ndarray):
        """
        단일 학습 스텝 (TRM 스타일 Step-wise Update)
        
        Args:
            states: 상태 배열 [batch_size, 256]
            expert_actions: 전문가 액션 배열 [batch_size]
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        expert_actions_tensor = torch.LongTensor(expert_actions).to(self.device)
        
        # 초기 액션 선택 (리워드 계산용)
        actions, log_probs, values = self.agent.actor_critic.get_action(states_tensor)
        actions_np = actions.cpu().numpy().flatten()
        
        # 리워드 계산 (일치율 기반)
        rewards = np.array([
            self.compute_imitation_reward(pred, expert)
            for pred, expert in zip(actions_np, expert_actions)
        ])
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # Advantage 계산
        advantages = rewards_tensor - values.squeeze()
        old_log_probs = log_probs.detach()
        
        # 통계 누적용
        total_loss_sum = 0
        total_actor_loss_sum = 0
        total_value_loss_sum = 0
        total_entropy_sum = 0
        
        # TRM 스타일: Step-wise Update (K번 반복)
        if self.agent.use_recurrent and self.agent.deep_supervision:
            batch_size = states_tensor.shape[0]
            # Latent 초기화
            latent = self.agent.actor_critic.init_latent.unsqueeze(0).expand(batch_size, -1).clone()
            
            # K번의 Supervision Loop
            for step in range(self.agent.n_supervision_steps):
                # 1. State Encoding
                state_emb = self.agent.actor_critic.encoder(states_tensor)
                
                # 2. Deep Recursion (One Step of M x N)
                next_latent, latent_grad, value, action_output = self.agent.actor_critic.deep_recursion(
                    state_emb, latent, self.agent.n_deep_loops, self.agent.n_latent_loops
                )
                
                # 3. Loss Calculation for THIS step
                value_pred = value.squeeze(-1)
                value_loss = F.mse_loss(value_pred, rewards_tensor)
                
                # Policy Loss & Entropy
                if self.agent.actor_critic.discrete_action:
                    action_logits = action_output
                    dist = torch.distributions.Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(expert_actions_tensor.squeeze(-1))
                    entropy = dist.entropy().mean()
                else:
                    action_mean, action_log_std = action_output
                    std = torch.exp(action_log_std)
                    dist = torch.distributions.Normal(action_mean, std)
                    action_inv = torch.atanh(torch.clamp(expert_actions_tensor, -0.999, 0.999))
                    log_prob = dist.log_prob(action_inv).sum(dim=-1, keepdim=True)
                    log_prob -= torch.log(1 - torch.tanh(action_inv).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
                    new_log_probs = log_prob
                    entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
                
                # Ratio & Surrogate Loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.agent.clip_epsilon, 1 + self.agent.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Total Loss for this step
                loss = actor_loss + self.agent.value_coef * value_loss - self.agent.entropy_coef * entropy
                
                # 4. Backward & Update (IMMEDIATELY - TRM Style)
                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), self.agent.max_grad_norm)
                self.agent.optimizer.step()
                
                # 통계 누적
                total_loss_sum += loss.item()
                total_actor_loss_sum += actor_loss.item()
                total_value_loss_sum += value_loss.item()
                total_entropy_sum += entropy.item()
                
                # 5. Pass detached latent to next step
                latent = next_latent
        else:
            # 기존 방식 (Non-recurrent 또는 Deep Supervision 비활성화)
            new_log_probs, new_values, entropy = self.agent.actor_critic.evaluate(
                states_tensor, expert_actions_tensor.unsqueeze(-1)
            )
            
            # PPO 손실 계산
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.agent.clip_epsilon, 1.0 + self.agent.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), rewards_tensor)
            
            # Entropy
            entropy_loss = -entropy.mean()
            
            # 총 손실
            total_loss = (
                actor_loss +
                self.agent.value_coef * value_loss +
                self.agent.entropy_coef * entropy_loss
            )
            
            # 역전파
            self.agent.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), self.agent.max_grad_norm)
            self.agent.optimizer.step()
            
            total_loss_sum = total_loss.item()
            total_actor_loss_sum = actor_loss.item()
            total_value_loss_sum = value_loss.item()
            total_entropy_sum = entropy.mean().item()
        
        # 통계 (평균 계산)
        if self.agent.use_recurrent and self.agent.deep_supervision:
            n_steps = self.agent.n_supervision_steps
        else:
            n_steps = 1
        
        match_rate = np.mean(actions_np == expert_actions)
        avg_reward = np.mean(rewards)
        
        return {
            'total_loss': total_loss_sum / n_steps,
            'actor_loss': total_actor_loss_sum / n_steps,
            'value_loss': total_value_loss_sum / n_steps,
            'entropy': total_entropy_sum / n_steps,
            'match_rate': match_rate,
            'avg_reward': avg_reward
        }
    
    def train(self, epochs: int = 100, save_path: str = 'imitation_rl_model.pth', verbose: bool = True):
        """
        학습 실행
        
        Args:
            epochs: 학습 에폭 수
            save_path: 모델 저장 경로
            verbose: 상세 출력
        """
        print(f"\n{'='*60}")
        print("Imitation Learning via Reinforcement Learning 시작")
        print(f"{'='*60}")
        print(f"데모 데이터: {len(self.demo_states)}개 샘플")
        print(f"학습 에폭: {epochs}")
        print(f"배치 크기: {self.batch_size}")
        print(f"{'='*60}\n")
        
        # 데이터를 텐서로 변환
        states_array = np.array(self.demo_states)  # [N, 256]
        actions_array = np.array(self.demo_actions)  # [N]
        
        # 데이터 셔플
        indices = np.arange(len(states_array))
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            shuffled_states = states_array[indices]
            shuffled_actions = actions_array[indices]
            
            epoch_stats = {
                'total_loss': [],
                'actor_loss': [],
                'value_loss': [],
                'entropy': [],
                'match_rate': [],
                'avg_reward': []
            }
            
            # 배치별 학습
            for i in range(0, len(shuffled_states), self.batch_size):
                batch_states = shuffled_states[i:i+self.batch_size]
                batch_actions = shuffled_actions[i:i+self.batch_size]
                
                if len(batch_states) < self.batch_size:
                    continue
                
                # 여러 번 업데이트
                for _ in range(self.update_epochs):
                    stats = self.train_step(batch_states, batch_actions)
                    
                    for key, value in stats.items():
                        epoch_stats[key].append(value)
            
            # 에폭 통계
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Loss: {np.mean(epoch_stats['total_loss']):.4f}")
                print(f"  Match Rate: {np.mean(epoch_stats['match_rate']):.2%}")
                print(f"  Avg Reward: {np.mean(epoch_stats['avg_reward']):.4f}")
                print()
        
        # 모델 저장
        self.agent.save(save_path)
        print(f"\n✅ 모델 저장 완료: {save_path}")
        
        # 최종 평가
        print("\n최종 평가 중...")
        final_match_rate = self.evaluate()
        print(f"최종 일치율: {final_match_rate:.2%}")
    
    def evaluate(self, num_samples: int = 1000) -> float:
        """
        모델 평가 (일치율 계산)
        
        Args:
            num_samples: 평가할 샘플 수
        
        Returns:
            match_rate: 일치율 (0.0 ~ 1.0)
        """
        self.agent.actor_critic.eval()
        
        indices = np.random.choice(len(self.demo_states), min(num_samples, len(self.demo_states)), replace=False)
        test_states = np.array([self.demo_states[i] for i in indices])
        test_actions = np.array([self.demo_actions[i] for i in indices])
        
        states_tensor = torch.FloatTensor(test_states).to(self.device)
        
        with torch.no_grad():
            actions, _, _ = self.agent.actor_critic.get_action(states_tensor)
            actions_np = actions.cpu().numpy().flatten()
        
        match_rate = np.mean(actions_np == test_actions)
        
        self.agent.actor_critic.train()
        
        return match_rate


def main():
    parser = argparse.ArgumentParser(
        description='Imitation Learning via Reinforcement Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 학습
  python train_imitation_rl.py --demos human_demos.pkl --epochs 100
  
  # 사전 학습된 모델로 시작
  python train_imitation_rl.py --demos human_demos.pkl --model pretrained.pth --epochs 100
        """
    )
    
    parser.add_argument('--demos', type=str, required=True,
                        help='사용자 데모 데이터 경로 (pickle 파일)')
    parser.add_argument('--model', type=str, default=None,
                        help='사전 학습된 모델 경로 (선택)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에폭 수 (기본: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='배치 크기 (기본: 64)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='학습률 (기본: 3e-4)')
    parser.add_argument('--save', type=str, default='imitation_rl_model.pth',
                        help='모델 저장 경로 (기본: imitation_rl_model.pth)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='디바이스 (기본: cpu)')
    
    args = parser.parse_args()
    
    # Trainer 생성
    trainer = ImitationRLTrainer(
        demos_path=args.demos,
        model_path=args.model,
        device=args.device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # 학습 실행
    trainer.train(
        epochs=args.epochs,
        save_path=args.save,
        verbose=True
    )


if __name__ == '__main__':
    main()

