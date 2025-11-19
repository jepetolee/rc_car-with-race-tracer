#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) 에이전트 구현
PyTorch를 사용한 직접 구현
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


class ActorCritic(nn.Module):
    """
    Actor-Critic 네트워크
    Actor: 정책 네트워크 (액션 확률 분포 출력)
    Critic: 가치 네트워크 (상태 가치 출력)
    """
    
    def __init__(self, state_dim=256, action_dim=2, hidden_dim=256, discrete_action=False, num_discrete_actions=5):
        """
        Args:
            state_dim: 상태 차원 (16x16 이미지 = 256)
            action_dim: 액션 차원 (연속 액션: left_speed, right_speed = 2)
            hidden_dim: 히든 레이어 차원
            discrete_action: 이산 액션 공간 사용 여부
            num_discrete_actions: 이산 액션 개수 (기본: 5)
        """
        super(ActorCritic, self).__init__()
        
        self.discrete_action = discrete_action
        self.num_discrete_actions = num_discrete_actions
        
        # 공통 특징 추출 레이어
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor 헤드 (정책 네트워크)
        if discrete_action:
            # 이산 액션 공간: Categorical 분포 (5개 액션)
            self.actor = nn.Linear(hidden_dim // 2, num_discrete_actions)
        else:
            # 연속 액션 공간: 평균과 표준편차 출력
            self.actor_mean = nn.Linear(hidden_dim // 2, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic 헤드 (가치 네트워크)
        self.critic = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state):
        """
        순전파
        
        Args:
            state: 상태 텐서
        
        Returns:
            이산 액션: (action_logits, None, value)
            연속 액션: (action_mean, action_log_std, value)
        """
        features = self.feature(state)
        value = self.critic(features)
        
        if self.discrete_action:
            # 이산 액션: 로그 확률 출력
            action_logits = self.actor(features)
            return action_logits, None, value
        else:
            # 연속 액션: 평균과 표준편차
            action_mean = torch.tanh(self.actor_mean(features))
            action_log_std = self.actor_log_std.expand_as(action_mean)
            return action_mean, action_log_std, value
    
    def get_action(self, state, deterministic=False):
        """
        액션 샘플링
        
        Args:
            state: 상태 텐서
            deterministic: True면 최대 확률 액션 사용, False면 확률적 샘플링
        
        Returns:
            action: 샘플링된 액션 (이산: 정수, 연속: 텐서)
            log_prob: 로그 확률
            value: 상태 가치
        """
        if self.discrete_action:
            # 이산 액션
            action_logits, _, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=action_logits)
            
            if deterministic:
                action = dist.probs.argmax(dim=-1)
                log_prob = dist.log_prob(action)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action, log_prob.unsqueeze(-1), value
        else:
            # 연속 액션
            action_mean, action_log_std, value = self.forward(state)
            
            if deterministic:
                action = action_mean
                log_prob = None
            else:
                # 정규분포에서 샘플링
                std = torch.exp(action_log_std)
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                
                # 액션 범위 제한 (-1 ~ 1)
                action = torch.tanh(action)
                
                # Tanh 변환에 대한 로그 확률 조정
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            return action, log_prob, value
    
    def evaluate(self, state, action):
        """
        주어진 상태와 액션에 대한 평가
        
        Args:
            state: 상태 텐서
            action: 액션 텐서 (이산: 정수, 연속: 텐서)
        
        Returns:
            log_prob: 로그 확률
            value: 상태 가치
            entropy: 엔트로피
        """
        if self.discrete_action:
            # 이산 액션
            action_logits, _, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=action_logits)
            
            # 액션을 정수로 변환
            if action.dtype != torch.long:
                action = action.long()
            
            log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
            
            return log_prob, value, entropy
        else:
            # 연속 액션
            action_mean, action_log_std, value = self.forward(state)
            
            std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, std)
            
            # 액션을 역변환 (tanh)
            action = torch.atanh(torch.clamp(action, -0.999, 0.999))
            
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            
            return log_prob, value, entropy


class PPOAgent:
    """
    PPO 에이전트
    """
    
    def __init__(
        self,
        state_dim=256,
        action_dim=2,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        discrete_action=False,
        num_discrete_actions=5
    ):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 액션 차원
            hidden_dim: 히든 레이어 차원
            lr_actor: Actor 학습률
            lr_critic: Critic 학습률
            gamma: 할인율
            gae_lambda: GAE 람다
            clip_epsilon: PPO 클립 범위
            value_coef: 가치 함수 손실 계수
            entropy_coef: 엔트로피 보너스 계수
            max_grad_norm: 그래디언트 클리핑 최대 노름
            device: 디바이스 (cuda/cpu)
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Actor-Critic 네트워크
        self.actor_critic = ActorCritic(
            state_dim, 
            action_dim, 
            hidden_dim,
            discrete_action=discrete_action,
            num_discrete_actions=num_discrete_actions
        ).to(device)
        
        # 옵티마이저
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=lr_actor
        )
        
        # 리플레이 버퍼
        self.reset_buffer()
    
    def reset_buffer(self):
        """리플레이 버퍼 초기화"""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """트랜지션 저장"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
    
    def compute_gae(self, rewards, values, dones, next_value=0):
        """
        Generalized Advantage Estimation (GAE) 계산
        
        Args:
            rewards: 리워드 리스트
            values: 가치 리스트
            dones: 종료 플래그 리스트
            next_value: 다음 상태 가치
        
        Returns:
            advantages: 어드밴티지
            returns: 리턴
        """
        advantages = []
        gae = 0
        next_value = next_value
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self, epochs=10):
        """
        PPO 업데이트
        
        Args:
            epochs: 업데이트 에폭 수
        
        Returns:
            loss_info: 손실 정보 딕셔너리
        """
        if len(self.buffer['states']) == 0:
            return {}
        
        # 버퍼를 텐서로 변환
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(self.device)
        old_values = torch.FloatTensor(np.array(self.buffer['values'])).to(self.device)
        rewards = np.array(self.buffer['rewards'])
        dones = np.array(self.buffer['dones'])
        
        # GAE 계산
        next_value = 0  # 마지막 상태의 다음 가치는 0 (또는 실제 다음 상태 가치 사용)
        advantages, returns = self.compute_gae(rewards, old_values.cpu().numpy(), dones, next_value)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # 여러 에폭 동안 업데이트
        for epoch in range(epochs):
            # 현재 정책으로 평가
            log_probs, values, entropy = self.actor_critic.evaluate(states, actions)
            
            # 정책 비율
            ratio = torch.exp(log_probs - old_log_probs)
            
            # PPO 클리핑 손실
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 가치 함수 손실
            value_loss = F.mse_loss(values, returns)
            
            # 엔트로피 손실
            entropy_loss = -entropy.mean()
            
            # 총 손실
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # 버퍼 초기화
        self.reset_buffer()
        
        return {
            'loss': total_loss / epochs,
            'policy_loss': total_policy_loss / epochs,
            'value_loss': total_value_loss / epochs,
            'entropy': total_entropy / epochs
        }
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # 테스트 코드
    agent = PPOAgent(state_dim=256, action_dim=2)
    
    # 더미 데이터로 테스트
    state = np.random.rand(256).astype(np.uint8)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    
    action, log_prob, value = agent.actor_critic.get_action(state_tensor)
    print(f"Action: {action}, Log Prob: {log_prob}, Value: {value}")

