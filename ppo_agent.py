#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) 에이전트 구현
TRM (Tiny Reasoning Model) 스타일 재귀 추론 + HRM 근사 그래디언트 적용
PyTorch를 사용한 직접 구현
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class LatentCarry:
    """에피소드 내 잠재 상태를 carry-over하기 위한 데이터 클래스"""
    latent: torch.Tensor
    
    def detach(self) -> 'LatentCarry':
        """그래디언트 분리"""
        return LatentCarry(latent=self.latent.detach())


class RecurrentReasoningBlock(nn.Module):
    """
    TRM 스타일 단일 재귀 추론 블록
    2-layer MLP with residual connection
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int):
        """
        Args:
            latent_dim: 잠재 상태 차원
            hidden_dim: 히든 레이어 차원
        """
        super(RecurrentReasoningBlock, self).__init__()
        
        # 2-layer MLP (TRM 스타일)
        # state_embedding + latent를 concat하여 입력
        self.reasoning = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, state_embedding: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        상태와 잠재 상태를 결합하여 추론 수행
        
        Args:
            state_embedding: 인코딩된 상태 [batch, latent_dim]
            latent: 현재 잠재 상태 [batch, latent_dim]
        
        Returns:
            updated_latent: 업데이트된 잠재 상태 [batch, latent_dim]
        """
        combined = torch.cat([state_embedding, latent], dim=-1)
        delta = self.reasoning(combined)
        # Residual connection with normalization
        return self.norm(latent + delta)


class RecurrentActorCritic(nn.Module):
    """
    TRM 스타일 재귀 추론을 사용하는 Actor-Critic 네트워크
    HRM의 근사 그래디언트 기법 적용 (내부 반복은 no_grad, 최종 단계만 grad)
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 2,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        n_cycles: int = 4,
        discrete_action: bool = False,
        num_discrete_actions: int = 5
    ):
        """
        Args:
            state_dim: 상태 차원 (16x16 이미지 = 256)
            action_dim: 액션 차원 (연속 액션: left_speed, right_speed = 2)
            latent_dim: 잠재 상태 차원
            hidden_dim: 히든 레이어 차원
            n_cycles: 재귀 추론 반복 횟수
            discrete_action: 이산 액션 공간 사용 여부
            num_discrete_actions: 이산 액션 개수
        """
        super(RecurrentActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.n_cycles = n_cycles
        self.discrete_action = discrete_action
        self.num_discrete_actions = num_discrete_actions
        
        # 상태 인코더: state -> state_embedding
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 초기 잠재 상태 (학습 가능)
        self.init_latent = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.init_latent, std=0.02)
        
        # TRM 스타일 재귀 추론 블록
        self.reasoning_block = RecurrentReasoningBlock(latent_dim, hidden_dim)
        
        # Actor 헤드 (정책 네트워크)
        if discrete_action:
            self.actor = nn.Linear(latent_dim, num_discrete_actions)
        else:
            self.actor_mean = nn.Linear(latent_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic 헤드 (가치 네트워크)
        self.critic = nn.Linear(latent_dim, 1)
    
    def init_carry(self, batch_size: int, device: torch.device) -> LatentCarry:
        """
        초기 잠재 상태 생성
        
        Args:
            batch_size: 배치 크기
            device: 디바이스
        
        Returns:
            초기 LatentCarry
        """
        latent = self.init_latent.unsqueeze(0).expand(batch_size, -1).to(device)
        return LatentCarry(latent=latent)
    
    def forward(
        self,
        state: torch.Tensor,
        carry: Optional[LatentCarry] = None,
        n_cycles: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, LatentCarry]:
        """
        순전파 with TRM 스타일 재귀 추론 + HRM 근사 그래디언트
        
        Args:
            state: 상태 텐서 [batch, state_dim]
            carry: 이전 잠재 상태 (None이면 초기화)
            n_cycles: 재귀 추론 횟수 (None이면 self.n_cycles 사용)
        
        Returns:
            이산 액션: (action_logits, None, value, new_carry)
            연속 액션: (action_mean, action_log_std, value, new_carry)
        """
        batch_size = state.shape[0]
        device = state.device
        
        if n_cycles is None:
            n_cycles = self.n_cycles
        
        # 상태 인코딩
        state_emb = self.encoder(state)
        
        # 잠재 상태 초기화 또는 carry-over
        if carry is None:
            latent = self.init_latent.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            latent = carry.latent
        
        # HRM 스타일 근사 그래디언트: 내부 반복은 no_grad
        if n_cycles > 1:
            with torch.no_grad():
                for _ in range(n_cycles - 1):
                    latent = self.reasoning_block(state_emb, latent)
        
        # 마지막 단계만 gradient 계산
        latent = self.reasoning_block(state_emb, latent.detach())
        
        # Actor/Critic 출력
        value = self.critic(latent)
        new_carry = LatentCarry(latent=latent.detach())
        
        if self.discrete_action:
            action_logits = self.actor(latent)
            return action_logits, None, value, new_carry
        else:
            action_mean = torch.tanh(self.actor_mean(latent))
            action_log_std = self.actor_log_std.expand_as(action_mean)
            return action_mean, action_log_std, value, new_carry
    
    def get_action(
        self,
        state: torch.Tensor,
        carry: Optional[LatentCarry] = None,
        deterministic: bool = False,
        n_cycles: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, LatentCarry]:
        """
        액션 샘플링 with 재귀 추론
        
        Args:
            state: 상태 텐서
            carry: 이전 잠재 상태
            deterministic: True면 최대 확률 액션 사용
            n_cycles: 재귀 추론 횟수
        
        Returns:
            action: 샘플링된 액션
            log_prob: 로그 확률
            value: 상태 가치
            new_carry: 새 잠재 상태
        """
        if self.discrete_action:
            action_logits, _, value, new_carry = self.forward(state, carry, n_cycles)
            dist = torch.distributions.Categorical(logits=action_logits)
            
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).unsqueeze(-1)
            return action, log_prob, value, new_carry
        else:
            action_mean, action_log_std, value, new_carry = self.forward(state, carry, n_cycles)
            
            if deterministic:
                action = action_mean
                log_prob = None
            else:
                std = torch.exp(action_log_std)
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                
                # 액션 범위 제한 (-1 ~ 1)
                action = torch.tanh(action)
                
                # Tanh 변환에 대한 로그 확률 조정
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            return action, log_prob, value, new_carry
    
    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        n_cycles: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        주어진 상태와 액션에 대한 평가
        
        Args:
            state: 상태 텐서
            action: 액션 텐서
            latent: 저장된 잠재 상태 (재현을 위해)
            n_cycles: 재귀 추론 횟수
        
        Returns:
            log_prob: 로그 확률
            value: 상태 가치
            entropy: 엔트로피
        """
        carry = LatentCarry(latent=latent) if latent is not None else None
        
        if self.discrete_action:
            action_logits, _, value, _ = self.forward(state, carry, n_cycles)
            dist = torch.distributions.Categorical(logits=action_logits)
            
            if action.dtype != torch.long:
                action = action.long()
            
            log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
            
            return log_prob, value, entropy
        else:
            action_mean, action_log_std, value, _ = self.forward(state, carry, n_cycles)
            
            std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, std)
            
            # 액션을 역변환 (tanh)
            action_inv = torch.atanh(torch.clamp(action, -0.999, 0.999))
            
            log_prob = dist.log_prob(action_inv).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1 - torch.tanh(action_inv).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            
            return log_prob, value, entropy


class ActorCritic(nn.Module):
    """
    Actor-Critic 네트워크
    Actor: 정책 네트워크 (액션 확률 분포 출력)
    Critic: 가치 네트워크 (상태 가치 출력)
    """
    
    def __init__(self, state_dim=256, action_dim=2, hidden_dim=64, discrete_action=False, num_discrete_actions=5):
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
    TRM-PPO 에이전트
    TRM 스타일 재귀 추론 + HRM 근사 그래디언트 적용
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 2,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        n_cycles: int = 4,
        carry_latent: bool = True,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        discrete_action: bool = False,
        num_discrete_actions: int = 5,
        use_recurrent: bool = True,
        use_monte_carlo: bool = False,
        total_steps: int = 1000000,
        lr_schedule: str = 'cosine'  # 'cosine', 'linear', 'none'
    ):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 액션 차원
            latent_dim: 잠재 상태 차원
            hidden_dim: 히든 레이어 차원
            n_cycles: 재귀 추론 반복 횟수
            carry_latent: 에피소드 내 잠재 상태 유지 여부
            lr_actor: Actor 학습률
            lr_critic: Critic 학습률
            gamma: 할인율
            gae_lambda: GAE 람다
            clip_epsilon: PPO 클립 범위
            value_coef: 가치 함수 손실 계수
            entropy_coef: 엔트로피 보너스 계수
            max_grad_norm: 그래디언트 클리핑 최대 노름
            device: 디바이스 (cuda/cpu)
            discrete_action: 이산 액션 공간 사용 여부
            num_discrete_actions: 이산 액션 개수
            use_recurrent: RecurrentActorCritic 사용 여부 (False면 기존 ActorCritic)
            use_monte_carlo: Monte Carlo 리턴 사용 여부 (True면 GAE 대신 MC 리턴)
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_cycles = n_cycles
        self.carry_latent = carry_latent
        self.use_recurrent = use_recurrent
        self.latent_dim = latent_dim
        self.use_monte_carlo = use_monte_carlo
        self.total_steps = total_steps
        self.lr_schedule = lr_schedule
        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic
        
        # Actor-Critic 네트워크 (TRM 스타일 또는 기존)
        if use_recurrent:
            self.actor_critic = RecurrentActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                n_cycles=n_cycles,
                discrete_action=discrete_action,
                num_discrete_actions=num_discrete_actions
            ).to(device)
        else:
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
        
        # 학습률 스케줄러
        if lr_schedule == 'cosine':
            # 코사인 감소: 초기 lr → 최종 lr (0.1 * initial_lr)
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=lr_actor * 0.1
            )
        elif lr_schedule == 'linear':
            # 선형 감소: 초기 lr → 최종 lr (0.1 * initial_lr)
            self.scheduler = lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        else:
            # 스케줄링 없음
            self.scheduler = None
        
        # 현재 잠재 상태 (에피소드 내 carry-over용)
        self.current_carry: Optional[LatentCarry] = None
        
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
            'values': [],
            'latents': []  # 잠재 상태 저장 (재현용)
        }
    
    def reset_carry(self):
        """에피소드 시작 시 잠재 상태 리셋"""
        self.current_carry = None
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        latent: Optional[np.ndarray] = None
    ):
        """
        트랜지션 저장
        
        Args:
            state: 상태
            action: 액션
            reward: 리워드
            done: 종료 플래그
            log_prob: 로그 확률
            value: 가치
            latent: 잠재 상태 (재현용)
        """
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
        if latent is not None:
            self.buffer['latents'].append(latent)
    
    def get_action_with_carry(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
        """
        잠재 상태 carry-over와 함께 액션 샘플링
        
        Args:
            state: 상태 텐서
            deterministic: True면 최대 확률 액션 사용
        
        Returns:
            action: 샘플링된 액션
            log_prob: 로그 확률
            value: 상태 가치
            latent_np: 잠재 상태 (numpy, 버퍼 저장용)
        """
        if self.use_recurrent:
            action, log_prob, value, new_carry = self.actor_critic.get_action(
                state, 
                carry=self.current_carry if self.carry_latent else None,
                deterministic=deterministic,
                n_cycles=self.n_cycles
            )
            
            # Carry-over 업데이트
            if self.carry_latent:
                self.current_carry = new_carry
            
            # 잠재 상태를 numpy로 변환 (버퍼 저장용)
            latent_np = new_carry.latent.detach().cpu().numpy()
            
            return action, log_prob, value, latent_np
        else:
            # 기존 ActorCritic 사용
            action, log_prob, value = self.actor_critic.get_action(state, deterministic)
            return action, log_prob, value, None
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0
    ) -> Tuple[list, list]:
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
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def compute_mc_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray
    ) -> Tuple[list, list]:
        """
        Monte Carlo 리턴 계산 (순수 에피소드 리턴, 부트스트래핑 없음)
        
        에피소드가 끝날 때까지의 실제 누적 리워드를 사용
        TD 추정치 대신 실제 리턴을 사용하여 더 안정적인 학습 보장
        
        Args:
            rewards: 리워드 리스트
            dones: 종료 플래그 리스트
            values: 가치 리스트 (어드밴티지 계산용)
        
        Returns:
            advantages: 어드밴티지 (returns - values)
            returns: Monte Carlo 리턴
        """
        n = len(rewards)
        returns = [0.0] * n
        running_return = 0
        
        # 역순으로 리턴 계산 (에피소드 경계 고려)
        # 중요: done 체크 전에 리턴을 계산해야 함
        for step in reversed(range(n)):
            # 먼저 현재 스텝의 리턴 계산
            running_return = rewards[step] + self.gamma * running_return
            returns[step] = running_return
            
            # 에피소드 종료 시 다음 에피소드를 위해 리셋
            # (다음 반복에서 이전 에피소드의 리턴이 누적되지 않도록)
            if dones[step]:
                running_return = 0
        
        # 어드밴티지 = MC 리턴 - 가치 추정치
        advantages = [r - v for r, v in zip(returns, values)]
        
        return advantages, returns
    
    def update(self, epochs: int = 10, progress: float = 0.0, return_gradients: bool = False) -> Dict[str, float]:
        """
        PPO 업데이트 with TRM 스타일 재귀 추론
        
        Args:
            epochs: 업데이트 에폭 수
            progress: 학습 진행률 [0, 1] (사용 안 함, 호환성 유지)
            return_gradients: True면 그래디언트만 계산하고 반환 (A3C용)
        
        Returns:
            loss_info: 손실 정보 딕셔너리 (return_gradients=True면 gradients도 포함)
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
        
        # 잠재 상태 텐서 (RecurrentActorCritic용)
        latents = None
        if self.use_recurrent and len(self.buffer['latents']) > 0:
            latents = torch.FloatTensor(np.array(self.buffer['latents'])).to(self.device)
            # 배치 차원 조정 (squeeze if needed)
            if latents.dim() == 3 and latents.shape[1] == 1:
                latents = latents.squeeze(1)
        
        # 리턴 및 어드밴티지 계산
        if self.use_monte_carlo:
            # Monte Carlo: 실제 에피소드 리턴 사용 (부트스트래핑 없음)
            advantages, returns = self.compute_mc_returns(
                rewards, dones, old_values.cpu().numpy()
            )
        else:
            # GAE: TD 기반 어드밴티지 추정
            next_value = 0
            advantages, returns = self.compute_gae(
                rewards, old_values.cpu().numpy(), dones, next_value
            )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 정규화 전 어드밴티지 통계 저장
        adv_mean_before = advantages.mean().item()
        adv_std_before = advantages.std().item()
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_ratio_mean = 0
        total_ratio_std = 0
        
        # 그래디언트 저장용 (return_gradients=True일 때)
        gradients = None
        
        # 여러 에폭 동안 업데이트
        for epoch in range(epochs):
            # 현재 정책으로 평가
            if self.use_recurrent:
                log_probs, values, entropy = self.actor_critic.evaluate(
                    states, actions, latent=latents, n_cycles=self.n_cycles
                )
            else:
                log_probs, values, entropy = self.actor_critic.evaluate(states, actions)
            
            # 정책 비율
            ratio = torch.exp(log_probs - old_log_probs)
            
            # 비율 통계 (디버깅용)
            total_ratio_mean += ratio.mean().item()
            total_ratio_std += ratio.std().item()
            
            # PPO 클리핑 손실
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 가치 함수 손실 (클리핑 적용하여 안정화)
            # Value loss 클리핑: 큰 오차를 제한하여 불안정성 방지
            value_pred = values.squeeze(-1)
            value_loss_clipped = torch.clamp(
                (value_pred - returns).pow(2),
                max=100.0  # 최대 손실 제한
            ).mean()
            value_loss = value_loss_clipped
            
            # 엔트로피 손실 (표준 PPO: 고정 계수 사용)
            entropy_loss = -entropy.mean()
            
            # 총 손실
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            # 그래디언트 클리핑 강화 (불안정성 방지)
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm * 0.5)  # 더 강한 클리핑
            
            # 그래디언트 수집 (return_gradients=True일 때)
            if return_gradients:
                if gradients is None:
                    gradients = {}
                for name, param in self.actor_critic.named_parameters():
                    if param.grad is not None:
                        if name not in gradients:
                            gradients[name] = param.grad.clone()
                        else:
                            gradients[name] += param.grad.clone()
            else:
                # 일반 모드: 가중치 업데이트
                self.optimizer.step()
                # 학습률 스케줄링 (매 에폭마다)
                if self.scheduler is not None:
                    self.scheduler.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # 버퍼 초기화
        self.reset_buffer()
        
        result = {
            'loss': total_loss / epochs,
            'policy_loss': total_policy_loss / epochs,
            'value_loss': total_value_loss / epochs,
            'entropy': total_entropy / epochs,
            'adv_mean': adv_mean_before,
            'adv_std': adv_std_before,
            'ratio_mean': total_ratio_mean / epochs,
            'ratio_std': total_ratio_std / epochs
        }
        
        # 그래디언트 반환 (있는 경우)
        if return_gradients and gradients is not None:
            result['gradients'] = gradients
        
        return result
    
    def save(self, path: str):
        """모델 저장"""
        save_dict = {
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'use_recurrent': self.use_recurrent,
                'n_cycles': self.n_cycles,
                'carry_latent': self.carry_latent,
                'latent_dim': self.latent_dim
            }
        }
        # 스케줄러 상태 저장 (있는 경우)
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """모델 로드 (라즈베리 파이 호환)"""
        try:
            # 라즈베리 파이에서는 CPU로 강제 로드 (메모리 정렬 문제 방지)
            map_location = 'cpu' if not torch.cuda.is_available() else self.device
            
            # 안전한 모델 로드 (weights_only 옵션은 PyTorch 1.13+)
            try:
                checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                # 구버전 PyTorch 호환
                checkpoint = torch.load(path, map_location=map_location)
            
            # 모델을 CPU로 로드한 후 필요시 디바이스로 이동
            self.actor_critic.load_state_dict(checkpoint['actor_critic'])
            self.actor_critic.to(self.device)
            
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # 스케줄러 상태 로드 (있는 경우)
            if 'scheduler' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.use_recurrent = config.get('use_recurrent', True)
                self.n_cycles = config.get('n_cycles', 4)
                self.carry_latent = config.get('carry_latent', True)
            
            print(f"Model loaded from {path}")
            print(f"Device: {self.device}")
        
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    print("=" * 60)
    print("TRM-PPO Agent 테스트")
    print("=" * 60)
    
    # TRM 스타일 RecurrentActorCritic 테스트
    print("\n[1] RecurrentActorCritic (TRM-style) 테스트")
    agent = PPOAgent(
        state_dim=256, 
        action_dim=2,
        latent_dim=256,
        hidden_dim=256,
        n_cycles=4,
        carry_latent=True,
        use_recurrent=True
    )
    
    # 더미 데이터로 테스트
    state = np.random.rand(256).astype(np.float32)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    
    # 첫 번째 액션 (초기 잠재 상태)
    action, log_prob, value, latent = agent.get_action_with_carry(state_tensor)
    print(f"Step 1 - Action: {action.shape}, Log Prob: {log_prob.shape if log_prob is not None else None}, Value: {value.shape}")
    print(f"         Latent: {latent.shape if latent is not None else None}")
    
    # 두 번째 액션 (잠재 상태 carry-over)
    state2 = np.random.rand(256).astype(np.float32)
    state_tensor2 = torch.FloatTensor(state2).unsqueeze(0).to(agent.device)
    action2, log_prob2, value2, latent2 = agent.get_action_with_carry(state_tensor2)
    print(f"Step 2 - Action: {action2.shape}, Log Prob: {log_prob2.shape if log_prob2 is not None else None}, Value: {value2.shape}")
    print(f"         Latent carry-over 활성화: {agent.carry_latent}")
    
    # 에피소드 리셋
    agent.reset_carry()
    print(f"         에피소드 리셋 후 carry: {agent.current_carry}")
    
    # 기존 ActorCritic 테스트 (backward compatibility)
    print("\n[2] 기존 ActorCritic (backward compatibility) 테스트")
    agent_legacy = PPOAgent(
        state_dim=256, 
        action_dim=2,
        use_recurrent=False
    )
    
    action_legacy, log_prob_legacy, value_legacy, _ = agent_legacy.get_action_with_carry(state_tensor)
    print(f"Legacy - Action: {action_legacy.shape}, Value: {value_legacy.shape}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

