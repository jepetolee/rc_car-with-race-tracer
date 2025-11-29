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
from datetime import datetime, timedelta
import sys
import time

from ppo_agent import PPOAgent, LatentCarry
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
        self.use_sequence_mode = True  # 시퀀스 모드: 에피소드 내 시퀀스 유지 및 latent 전달
        
        # 데모 데이터 로드
        print(f"📂 데모 데이터 로드: {demos_path}")
        with open(demos_path, 'rb') as f:
            data = pickle.load(f)
        
        all_demos = data.get('demonstrations', [])
        if len(all_demos) == 0:
            raise ValueError("데모 데이터가 비어있습니다.")
        
        # states나 actions가 없거나 비어있는 에피소드 필터링
        self.demos = []
        filtered_count = 0
        
        for episode in all_demos:
            states = episode.get('states', [])
            actions = episode.get('actions', [])
            
            # states나 actions가 없거나 비어있는 경우 제외
            if not states or not actions or len(states) == 0 or len(actions) == 0:
                filtered_count += 1
                continue
            
            # 길이 맞추기
            if len(states) != len(actions):
                min_len = min(len(states), len(actions))
                states = states[:min_len]
                actions = actions[:min_len]
            
            # 유효한 데이터만 포함
            if len(states) > 0 and len(actions) > 0:
                self.demos.append({
                    'states': states,
                    'actions': actions
                })
            else:
                filtered_count += 1
        
        if len(self.demos) == 0:
            raise ValueError("유효한 데모 데이터가 없습니다 (모든 에피소드가 필터링되었습니다).")
        
        if filtered_count > 0:
            print(f"⚠️  {filtered_count}개 에피소드 필터링됨 (states나 actions가 없거나 비어있음)")
        
        print(f"✅ {len(self.demos)}개 유효한 에피소드 로드 완료")
        
        # 모든 (state, action) 쌍 추출
        # 주의: Imitation Learning이므로 pkl의 'rewards'는 사용하지 않음
        # 리워드는 모델 액션과 전문가 액션을 비교하여 자동 생성됨
        self.demo_states = []
        self.demo_actions = []
        
        for episode in self.demos:
            states = episode.get('states', [])
            actions = episode.get('actions', [])
            # rewards, dones, timestamps는 사용하지 않음 (Imitation Learning)
            
            self.demo_states.extend(states)
            self.demo_actions.extend(actions)
        
        print(f"✅ 총 {len(self.demo_states)}개 (state, action) 쌍")
        
        # 상태 차원 자동 감지
        if len(self.demo_states) > 0:
            # 첫 번째 상태를 확인하여 차원 결정
            first_state = np.array(self.demo_states[0])
            state_dim = first_state.shape[0] if len(first_state.shape) == 1 else first_state.size
            print(f"📐 상태 차원 자동 감지: {state_dim}")
        else:
            raise ValueError("데모 데이터에 상태가 없습니다.")
        
        # 에이전트 생성
        print(f"\n🤖 에이전트 생성...")
        # PPOAgent는 lr_actor, lr_critic 파라미터를 사용 (learning_rate가 아님)
        actor_lr = float(learning_rate)
        critic_lr = float(learning_rate)
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=5,
            discrete_action=True,
            lr_actor=actor_lr,
            lr_critic=critic_lr,
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
        # model_path가 제공되지 않으면 기본값으로 a3c_model_best.pth 사용
        if not model_path:
            default_model = 'a3c_model_best.pth'
            # 프로젝트 루트에서 확인
            if os.path.exists(default_model):
                model_path = default_model
                print(f"📥 기본 모델 자동 감지: {default_model}")
            else:
                print(f"⚠️  기본 모델({default_model})을 찾을 수 없습니다. 랜덤 초기화로 시작합니다.")
                model_path = None
        
        if model_path and os.path.exists(model_path):
            print(f"📥 사전 학습된 모델 로드: {model_path}")
            try:
                self.agent.load(model_path)
                print("✅ 모델 로드 완료")
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {e}")
                print("랜덤 초기화로 시작합니다.")
        elif model_path:
            print(f"⚠️  모델 파일이 존재하지 않습니다: {model_path}")
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
    
    def train_step_sequence(
        self, 
        states: np.ndarray, 
        expert_actions: np.ndarray,
        is_first_batch: bool = False,
        prev_latent: torch.Tensor = None
    ):
        """
        시퀀스 학습 스텝 (이전 latent 전달)
        
        Args:
            states: 상태 배열 [batch_size, 256]
            expert_actions: 전문가 액션 배열 [batch_size]
            is_first_batch: 에피소드의 첫 배치인지
            prev_latent: 이전 배치의 latent (None이면 초기화)
        
        Returns:
            stats: 통계 딕셔너리
            next_latent: 다음 배치로 전달할 latent
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        expert_actions_tensor = torch.LongTensor(expert_actions).to(self.device)
        
        batch_size = states_tensor.shape[0]
        
        # Latent 초기화 또는 이전 latent 사용
        if is_first_batch or prev_latent is None:
            latent = self.agent.actor_critic.init_latent.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            # 이전 배치의 마지막 latent 사용 (배치 크기가 다를 수 있으므로 조정)
            if prev_latent.shape[0] == batch_size:
                latent = prev_latent.clone()
            else:
                # 배치 크기가 다르면 마지막 latent를 복제
                latent = prev_latent[-1:].expand(batch_size, -1).clone()
        
        # 초기 액션 선택 (리워드 계산용)
        # 시퀀스 모드에서는 carry를 사용하여 이전 정보 전달
        # actor_critic.get_action을 직접 호출하여 carry를 전달
        if prev_latent is not None and not is_first_batch:
            # prev_latent의 배치 크기를 현재 배치 크기에 맞춤
            if prev_latent.shape[0] == batch_size:
                carry_latent = prev_latent.clone()
            else:
                # 배치 크기가 다르면 마지막 latent를 expand하여 사용
                carry_latent = prev_latent[-1:].expand(batch_size, -1).clone()
            carry = LatentCarry(latent=carry_latent)
        else:
            carry = None
        
        # actor_critic.get_action 직접 호출 (carry 전달)
        actions, log_probs, values, new_carry = self.agent.actor_critic.get_action(
            states_tensor,
            carry=carry,
            deterministic=False,
            n_cycles=None  # Deep Supervision은 학습 루프에서 처리
        )
        
        # 다음 배치를 위한 latent 업데이트 (초기 액션 선택 후)
        # new_carry.latent의 배치 크기 확인
        if new_carry is not None:
            if new_carry.latent.shape[0] == batch_size:
                latent = new_carry.latent.clone()
            else:
                # 배치 크기가 다르면 마지막 latent를 expand
                latent = new_carry.latent[-1:].expand(batch_size, -1).clone()
        actions_np = actions.cpu().numpy().flatten()
        
        # 리워드 계산
        rewards = np.array([
            self.compute_imitation_reward(pred, expert)
            for pred, expert in zip(actions_np, expert_actions)
        ])
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # Advantage 계산 (detach하여 그래프 분리)
        advantages = (rewards_tensor - values.squeeze()).detach()
        old_log_probs = log_probs.detach()
        
        # 통계 누적용
        total_loss_sum = 0
        total_actor_loss_sum = 0
        total_value_loss_sum = 0
        total_entropy_sum = 0
        
        # TRM 스타일: Step-wise Update (K번 반복)
        # 시퀀스 모드에서는 각 상태마다 latent를 전달
        # 다음 배치로 전달할 latent 초기화 (초기 액션 선택 후의 latent)
        
        if self.agent.use_recurrent and self.agent.deep_supervision:
            # K번의 Supervision Loop
            # 각 step에서 새로운 계산 그래프를 생성하기 위해 latent를 detach
            # latent가 그래프에 연결되어 있을 수 있으므로 detach
            if new_carry is not None:
                current_latent = latent.detach().clone().requires_grad_(False)
            else:
                current_latent = latent.clone().detach().requires_grad_(False)
            
            for step in range(self.agent.n_supervision_steps):
                # 매 step마다 완전히 새로운 forward pass를 위해 모든 텐서를 새로 생성
                # numpy array에서 새로 변환하여 그래프 연결 방지
                states_tensor_fresh = torch.FloatTensor(states).to(self.device)
                expert_actions_tensor_fresh = torch.LongTensor(expert_actions).to(self.device)
                rewards_tensor_fresh = torch.FloatTensor(rewards).to(self.device)
                
                # 1. State Encoding (매 step마다 새로 계산 - 새로운 그래프)
                state_emb = self.agent.actor_critic.encoder(states_tensor_fresh)
                
                # 2. Deep Recursion (One Step of M x N)
                # current_latent는 detach되어 있어서 새로운 그래프를 생성
                next_latent, latent_grad, value, action_output = self.agent.actor_critic.deep_recursion(
                    state_emb, current_latent, self.agent.n_deep_loops, self.agent.n_latent_loops
                )
                
                # 3. Loss Calculation
                value_pred = value.squeeze(-1)
                # rewards_tensor_fresh를 detach하여 value loss에만 사용 (advantage 계산은 초기 액션 기준)
                value_loss = F.mse_loss(value_pred, rewards_tensor_fresh.detach())
                
                # Policy Loss & Entropy
                if self.agent.actor_critic.discrete_action:
                    action_logits = action_output
                    dist = torch.distributions.Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(expert_actions_tensor_fresh.squeeze(-1))
                    entropy = dist.entropy().mean()
                else:
                    action_mean, action_log_std = action_output
                    std = torch.exp(action_log_std)
                    dist = torch.distributions.Normal(action_mean, std)
                    action_inv = torch.atanh(torch.clamp(expert_actions_tensor_fresh, -0.999, 0.999))
                    log_prob = dist.log_prob(action_inv).sum(dim=-1, keepdim=True)
                    log_prob -= torch.log(1 - torch.tanh(action_inv).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
                    new_log_probs = log_prob
                    entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
                
                # Ratio & Surrogate Loss (Deep Supervision에서는 매 step마다 새로운 advantages 계산)
                # 현재 step의 value로 advantages 재계산
                current_advantages = (rewards_tensor_fresh - value_pred).detach()
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * current_advantages
                surr2 = torch.clamp(ratio, 1 - self.agent.clip_epsilon, 1 + self.agent.clip_epsilon) * current_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Total Loss for this step
                loss = actor_loss + self.agent.value_coef * value_loss - self.agent.entropy_coef * entropy
                
                # 4. Backward & Update (각 step마다 독립적인 그래프)
                self.agent.optimizer.zero_grad()
                loss.backward(retain_graph=False)  # retain_graph=False로 명시적으로 설정
                
                torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), self.agent.max_grad_norm)
                self.agent.optimizer.step()
                
                # 통계 누적
                total_loss_sum += loss.item()
                total_actor_loss_sum += actor_loss.item()
                total_value_loss_sum += value_loss.item()
                total_entropy_sum += entropy.item()
                
                # 5. 다음 step을 위한 latent 준비 (detach하여 새로운 그래프 생성)
                # next_latent는 deep_recursion에서 이미 detach되어 반환되지만, 명시적으로 다시 detach
                current_latent = next_latent.detach().clone().requires_grad_(False)
            
            # 다음 배치로 전달할 latent (마지막 상태의 latent, 루프 후 current_latent 사용)
            # current_latent는 마지막 step에서 계산된 latent
            if current_latent is not None and current_latent.shape[0] > 0:
                next_latent = current_latent[-1:].detach().clone()  # 마지막 상태의 latent만 전달
            else:
                next_latent = None
        else:
            # 기존 방식
            new_log_probs, new_values, entropy = self.agent.actor_critic.evaluate(
                states_tensor, expert_actions_tensor.unsqueeze(-1)
            )
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.agent.clip_epsilon, 1.0 + self.agent.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(new_values.squeeze(), rewards_tensor)
            entropy_loss = -entropy.mean()
            
            total_loss = actor_loss + self.agent.value_coef * value_loss + self.agent.entropy_coef * entropy_loss
            
            self.agent.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), self.agent.max_grad_norm)
            self.agent.optimizer.step()
            
            total_loss_sum = total_loss.item()
            total_actor_loss_sum = actor_loss.item()
            total_value_loss_sum = value_loss.item()
            total_entropy_sum = entropy.mean().item()
            
            # 다음 배치를 위한 latent 업데이트 (Deep Supervision이 아닌 경우)
            if new_carry is not None:
                next_latent = new_carry.latent[-1:].detach() if new_carry.latent.shape[0] > 0 else None
        
        match_rate = np.mean(actions_np == expert_actions)
        avg_reward = np.mean(rewards)
        
        stats = {
            'total_loss': total_loss_sum / self.agent.n_supervision_steps if self.agent.use_recurrent else total_loss_sum,
            'actor_loss': total_actor_loss_sum / self.agent.n_supervision_steps if self.agent.use_recurrent else total_actor_loss_sum,
            'value_loss': total_value_loss_sum / self.agent.n_supervision_steps if self.agent.use_recurrent else total_value_loss_sum,
            'entropy': total_entropy_sum / self.agent.n_supervision_steps if self.agent.use_recurrent else total_entropy_sum,
            'match_rate': match_rate,
            'avg_reward': avg_reward
        }
        
        return stats, next_latent
    
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
            # 각 step에서 새로운 계산 그래프를 생성하기 위해 latent를 detach
            current_latent_step = latent.clone().detach().requires_grad_(False)
            
            for step in range(self.agent.n_supervision_steps):
                # 매 step마다 완전히 새로운 forward pass를 위해 텐서를 새로 생성
                states_tensor_step = torch.FloatTensor(states).to(self.device)
                expert_actions_tensor_step = torch.LongTensor(expert_actions).to(self.device)
                rewards_tensor_step = torch.FloatTensor(rewards).to(self.device)
                
                # 1. State Encoding
                state_emb = self.agent.actor_critic.encoder(states_tensor_step)
                
                # 2. Deep Recursion (One Step of M x N)
                next_latent, latent_grad, value, action_output = self.agent.actor_critic.deep_recursion(
                    state_emb, current_latent_step, self.agent.n_deep_loops, self.agent.n_latent_loops
                )
                
                # 3. Loss Calculation for THIS step
                value_pred = value.squeeze(-1)
                value_loss = F.mse_loss(value_pred, rewards_tensor_step)
                
                # Policy Loss & Entropy
                if self.agent.actor_critic.discrete_action:
                    action_logits = action_output
                    dist = torch.distributions.Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(expert_actions_tensor_step.squeeze(-1))
                    entropy = dist.entropy().mean()
                else:
                    action_mean, action_log_std = action_output
                    std = torch.exp(action_log_std)
                    dist = torch.distributions.Normal(action_mean, std)
                    action_inv = torch.atanh(torch.clamp(expert_actions_tensor_step, -0.999, 0.999))
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
                current_latent_step = next_latent.detach().clone().requires_grad_(False)
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
        print(f"에피소드 수: {len(self.demos)}개")
        print(f"학습 에폭: {epochs}")
        print(f"배치 크기: {self.batch_size}")
        print(f"배치당 업데이트: {self.update_epochs}번")
        
        # 전체 작업량 계산
        total_samples = len(self.demo_states)
        total_batches = 0
        if self.use_sequence_mode:
            for episode in self.demos:
                episode_len = len(episode.get('states', []))
                if episode_len > 0:
                    total_batches += (episode_len + self.batch_size - 1) // self.batch_size
        else:
            total_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        total_updates = total_batches * self.update_epochs * epochs
        print(f"총 배치 수: {total_batches}개/에폭")
        print(f"총 업데이트: {total_updates:,}번 ({total_batches} 배치 × {self.update_epochs} 업데이트 × {epochs} 에폭)")
        print(f"{'='*60}\n")
        
        # 시간 측정
        start_time = time.time()
        epoch_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_stats = {
                'total_loss': [],
                'actor_loss': [],
                'value_loss': [],
                'entropy': [],
                'match_rate': [],
                'avg_reward': []
            }
            
            if self.use_sequence_mode:
                # 시퀀스 모드: 에피소드별로 학습, 이전 latent 전달
                # 에피소드 순서 셔플 (에피소드 내 시퀀스는 유지)
                episode_indices = list(range(len(self.demos)))
                np.random.shuffle(episode_indices)
                
                # 에포크 진행 상황 표시
                epoch_progress = (epoch + 1) / epochs * 100
                elapsed_time = time.time() - start_time
                if epoch > 0:
                    avg_epoch_time = elapsed_time / epoch
                    remaining_epochs = epochs - epoch - 1
                    eta_seconds = avg_epoch_time * remaining_epochs
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                else:
                    eta_str = "계산 중..."
                
                print(f"\n[{epoch+1}/{epochs}] 에포크 시작 ({epoch_progress:.1f}%) | 예상 남은 시간: {eta_str}")
                print(f"{'='*60}")
                
                episode_count = 0
                batch_count = 0
                update_count = 0
                
                for ep_idx in episode_indices:
                    episode = self.demos[ep_idx]
                    states = np.array(episode.get('states', []))
                    actions = np.array(episode.get('actions', []))
                    
                    if len(states) == 0 or len(actions) == 0:
                        continue
                    
                    # 에피소드 내 시퀀스를 배치로 나누어 학습 (latent 전달)
                    episode_count += 1
                    episode_len = len(states)
                    num_batches_episode = (episode_len + self.batch_size - 1) // self.batch_size
                    
                    prev_latent = None
                    for batch_idx, i in enumerate(range(0, len(states), self.batch_size)):
                        batch_states = states[i:i+self.batch_size]
                        batch_actions = actions[i:i+self.batch_size]
                        
                        if len(batch_states) < 1:  # 최소 1개는 필요
                            continue
                        
                        batch_count += 1
                        
                        # 여러 번 업데이트
                        for update_iter in range(self.update_epochs):
                            is_first = (i == 0 and update_iter == 0)
                            stats, prev_latent = self.train_step_sequence(
                                batch_states, 
                                batch_actions,
                                is_first_batch=is_first,
                                prev_latent=prev_latent if not is_first else None
                            )
                            
                            update_count += 1
                            
                            for key, value in stats.items():
                                epoch_stats[key].append(value)
                        
                        # 배치 진행 상황 출력 (에피소드당 첫 배치와 마지막 배치, 또는 5개 배치마다)
                        should_print = (batch_idx == 0 or 
                                       batch_idx == num_batches_episode - 1 or 
                                       (batch_idx + 1) % 5 == 0)
                        
                        if should_print and epoch_stats.get('match_rate'):
                            current_match_rate = np.mean(epoch_stats['match_rate'])
                            current_loss = np.mean(epoch_stats['total_loss']) if epoch_stats.get('total_loss') else 0
                            print(f"  [에피소드 {episode_count}/{len(episode_indices)}] "
                                  f"배치 {batch_idx+1}/{num_batches_episode} "
                                  f"| 업데이트: {update_count:,} | "
                                  f"Match: {current_match_rate:.1%} | "
                                  f"Loss: {current_loss:.4f}", end='\r', flush=True)
                
                # 모든 에피소드 처리 완료 후 줄바꿈
                if episode_count == len(episode_indices):
                    print()  # 줄바꿈
            else:
                # 기존 모드: 셔플된 독립 샘플 학습
                states_array = np.array(self.demo_states)  # [N, 256]
                actions_array = np.array(self.demo_actions)  # [N]
                
                # 데이터 셔플
                indices = np.arange(len(states_array))
                np.random.shuffle(indices)
                shuffled_states = states_array[indices]
                shuffled_actions = actions_array[indices]
                
                # 에포크 진행 상황 표시 (기존 모드)
                epoch_progress = (epoch + 1) / epochs * 100
                elapsed_time = time.time() - start_time
                if epoch > 0:
                    avg_epoch_time = elapsed_time / epoch
                    remaining_epochs = epochs - epoch - 1
                    eta_seconds = avg_epoch_time * remaining_epochs
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                else:
                    eta_str = "계산 중..."
                
                print(f"\n[{epoch+1}/{epochs}] 에포크 시작 ({epoch_progress:.1f}%) | 예상 남은 시간: {eta_str}")
                print(f"{'='*60}")
                
                num_batches = (len(shuffled_states) + self.batch_size - 1) // self.batch_size
                batch_count = 0
                update_count = 0
                
                # 배치별 학습
                for batch_idx, i in enumerate(range(0, len(shuffled_states), self.batch_size)):
                    batch_states = shuffled_states[i:i+self.batch_size]
                    batch_actions = shuffled_actions[i:i+self.batch_size]
                    
                    if len(batch_states) < self.batch_size:
                        continue
                    
                    batch_count += 1
                    
                    # 여러 번 업데이트
                    for _ in range(self.update_epochs):
                        stats = self.train_step(batch_states, batch_actions)
                        update_count += 1
                        
                        for key, value in stats.items():
                            epoch_stats[key].append(value)
                    
                    # 배치 진행 상황 출력 (5개 배치마다 또는 마지막 배치)
                    if ((batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1) and epoch_stats.get('match_rate'):
                        current_match_rate = np.mean(epoch_stats['match_rate'])
                        current_loss = np.mean(epoch_stats['total_loss']) if epoch_stats.get('total_loss') else 0
                        print(f"  배치 {batch_idx+1}/{num_batches} | "
                              f"업데이트: {update_count:,} | "
                              f"Match: {current_match_rate:.1%} | "
                              f"Loss: {current_loss:.4f}", end='\r', flush=True)
                
                print()  # 줄바꿈
            
            # 에폭 통계
            epoch_time = time.time() - epoch_start_time
            epoch_start_time = time.time()
            
            if verbose:
                avg_loss = np.mean(epoch_stats['total_loss']) if epoch_stats['total_loss'] else 0
                avg_actor_loss = np.mean(epoch_stats['actor_loss']) if epoch_stats['actor_loss'] else 0
                avg_value_loss = np.mean(epoch_stats['value_loss']) if epoch_stats['value_loss'] else 0
                avg_match_rate = np.mean(epoch_stats['match_rate']) if epoch_stats['match_rate'] else 0
                avg_reward = np.mean(epoch_stats['avg_reward']) if epoch_stats['avg_reward'] else 0
                avg_entropy = np.mean(epoch_stats['entropy']) if epoch_stats['entropy'] else 0
                
                # 에포크별 통계 출력
                print(f"\n[에포크 {epoch+1}/{epochs} 완료] ({epoch_time:.1f}초)")
                print(f"  📊 통계:")
                print(f"    - Match Rate: {avg_match_rate:.2%} (목표: 100%)")
                print(f"    - Avg Reward: {avg_reward:.4f}")
                print(f"    - Total Loss: {avg_loss:.4f}")
                print(f"    - Actor Loss: {avg_actor_loss:.4f}")
                print(f"    - Value Loss: {avg_value_loss:.4f}")
                print(f"    - Entropy: {avg_entropy:.4f}")
                print(f"  📈 업데이트: {len(epoch_stats['total_loss']):,}번")
                
                # 전체 진행 상황
                total_progress = ((epoch + 1) / epochs) * 100
                elapsed_total = time.time() - start_time
                if epoch > 0:
                    avg_epoch_time = elapsed_total / (epoch + 1)
                    remaining_epochs = epochs - epoch - 1
                    eta_seconds = avg_epoch_time * remaining_epochs
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    total_eta_str = str(timedelta(seconds=int(elapsed_total + eta_seconds)))
                    
                    print(f"  ⏱️  시간: {str(timedelta(seconds=int(elapsed_total)))} / "
                          f"예상 총 시간: {total_eta_str} (남은: {eta_str})")
                    print(f"  📍 진행률: {total_progress:.1f}% ({'█' * int(total_progress / 2)}{'░' * (50 - int(total_progress / 2))})")
                print()
        
        # 모델 저장
        self.agent.save(save_path)
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("✅ 학습 완료!")
        print(f"{'='*60}")
        print(f"총 학습 시간: {str(timedelta(seconds=int(total_time)))}")
        print(f"평균 에포크 시간: {total_time/epochs:.1f}초")
        print(f"총 업데이트 횟수: {total_updates:,}번")
        print(f"모델 저장 경로: {save_path}")
        print(f"{'='*60}")
        
        # 최종 평가
        print("\n📊 최종 평가 중...")
        final_match_rate = self.evaluate()
        print(f"\n🎯 최종 일치율: {final_match_rate:.2%}")
        print(f"{'='*60}\n")
    
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
            # RecurrentActorCritic.get_action은 4개 값을 반환: (action, log_prob, value, new_carry)
            actions, _, _, _ = self.agent.actor_critic.get_action(states_tensor)
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

