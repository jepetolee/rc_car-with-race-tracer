#!/usr/bin/env python3
"""
DQN 기반 TRM Actor (기존 PPO/Actor-Critic 폐기)
"""

import random
from collections import deque, namedtuple
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """경험 리플레이 버퍼"""

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states = np.stack([b.state for b in batch], axis=0)
        actions = np.array([b.action for b in batch])
        rewards = np.array([b.reward for b in batch])
        next_states = np.stack([b.next_state for b in batch], axis=0)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class RecurrentReasoningBlock(nn.Module):
    """TRM 스타일 reasoning 블록"""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.GELU()
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, state_emb: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_emb, latent], dim=-1)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        return latent + self.res_scale * x


class TRMQNetwork(nn.Module):
    """시각 임베딩 → TRM reasoning → Q-values"""

    def __init__(
        self,
        state_dim: int = 784,
        action_dim: int = 5,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        n_deep_loops: int = 2,
        n_latent_loops: int = 2,
    ):
        super().__init__()
        self.n_deep_loops = n_deep_loops
        self.n_latent_loops = n_latent_loops

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        self.reasoning_block = RecurrentReasoningBlock(latent_dim, hidden_dim)
        self.q_head = nn.Linear(latent_dim, action_dim)

    def latent_recursion(self, state_emb: torch.Tensor, latent: torch.Tensor):
        for _ in range(self.n_latent_loops):
            latent = self.reasoning_block(state_emb, latent)
        return latent

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state_emb = self.encoder(state)
        latent = state_emb.clone()

        if self.n_deep_loops > 1:
            with torch.no_grad():
                for _ in range(self.n_deep_loops - 1):
                    latent = self.latent_recursion(state_emb, latent)

        latent = self.latent_recursion(state_emb, latent)
        q_values = self.q_head(latent)
        return q_values


class DQNAgent:
    """
    DQN + TRM Actor
    - epsilon-greedy 정책
    - Target network & Replay buffer
    - Teacher Forcing / Imitation을 위한 supervised step 제공
    """

    def __init__(
        self,
        state_dim: int = 784,
        action_dim: int = 5,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        gamma: float = 0.99,
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        buffer_size: int = 200_000,
        batch_size: int = 64,
        target_update_interval: int = 1000,
        n_deep_loops: int = 2,
        n_latent_loops: int = 2,
        max_grad_norm: float = 1.0,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.update_counter = 0
        self.max_grad_norm = max_grad_norm

        self.q_network = TRMQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_deep_loops=n_deep_loops,
            n_latent_loops=n_latent_loops,
        ).to(self.device)
        self.target_network = TRMQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_deep_loops=n_deep_loops,
            n_latent_loops=n_latent_loops,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def select_action(
        self, state: np.ndarray, epsilon: float = 0.1, deterministic: bool = False
    ) -> int:
        if not deterministic and random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        return self.update_from_batch(states, actions, rewards, next_states, dones)

    def update_from_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        """외부에서 준비된 배치로 학습 (멀티 워커용)"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": loss.item(),
            "td_error": (target_q - current_q).abs().mean().item(),
        }

    def supervised_step(self, states: np.ndarray, actions: np.ndarray) -> float:
        """Teacher Forcing / Imitation 용 cross-entropy 학습"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        q_values = self.q_network(states)
        loss = F.cross_entropy(q_values, actions)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.update_counter += 1
        if self.update_counter % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return loss.item()

    def predict(self, states: np.ndarray) -> np.ndarray:
        self.q_network.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(states).to(self.device)
            q_values = self.q_network(tensor)
        self.q_network.train()
        return q_values.cpu().numpy()

    def act_greedy(self, state: np.ndarray) -> int:
        return self.select_action(state, epsilon=0.0, deterministic=True)

    def save(self, path: str):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str, strict: bool = True):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"], strict=strict)
        if "target_network" in checkpoint:
            self.target_network.load_state_dict(
                checkpoint["target_network"], strict=strict
            )
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Model loaded from {path}")

