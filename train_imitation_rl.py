#!/usr/bin/env python3
"""
TRM-DQN 기반 Imitation RL (오프라인 Q-learning)
"""

import argparse
import os
import pickle
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch

from ppo_agent import DQNAgent


def load_demonstrations(path: str) -> List[dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("demonstrations", data)


def flatten_episodes(demos: List[dict]) -> Tuple[np.ndarray, ...]:
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for episode in demos:
        ep_states = episode.get("states", [])
        ep_actions = episode.get("actions", [])
        ep_rewards = episode.get("rewards", [])
        ep_dones = episode.get("dones", [])

        if len(ep_states) == 0 or len(ep_actions) == 0:
            continue

        for idx in range(len(ep_actions)):
            state = ep_states[idx]
            next_state = ep_states[idx + 1] if idx + 1 < len(ep_states) else state

            state = state.astype(np.float32).reshape(-1)
            next_state = next_state.astype(np.float32).reshape(-1)
            if state.max() > 1.0:
                state = state / 255.0
            if next_state.max() > 1.0:
                next_state = next_state / 255.0

            states.append(state)
            next_states.append(next_state)
            actions.append(int(ep_actions[idx]))
            rewards.append(float(ep_rewards[idx] if idx < len(ep_rewards) else 0.0))
            dones.append(float(ep_dones[idx] if idx < len(ep_dones) else 0.0))

    if len(states) == 0:
        raise ValueError("데모 데이터에 유효한 트랜지션이 없습니다.")

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(dones, dtype=np.float32),
    )


class ImitationRLTrainer:
    def __init__(
        self,
        demos_path: str,
        model_path: str = None,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        batch_size: int = 128,
    ):
        self.device = device
        self.batch_size = batch_size
        self.demos = load_demonstrations(demos_path)
        (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
        ) = flatten_episodes(self.demos)

        self.agent = DQNAgent(
            state_dim=self.states.shape[1],
            action_dim=int(np.max(self.actions)) + 1,
            lr=learning_rate,
            device=device,
            batch_size=batch_size,
        )
        if model_path and os.path.exists(model_path):
            print(f"📥 사전 학습 모델 로드: {model_path}")
            self.agent.load(model_path, strict=False)

    def _populate_buffer(self):
        for (
            state,
            action,
            reward,
            next_state,
            done,
        ) in zip(self.states, self.actions, self.rewards, self.next_states, self.dones):
            self.agent.store_transition(state, action, reward, next_state, done)

    def train(self, epochs: int = 50, updates_per_epoch: int = 1000):
        self._populate_buffer()
        print("=" * 60)
        print("Imitation RL via Offline Q-learning (TRM-DQN)")
        print(f"총 샘플 수: {len(self.states)}")
        print(f"Epoch: {epochs}, epoch당 업데이트: {updates_per_epoch}")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            valid = 0
            for _ in range(updates_per_epoch):
                info = self.agent.update()
                if info:
                    epoch_loss += info["loss"]
                    valid += 1
            avg_loss = epoch_loss / max(valid, 1)
            match = self.evaluate(num_samples=512)
            print(
                f"[{epoch}/{epochs}] Loss: {avg_loss:.4f} | Match: {match*100:.2f}%"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("trained_models", f"imitation_dqn_{timestamp}.pth")
        os.makedirs("trained_models", exist_ok=True)
        self.agent.save(save_path)
        return save_path

    def evaluate(self, num_samples: int = 512) -> float:
        idx = np.random.choice(len(self.states), min(num_samples, len(self.states)), replace=False)
        sample_states = self.states[idx]
        sample_actions = self.actions[idx]
        q_values = self.agent.predict(sample_states)
        preds = np.argmax(q_values, axis=1)
        return float(np.mean(preds == sample_actions))


def main():
    parser = argparse.ArgumentParser(description="TRM-DQN Imitation RL")
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--updates-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    trainer = ImitationRLTrainer(
        demos_path=args.demos,
        model_path=args.model,
        device=args.device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    trainer.train(epochs=args.epochs, updates_per_epoch=args.updates_per_epoch)


if __name__ == "__main__":
    main()

