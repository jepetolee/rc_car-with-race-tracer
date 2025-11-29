#!/usr/bin/env python3
"""
TRM-DQN Teacher Forcing + Offline Q-learning
"""

import argparse
import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
import torch

from ppo_agent import DQNAgent

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def load_demonstrations(path: str) -> List[dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("demonstrations", data)


class TeacherForcingTrainer:
    def __init__(
        self,
        agent: DQNAgent,
        demonstrations: List[dict],
        device: str = "cpu",
        lr: float = 3e-4,
    ):
        self.agent = agent
        self.device = torch.device(device)
        self.demonstrations = demonstrations
        if lr is not None:
            for group in self.agent.optimizer.param_groups:
                group["lr"] = lr

        (
            self.states,
            self.actions,
            self.next_states,
            self.rewards,
            self.dones,
        ) = self._prepare_data(demonstrations)

    def _prepare_data(self, demos):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

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
            raise ValueError("데모 데이터에 유효한 (state, action) 쌍이 없습니다.")

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(next_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def pretrain(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        save_path: str = "trained_models/pretrained_dqn.pth",
        log_dir: str = "runs",
        verbose: bool = True,
    ):
        writer = None
        if HAS_TENSORBOARD:
            writer = SummaryWriter(
                os.path.join(log_dir, f"teacher_forcing_{datetime.now():%Y%m%d_%H%M%S}")
            )

        num_samples = len(self.states)
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(num_samples)
            epoch_loss = 0
            total = 0
            correct = 0

            for start in range(0, num_samples, batch_size):
                idx = perm[start : start + batch_size]
                batch_states = self.states[idx]
                batch_actions = self.actions[idx]

                loss = self.agent.supervised_step(batch_states, batch_actions)
                epoch_loss += loss * len(idx)
                total += len(idx)

                q_values = self.agent.predict(batch_states)
                preds = np.argmax(q_values, axis=1)
                correct += np.sum(preds == batch_actions)

            avg_loss = epoch_loss / max(total, 1)
            accuracy = correct / max(total, 1)
            if verbose:
                print(
                    f"[Pretrain {epoch}/{epochs}] Loss: {avg_loss:.4f} | Acc: {accuracy*100:.2f}%"
                )
            if writer:
                writer.add_scalar("Pretrain/Loss", avg_loss, epoch)
                writer.add_scalar("Pretrain/Accuracy", accuracy, epoch)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.agent.save(save_path)
        if writer:
            writer.close()
        print(f"사전 학습 모델 저장: {save_path}")

    def offline_q_learning(self, steps: int = 10_000):
        transitions = list(
            zip(
                self.states,
                self.actions,
                self.rewards,
                self.next_states,
                self.dones,
            )
        )
        for state, action, reward, next_state, done in transitions:
            self.agent.store_transition(state, action, reward, next_state, done)

        for step in range(steps):
            info = self.agent.update()
            if info and step % 1000 == 0:
                print(
                    f"[Offline Q] Step {step}/{steps} | Loss: {info['loss']:.4f} | TD: {info['td_error']:.4f}"
                )


def main():
    parser = argparse.ArgumentParser(description="TRM-DQN Teacher Forcing")
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--save-path", type=str, default="trained_models/pretrained_dqn.pth"
    )
    parser.add_argument("--offline-steps", type=int, default=0)
    parser.add_argument("--state-dim", type=int, default=784)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--n-deep-loops", type=int, default=2)
    parser.add_argument("--n-latent-loops", type=int, default=2)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    demos = load_demonstrations(args.demos)
    agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        lr=args.learning_rate,
        device=args.device,
        n_deep_loops=args.n_deep_loops,
        n_latent_loops=args.n_latent_loops,
    )

    trainer = TeacherForcingTrainer(
        agent, demos, device=args.device, lr=args.learning_rate
    )
    trainer.pretrain(
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        log_dir=args.log_dir,
        verbose=args.verbose,
    )

    if args.offline_steps > 0:
        trainer.offline_q_learning(steps=args.offline_steps)


if __name__ == "__main__":
    main()

