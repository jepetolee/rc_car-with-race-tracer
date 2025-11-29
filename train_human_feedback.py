#!/usr/bin/env python3
"""
사람 평가 기반 TRM-DQN 미세 조정
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

from rc_car_env import RCCarEnv
from rc_car_controller import RCCarController
from ppo_agent import DQNAgent

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class HumanFeedbackTrainer:
    """사람 평가를 받아 DQN을 보정"""

    def __init__(
        self,
        agent: DQNAgent,
        port: str = "/dev/ttyACM0",
        max_steps: int = 500,
        action_delay: float = 0.1,
        eval_epsilon: float = 0.0,
        updates_per_episode: int = 500,
        device: Optional[str] = None,
    ):
        self.agent = agent
        self.port = port
        self.max_steps = max_steps
        self.action_delay = action_delay
        self.eval_epsilon = eval_epsilon
        self.updates_per_episode = updates_per_episode
        self.device = torch.device(device or agent.device)

        self.env = RCCarEnv(
            max_steps=max_steps,
            use_extended_actions=True,
            use_discrete_actions=True,
        )
        self.controller = RCCarController(port=port, delay=action_delay)

    @staticmethod
    def _normalize_state(state: np.ndarray) -> np.ndarray:
        arr = state.astype(np.float32).reshape(-1)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    def run_episode(self, verbose: bool = True) -> Dict[str, List]:
        reset = self.env.reset()
        state = reset[0] if isinstance(reset, tuple) else reset
        episode = {
            "states": [],
            "actions": [],
            "next_states": [],
            "dones": [],
        }

        if verbose:
            print("\n" + "=" * 60)
            print("모델 주행 평가 중...")
            print("=" * 60)

        try:
            for step in range(self.max_steps):
                state_vec = self._normalize_state(state)
                action = self.agent.select_action(
                    state_vec,
                    epsilon=self.eval_epsilon,
                    deterministic=self.eval_epsilon <= 0.0,
                )

                self.controller.execute_discrete_action(action)
                next_state, _, done, _ = self.env.step(action)
                next_state_vec = self._normalize_state(next_state)

                episode["states"].append(state_vec)
                episode["actions"].append(action)
                episode["next_states"].append(next_state_vec)
                episode["dones"].append(float(done))

                if verbose and (step + 1) % 50 == 0:
                    print(f"[Step {step+1:4d}] Action: {action}")

                time.sleep(self.action_delay)

                state = next_state
                if done:
                    break
        except KeyboardInterrupt:
            print("\n⚠️  사용자에 의해 중단되었습니다.")
        finally:
            self.controller.stop()

        return episode

    def get_human_feedback(self) -> Optional[float]:
        print("\n" + "=" * 60)
        print("주행 평가")
        print("=" * 60)
        print("평가 점수를 입력하세요 (0.0 ~ 1.0)")
        print("  0.0: 매우 나쁨 | 0.5: 보통 | 1.0: 매우 좋음")
        print("=" * 60)
        while True:
            try:
                score = float(input("점수 (0.0-1.0): "))
                if 0.0 <= score <= 1.0:
                    return score
                print("⚠️  0.0과 1.0 사이의 값을 입력하세요.")
            except ValueError:
                print("⚠️  숫자를 입력하세요.")
            except KeyboardInterrupt:
                print("\n⚠️  평가 취소")
                return None

    def train_with_feedback(
        self,
        num_episodes: int = 5,
        save_path: str = "trained_models/hf_dqn.pth",
        log_dir: str = "runs",
    ):
        writer = None
        if HAS_TENSORBOARD:
            writer = SummaryWriter(
                os.path.join(log_dir, f"human_feedback_{datetime.now():%Y%m%d_%H%M%S}")
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for episode in range(1, num_episodes + 1):
            data = self.run_episode(verbose=True)
            score = self.get_human_feedback()
            if score is None:
                print("⚠️  평가가 제공되지 않아 업데이트를 건너뜁니다.")
                continue

            transitions = len(data["states"])
            for idx in range(transitions):
                self.agent.store_transition(
                    data["states"][idx],
                    data["actions"][idx],
                    score,
                    data["next_states"][idx],
                    data["dones"][idx],
                )

            avg_loss = 0.0
            updates = 0
            for _ in range(self.updates_per_episode):
                info = self.agent.update()
                if info:
                    avg_loss += info["loss"]
                    updates += 1
            avg_loss = avg_loss / max(updates, 1)

            print(
                f"[Feedback {episode}/{num_episodes}] Score: {score:.2f} | Buffer: {len(self.agent.replay_buffer)} | Loss: {avg_loss:.4f}"
            )
            self.agent.save(save_path)
            if writer:
                writer.add_scalar("Feedback/Score", score, episode)
                writer.add_scalar("Feedback/Loss", avg_loss, episode)

        if writer:
            writer.close()
        print(f"모델 저장 완료: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="TRM-DQN Human Feedback")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="trained_models/hf_dqn.pth")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--action-delay", type=float, default=0.1)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--updates-per-episode", type=int, default=500)
    parser.add_argument("--state-dim", type=int, default=784)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-deep-loops", type=int, default=2)
    parser.add_argument("--n-latent-loops", type=int, default=2)
    parser.add_argument("--log-dir", type=str, default="runs")
    args = parser.parse_args()

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
    if args.model and os.path.exists(args.model):
        agent.load(args.model, strict=False)

    trainer = HumanFeedbackTrainer(
        agent=agent,
        port=args.port,
        max_steps=args.max_steps,
        action_delay=args.action_delay,
        eval_epsilon=args.eval_epsilon,
        updates_per_episode=args.updates_per_episode,
        device=args.device,
    )
    trainer.train_with_feedback(
        num_episodes=args.episodes,
        save_path=args.save_path,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()

