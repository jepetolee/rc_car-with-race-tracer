#!/usr/bin/env python3
"""
TRM-DQN 학습 (시뮬/CarRacing/실기)
"""

import argparse
import os
from collections import deque
from datetime import datetime

import numpy as np
import torch

from rc_car_sim_env import RCCarSimEnv
from car_racing_env import CarRacingEnvWrapper
from ppo_agent import DQNAgent

try:
    from rc_car_env import RCCarEnv

    HAS_REAL_ENV = True
except ImportError:
    RCCarEnv = None
    HAS_REAL_ENV = False

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def linear_epsilon(step, start, end, decay_steps):
    if decay_steps <= 0:
        return end
    return max(end, start - (start - end) * (step / decay_steps))


def create_env(env_type: str, args):
    if env_type == "carracing":
        return CarRacingEnvWrapper(
            max_steps=args.max_episode_steps,
            use_extended_actions=True,
            use_discrete_actions=True,
        )
    if env_type == "sim":
        return RCCarSimEnv(
            max_steps=args.max_episode_steps,
            use_extended_actions=True,
            use_discrete_actions=True,
        )
    if env_type == "real":
        if not HAS_REAL_ENV:
            raise RuntimeError("rc_car_env를 가져올 수 없습니다. --env-type real 사용 불가")
        return RCCarEnv(
            max_steps=args.max_episode_steps,
            use_extended_actions=True,
            use_discrete_actions=True,
        )
    raise ValueError(f"알 수 없는 env_type: {env_type}")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = create_env(args.env_type, args)

    agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gamma=args.gamma,
        lr=args.learning_rate,
        device=device,
        buffer_size=args.replay_buffer,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        n_deep_loops=args.n_deep_loops,
        n_latent_loops=args.n_latent_loops,
        max_grad_norm=args.max_grad_norm,
    )

    writer = None
    if args.use_tensorboard and HAS_TENSORBOARD:
        log_dir = os.path.join(
            "runs",
            f"dqn_{args.env_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")

    total_steps = 0
    recent_rewards = deque(maxlen=100)

    for episode in range(1, args.max_episodes + 1):
        reset = env.reset()
        state = reset[0] if isinstance(reset, tuple) else reset
        state = state.astype(np.float32).reshape(-1) / 255.0

        done = False
        episode_reward = 0.0
        step = 0

        while not done and step < args.max_episode_steps:
            epsilon = linear_epsilon(
                total_steps, args.eps_start, args.eps_end, args.eps_decay
            )
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state_vec = next_state.astype(np.float32).reshape(-1) / 255.0

            agent.store_transition(state, action, reward, next_state_vec, done)
            info = agent.update()

            state = next_state_vec
            episode_reward += reward
            step += 1
            total_steps += 1

            if writer and info:
                writer.add_scalar("Train/Loss", info["loss"], total_steps)
                writer.add_scalar("Train/TD_Error", info["td_error"], total_steps)
                writer.add_scalar("Train/Epsilon", epsilon, total_steps)

        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)
        print(
            f"[{args.env_type}] Episode {episode}/{args.max_episodes} "
            f"Reward: {episode_reward:.2f} | Avg(100): {avg_reward:.2f} | "
            f"Epsilon: {epsilon:.3f} | Steps: {total_steps}"
        )

        if writer:
            writer.add_scalar("Eval/EpisodeReward", episode_reward, episode)
            writer.add_scalar("Eval/AvgReward100", avg_reward, episode)

        if episode % args.save_interval == 0 or episode == args.max_episodes:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(
                args.save_dir,
                f"dqn_{args.env_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
            )
            agent.save(save_path)

    env.close()
    if writer:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="TRM-DQN 학습 (멀티 환경)")
    parser.add_argument("--env-type", choices=["carracing", "sim", "real"], default="carracing")
    parser.add_argument("--state-dim", type=int, default=784)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--n-deep-loops", type=int, default=2)
    parser.add_argument("--n-latent-loops", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-buffer", type=int, default=200_000)
    parser.add_argument("--target-update-interval", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-episodes", type=int, default=5000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=300_000)
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--use-tensorboard", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

