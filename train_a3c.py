#!/usr/bin/env python3
"""
TRM 기반 Multi-Worker DQN 학습 스크립트 (A3C 스타일)
여러 worker가 병렬로 환경을 실행하며 각자 학습하고, 주기적으로 global agent와 동기화
"""

import argparse
import os
import time
from collections import deque
from datetime import datetime
from multiprocessing import Process, Queue, Manager
import copy

import numpy as np
import torch
import torch.multiprocessing as mp

from car_racing_env import CarRacingEnvWrapper
from ppo_agent import DQNAgent

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def linear_epsilon(step, start, end, decay_steps):
    if decay_steps <= 0:
        return end
    return max(end, start - (start - end) * (step / decay_steps))


def worker_process(
    worker_id: int,
    args,
    global_step_counter,
    episode_rewards_queue: Queue,
    global_weights_queue: Queue,
    worker_weights_queue: Queue,  # Worker가 학습한 가중치를 전송하는 Queue
    lock,
    device: str = "cpu",
):
    """각 워커가 독립적으로 환경을 실행하며 학습"""
    print(f"[Worker {worker_id}] 시작 (Device: {device})...", flush=True)
    print(f"[Worker {worker_id}] 환경 초기화 중...", flush=True)
    env = CarRacingEnvWrapper(
        max_steps=args.max_episode_steps,
        use_extended_actions=True,
        use_discrete_actions=True,
    )
    print(f"[Worker {worker_id}] 환경 초기화 완료", flush=True)

    # Worker별 로컬 DQN
    print(f"[Worker {worker_id}] DQN Agent 생성 중 (Device: {device})...", flush=True)
    if device.startswith("cuda"):
        # 여러 프로세스가 같은 GPU를 공유하므로 메모리 할당을 조절
        torch.cuda.set_device(0)  # 첫 번째 GPU 사용
    local_agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gamma=args.gamma,
        lr=args.learning_rate,
        device=device,
        buffer_size=args.replay_buffer // args.num_workers,  # Worker당 버퍼 크기
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        n_deep_loops=args.n_deep_loops,
        n_latent_loops=args.n_latent_loops,
        max_grad_norm=args.max_grad_norm,
    )
    print(f"[Worker {worker_id}] DQN Agent 생성 완료. 학습 시작!", flush=True)

    episode = 0
    local_steps = 0
    sync_counter = 0

    while True:
        with lock:
            total_steps = global_step_counter.value
        if total_steps >= args.max_steps:
            break

        # 에피소드 시작 전에 항상 Global weights 다운로드 (최신 가중치 사용)
        if not global_weights_queue.empty():
            try:
                global_weights = global_weights_queue.get_nowait()
                local_agent.q_network.load_state_dict(global_weights)
                local_agent.target_network.load_state_dict(global_weights)
            except:
                pass  # Queue가 비어있거나 오류 시 이전 가중치 유지

        # 에피소드 시작
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result

        state = state.astype(np.float32).reshape(-1) / 255.0
        done = False
        episode_reward = 0
        step = 0

        while not done and step < args.max_episode_steps:
            # Epsilon 계산
            epsilon = linear_epsilon(
                total_steps, args.eps_start, args.eps_end, args.eps_decay
            )

            # 액션 선택
            action = local_agent.select_action(state, epsilon=epsilon)

            # 환경 스텝
            next_state, reward, done, _ = env.step(action)
            next_state_norm = next_state.astype(np.float32).reshape(-1) / 255.0

            # 로컬 replay buffer에 저장
            local_agent.store_transition(state.copy(), action, reward, next_state_norm.copy(), done)

            # 로컬 학습
            if len(local_agent.replay_buffer) >= args.batch_size:
                local_agent.update()

            state = next_state_norm
            episode_reward += reward
            step += 1
            local_steps += 1
            sync_counter += 1

            # Global step counter 업데이트
            with lock:
                global_step_counter.value += 1

        episode += 1

        # 에피소드 리워드 전송
        episode_rewards_queue.put((worker_id, episode, episode_reward, local_steps))
        
        # 에피소드 종료 시 학습한 가중치를 coordinator에게 전송 (A3C 스타일)
        try:
            weights = copy.deepcopy(local_agent.q_network.state_dict())
            worker_weights_queue.put((worker_id, episode, weights), block=False)
        except:
            pass  # Queue가 가득 차면 스킵 (non-blocking)

        # 모든 에피소드 로그 출력 (첫 10개, 그 후 10개마다)
        if episode <= 10 or episode % 10 == 0:
            print(
                f"[Worker {worker_id}] Episode {episode} | "
                f"Reward: {episode_reward:.2f} | Local Steps: {local_steps} | "
                f"Buffer: {len(local_agent.replay_buffer)}/{local_agent.replay_buffer.capacity}",
                flush=True
            )

    env.close()


def coordinator_process(
    args,
    global_step_counter,
    episode_rewards_queue: Queue,
    global_weights_queue: Queue,
    worker_weights_queue: Queue,  # Worker가 학습한 가중치를 받는 Queue
    lock,
    device: str = "cuda",
):
    """Coordinator가 global agent를 관리하고 모든 워커와 동기화"""
    print("[Coordinator] Global Agent 초기화 중...", flush=True)
    # Global agent (모든 워커의 가중치를 집계)
    global_agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gamma=args.gamma,
        lr=args.learning_rate,
        device=device,
        buffer_size=0,  # Coordinator는 학습하지 않음
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        n_deep_loops=args.n_deep_loops,
        n_latent_loops=args.n_latent_loops,
        max_grad_norm=args.max_grad_norm,
    )

    writer = None
    if args.use_tensorboard and HAS_TENSORBOARD:
        log_dir = os.path.join(
            "runs", f"dqn_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")

    recent_rewards = deque(maxlen=100)
    last_sync = 0
    best_avg = float("-inf")
    best_model_path = os.path.join(args.save_dir, "dqn_multi_best.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    print("[Coordinator] 시작됨. 워커들로부터 에피소드 리워드 수집 중...", flush=True)

    while True:
        with lock:
            total_steps = global_step_counter.value
        if total_steps >= args.max_steps:
            break

        # Worker로부터 학습한 가중치 수집
        # 각 워커가 에피소드 종료 시마다 (worker_id, episode, weights)를 전송
        collected_updates = []  # (worker_id, episode, weights) 튜플 리스트
        while not worker_weights_queue.empty():
            try:
                update = worker_weights_queue.get_nowait()
                collected_updates.append(update)
            except:
                break
        
        # Worker 가중치 집계 및 Global Agent 업데이트 (A3C 스타일)
        # 집계 방식: 모든 워커의 가중치를 파라미터별로 평균화
        if collected_updates:
            # 각 워커의 최신 가중치만 사용 (같은 워커가 여러 번 보낸 경우 최신 것만)
            latest_weights_per_worker = {}
            for worker_id, episode, weights in collected_updates:
                if worker_id not in latest_weights_per_worker or episode > latest_weights_per_worker[worker_id][0]:
                    latest_weights_per_worker[worker_id] = (episode, weights)
            
            # A3C 스타일: 모든 워커의 가중치를 평균화
            # 집계 방식: 각 파라미터(weight, bias 등)에 대해 모든 워커의 값을 평균
            # 예: W_global = (W_worker1 + W_worker2 + ... + W_workerN) / N
            if len(latest_weights_per_worker) > 0:
                avg_weights = {}
                for param_name in global_agent.q_network.state_dict().keys():
                    param_tensors = [weights[param_name] for _, weights in latest_weights_per_worker.values()]
                    # 모든 워커의 파라미터 텐서를 스택하고 평균 계산
                    avg_param = torch.stack(param_tensors).mean(dim=0)
                    avg_weights[param_name] = avg_param
                
                # Global agent에 평균화된 가중치 로드
                global_agent.q_network.load_state_dict(avg_weights)
                global_agent.target_network.load_state_dict(avg_weights)
                
                worker_ids = list(latest_weights_per_worker.keys())
                print(
                    f"[Coordinator] Worker 가중치 집계 완료 | "
                    f"Workers: {worker_ids} ({len(worker_ids)}개) | "
                    f"Steps: {total_steps}",
                    flush=True
                )
        
        # 에피소드마다 global weights를 worker들에게 전송 (항상 최신 가중치 유지)
        current_weights = copy.deepcopy(global_agent.q_network.state_dict())
        # Queue 비우고 최신 weights 넣기
        while not global_weights_queue.empty():
            try:
                global_weights_queue.get_nowait()
            except:
                break
        global_weights_queue.put(current_weights)

        # 에피소드 리워드 수집
        collected = 0
        while not episode_rewards_queue.empty() and collected < 100:
            try:
                worker_id, episode, reward, local_steps = episode_rewards_queue.get_nowait()
                recent_rewards.append(reward)
                collected += 1
                
                # 처음 20개 에피소드는 모두 출력, 이후는 10개마다
                total_episodes = len(recent_rewards)
                if total_episodes <= 20 or total_episodes % 10 == 0:
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                    print(
                        f"[Coordinator] Worker {worker_id} Ep{episode} | "
                        f"Reward: {reward:.2f} | Avg(100): {avg_reward:.2f} | "
                        f"Steps: {total_steps}",
                        flush=True
                    )

                # Best model 갱신 (충분한 에피소드가 모였을 때만)
                if len(recent_rewards) >= 10:
                    avg_reward = np.mean(recent_rewards)
                    
                    # TensorBoard 로깅
                    if writer:
                        writer.add_scalar("Eval/EpisodeReward", reward, total_steps)
                        writer.add_scalar("Eval/AvgReward100", avg_reward, total_steps)
                        writer.add_scalar("Train/GlobalSteps", total_steps, total_steps)
                        writer.add_scalar("Eval/BestAvgReward", best_avg, total_steps)
                    
                    # Best model 갱신 (평균 리워드가 개선되면 저장)
                    if avg_reward > best_avg:
                        old_best = best_avg
                        best_avg = avg_reward
                        global_agent.save(best_model_path)
                        print(
                            f"[Coordinator] 🏆 Best Model 갱신! | "
                            f"Avg Reward: {avg_reward:.2f} (이전: {old_best:.2f}) | "
                            f"Steps: {total_steps} | 저장: {best_model_path}",
                            flush=True
                        )

            except:
                break

        time.sleep(0.1)  # CPU 사용률 조절

    # 학습 종료 시 최종 모델 저장 (Best model과 별도)
    final_save_path = os.path.join(
        args.save_dir,
        f"dqn_multi_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
    )
    global_agent.save(final_save_path)
    avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
    print(
        f"[Coordinator] 학습 종료 | "
        f"최종 모델: {final_save_path} | "
        f"Best 모델: {best_model_path} (Avg Reward: {best_avg:.2f}) | "
        f"최종 Avg Reward: {avg_reward:.2f} | "
        f"Steps: {global_step_counter.value}",
        flush=True
    )

    if writer:
        writer.close()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("Multi-Worker TRM-DQN 학습 시작 (A3C 스타일)")
    print(f"디바이스: {device}")
    print(f"워커 수: {args.num_workers}")
    print(f"상태 차원: {args.state_dim}")
    print(f"액션 개수: {args.action_dim}")
    print(f"최대 스텝: {args.max_steps}")
    print(f"동기화 간격: {args.sync_interval} 스텝")
    print("=" * 60)

    # Shared objects
    manager = Manager()
    global_step_counter = manager.Value("i", 0)
    lock = manager.Lock()
    episode_rewards_queue = Queue()
    global_weights_queue = Queue()
    worker_weights_queue = Queue()  # Worker가 학습한 가중치를 전송하는 Queue

    # Processes
    processes = []

    # Coordinator process (global agent 관리) - 학습하지 않으므로 CPU 사용
    coordinator_p = Process(
        target=coordinator_process,
        args=(
            args,
            global_step_counter,
            episode_rewards_queue,
            global_weights_queue,
            worker_weights_queue,  # Worker 가중치 수집용
            lock,
            "cpu",  # Coordinator는 학습하지 않으므로 CPU만 사용
        ),
    )
    coordinator_p.start()
    processes.append(coordinator_p)

    # Worker processes
    for worker_id in range(args.num_workers):
        # 모든 워커가 GPU를 공유하여 사용 (PyTorch는 멀티프로세스 GPU 공유 지원)
        worker_device = device if torch.cuda.is_available() else "cpu"
        worker_p = Process(
            target=worker_process,
            args=(
                worker_id,
                args,
                global_step_counter,
                episode_rewards_queue,
                global_weights_queue,
                worker_weights_queue,  # Worker 가중치 전송용
                lock,
                worker_device,
            ),
        )
        worker_p.start()
        processes.append(worker_p)

    # 모든 프로세스 종료 대기
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n⚠️  학습 중단 중...")
        for p in processes:
            p.terminate()
            p.join()

    print("학습 종료")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Worker TRM-DQN 학습 (A3C 스타일)")
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
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument(
        "--num-workers", type=int, default=4, help="워커 프로세스 수 (A3C 스타일)"
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=1000,
        help="Global agent와 동기화 간격 (스텝)",
    )
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=300_000)
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument(
        "--save-interval-steps", type=int, default=50000, help="저장 간격 (스텝)"
    )
    parser.add_argument("--use-tensorboard", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Windows/Mac 호환성
    args = parse_args()
    train(args)
