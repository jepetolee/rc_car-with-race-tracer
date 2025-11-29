#!/usr/bin/env python3
"""
A3C (Asynchronous Advantage Actor-Critic) 학습 스크립트
기존 PPOAgent를 재사용하면서 병렬 환경 워커로 학습 효율성 향상
GPU 사용률 최대화 및 학습 속도 개선
"""

import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import os
import sys
import warnings
from collections import deque
from datetime import datetime

# 불필요한 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='pygame')
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

from car_racing_env import CarRacingEnvWrapper
from ppo_agent import PPOAgent

# TensorBoard 지원
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def worker(worker_id, global_agent, args, global_step, global_episode, 
           global_rewards, best_avg_reward, lock, device):
    """
    A3C 워커 프로세스
    기존 PPOAgent를 사용하여 각 워커가 독립적으로 환경과 상호작용하며 학습
    """
    # 워커 프로세스에서도 경고 억제
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='pygame')
    warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
    
    # 로컬 에이전트 생성 (글로벌과 동일한 구조, 단 스케줄러는 사용 안 함)
    # 로컬 에이전트는 그래디언트만 계산하므로 스케줄러가 필요 없음
    local_agent = PPOAgent(
        state_dim=args.state_dim,
        action_dim=5,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_cycles=args.n_cycles,
        carry_latent=args.carry_latent,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=device,
        discrete_action=True,
        num_discrete_actions=5,
        use_recurrent=args.use_recurrent,
        use_monte_carlo=args.use_mc,
        total_steps=args.total_steps,
        lr_schedule='none'  # 로컬 에이전트는 스케줄러 사용 안 함
    )
    
    # 글로벌 에이전트의 가중치 복사
    local_agent.actor_critic.load_state_dict(global_agent.actor_critic.state_dict())
    
    # 환경 생성
    env = CarRacingEnvWrapper(
        max_steps=args.max_episode_steps,
        use_extended_actions=True,
        use_discrete_actions=True
    )
    
    episode_count = 0
    step_count = 0
    
    try:
        while step_count < args.total_steps // args.num_workers:
            # 에피소드 시작
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                state, _ = reset_result
            else:
                state = reset_result
            
            # 잠재 상태 초기화
            if args.use_recurrent:
                local_agent.reset_carry()
            
            episode_reward = 0
            episode_length = 0
            
            # 에피소드 실행
            while episode_length < args.max_episode_steps:
                # 상태 정규화
                state_normalized = state.astype(np.float32) / 255.0
                state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(device)
                
                # 액션 선택 (기존 PPOAgent 메소드 사용)
                if args.use_recurrent:
                    action, log_prob, value, latent_np = local_agent.get_action_with_carry(state_tensor)
                else:
                    action, log_prob, value = local_agent.actor_critic.get_action(state_tensor)
                    latent_np = None
                
                action_np = action.squeeze(0).cpu().detach().numpy().item()
                log_prob_np = log_prob.squeeze(0).cpu().item() if log_prob is not None else 0.0
                value_np = value.squeeze(0).cpu().item()
                
                # 환경 스텝
                next_state, reward, done, info = env.step(action_np)
                
                # 버퍼에 저장 (기존 PPOAgent 메소드 사용)
                is_terminal = done or (episode_length + 1) >= args.max_episode_steps
                done_for_buffer = done if not args.mc_update_on_done else is_terminal
                
                if args.use_recurrent:
                    local_agent.store_transition(
                        state_normalized.copy(),
                        action_np,
                        reward,
                        done_for_buffer,
                        log_prob_np,
                        value_np,
                        latent=latent_np
                    )
                else:
                    local_agent.store_transition(
                        state_normalized.copy(),
                        action_np,
                        reward,
                        done_for_buffer,
                        log_prob_np,
                        value_np
                    )
                
                episode_reward += reward
                episode_length += 1
                step_count += 1
                state = next_state
                
                # 글로벌 스텝 업데이트 (매 스텝마다)
                with lock:
                    global_step.value += 1
                
                # 주기적 업데이트 (A3C 스타일: 작은 배치로 자주 업데이트)
                should_update = False
                if args.mc_update_on_done:
                    should_update = is_terminal and len(local_agent.buffer['states']) > 0
                else:
                    should_update = len(local_agent.buffer['states']) >= args.update_frequency
                
                if should_update:
                    # 업데이트 전 버퍼 크기 저장 (로깅용)
                    buffer_size_before = len(local_agent.buffer['states'])
                    
                    # 진행률 계산 (엔트로피 스케줄링용)
                    progress = min(global_step.value / args.total_steps, 1.0) if args.total_steps > 0 else 0.0
                    
                    # TRM Step-wise Update: 각 Epoch마다 동기화, Epoch 내에서는 K번 연속 수행
                    # 이렇게 하면 TRM의 점진적 개선 효과를 유지하면서도 안정성 확보
                    total_loss_sum = 0
                    total_policy_loss_sum = 0
                    total_value_loss_sum = 0
                    total_entropy_sum = 0
                    
                    # 모든 Epoch의 그래디언트를 누적
                    accumulated_gradients = None
                    
                    for epoch in range(args.update_epochs):
                        # 각 Epoch 시작 시에만 메인 모델과 동기화
                        with lock:
                            local_agent.actor_critic.load_state_dict(
                                global_agent.actor_critic.state_dict()
                            )
                        
                        # Epoch 내에서 K번의 Supervision Step을 연속으로 수행
                        # (TRM의 점진적 개선 효과 유지)
                        loss_info = local_agent.update(
                            epochs=1,  # 한 Epoch = K번의 Step
                            progress=progress, 
                            return_gradients=True,
                            supervision_step_only=False  # 전체 K번 수행
                        )
                        
                        # 그래디언트 누적
                        if 'gradients' in loss_info and loss_info['gradients']:
                            gradients = loss_info['gradients']
                            
                            if accumulated_gradients is None:
                                accumulated_gradients = {}
                                for name, grad in gradients.items():
                                    accumulated_gradients[name] = grad.clone()
                            else:
                                # 그래디언트 누적
                                for name, grad in gradients.items():
                                    if name in accumulated_gradients:
                                        accumulated_gradients[name] += grad.clone()
                                    else:
                                        accumulated_gradients[name] = grad.clone()
                        
                        # 통계 누적
                        if loss_info:
                            total_loss_sum += loss_info.get('loss', 0)
                            total_policy_loss_sum += loss_info.get('policy_loss', 0)
                            total_value_loss_sum += loss_info.get('value_loss', 0)
                            total_entropy_sum += loss_info.get('entropy', 0)
                    
                    # 모든 Epoch의 그래디언트를 한 번에 적용 (Lock 경합 감소)
                    with lock:
                        if accumulated_gradients is not None:
                            # 글로벌 네트워크에 그래디언트 적용
                            global_agent.optimizer.zero_grad()
                            
                            # 누적된 그래디언트 설정
                            for name, param in global_agent.actor_critic.named_parameters():
                                if name in accumulated_gradients:
                                    param.grad = accumulated_gradients[name].clone()
                            
                            # 그래디언트 클리핑
                            torch.nn.utils.clip_grad_norm_(
                                global_agent.actor_critic.parameters(), 
                                global_agent.max_grad_norm
                            )
                            
                            # 글로벌 가중치 업데이트
                            global_agent.optimizer.step()
                            
                            # 최종 동기화 (다음 업데이트를 위해)
                            local_agent.actor_critic.load_state_dict(
                                global_agent.actor_critic.state_dict()
                            )
                    
                    # 업데이트 정보 출력 (워커 0만, 평균값)
                    if worker_id == 0:
                        n_steps = args.update_epochs * local_agent.n_supervision_steps
                        current_step = global_step.value
                        print(f"[Update] Step {current_step}: "
                              f"Loss={total_loss_sum/args.update_epochs:.4f}, "
                              f"π={total_policy_loss_sum/args.update_epochs:.4f}, "
                              f"V={total_value_loss_sum/args.update_epochs:.4f}, "
                              f"H={total_entropy_sum/args.update_epochs:.3f}, "
                              f"Buffer={buffer_size_before}", flush=True)
                
                if done or episode_length >= args.max_episode_steps:
                    break
            
            # 에피소드 종료
            episode_count += 1
            
            # 글로벌 통계 업데이트
            with lock:
                global_episode.value += 1
                # maxlen=100 구현 (list 사용)
                global_rewards.append(episode_reward)
                if len(global_rewards) > 100:
                    global_rewards.pop(0)  # 첫 번째 요소 제거
                
                # 에피소드 정보 출력 및 best model 저장 (워커 0만, 매 에피소드)
                if worker_id == 0:
                    avg_reward = np.mean(list(global_rewards)) if global_rewards else episode_reward
                    progress = (global_step.value / args.total_steps * 100) if args.total_steps > 0 else 0
                    
                    # Best model 저장 (평균 리워드가 개선되었을 때)
                    if avg_reward > best_avg_reward.value:
                        best_avg_reward.value = avg_reward
                        best_model_path = args.save_path.replace('.pth', '_best.pth')
                        global_agent.save(best_model_path)
                        print(f"🏆 새로운 최고 기록! Avg Reward: {avg_reward:.2f} → Best Model 저장: {best_model_path}", flush=True)
                    
                    print(f"[Ep {global_episode.value}] "
                          f"R={episode_reward:.2f} (Avg100={avg_reward:.2f}, Best={best_avg_reward.value:.2f}), "
                          f"Len={episode_length}, "
                          f"Steps={global_step.value:,}/{args.total_steps:,} ({progress:.1f}%)", flush=True)
    
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='A3C 강화학습 훈련 (PPOAgent 호환)')
    
    # 환경 파라미터
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--state-dim', type=int, default=784)
    
    # 학습 파라미터
    parser.add_argument('--total-steps', type=int, default=1000000)
    parser.add_argument('--update-frequency', type=int, default=20)  # A3C는 작은 배치
    parser.add_argument('--update-epochs', type=int, default=1)  # A3C는 1 에폭
    parser.add_argument('--num-workers', type=int, default=4, help='병렬 워커 수')
    
    # 네트워크 파라미터 (PPOAgent와 동일)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--lr-schedule', type=str, default='cosine', 
                        choices=['cosine', 'linear', 'none'],
                        help='학습률 스케줄링 방식: cosine (코사인 감소), linear (선형 감소), none (없음)')
    
    # TRM-PPO 파라미터 (PPOAgent와 동일)
    parser.add_argument('--use-recurrent', action='store_true', default=True)
    parser.add_argument('--n-cycles', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--carry-latent', action='store_true', default=True)
    parser.add_argument('--use-mc', action='store_true', default=False)
    parser.add_argument('--mc-update-on-done', action='store_true', default=False)
    
    # 저장
    parser.add_argument('--save-path', type=str, default='a3c_model.pth')
    parser.add_argument('--save-frequency', type=int, default=10000)
    
    # 디바이스
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("A3C (Asynchronous Advantage Actor-Critic) 학습")
    print("  -> 기존 PPOAgent 구조 재사용")
    print("=" * 60)
    print(f"디바이스: {device}")
    print(f"워커 수: {args.num_workers}")
    print(f"상태 차원: {args.state_dim} ({int(args.state_dim**0.5)}x{int(args.state_dim**0.5)})")
    print(f"총 스텝: {args.total_steps}")
    if args.use_recurrent:
        print(f"TRM-PPO 모드: n_cycles={args.n_cycles}, latent_dim={args.latent_dim}")
    if args.use_mc:
        print(f"Monte Carlo 모드 활성화")
    print("=" * 60)
    
    # 글로벌 에이전트 생성 (PPOAgent 사용)
    global_agent = PPOAgent(
        state_dim=args.state_dim,
        action_dim=5,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_cycles=args.n_cycles,
        carry_latent=args.carry_latent,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=device,
        discrete_action=True,
        num_discrete_actions=5,
        use_recurrent=args.use_recurrent,
        use_monte_carlo=args.use_mc,
        total_steps=args.total_steps,
        lr_schedule='none'  # A3C에서는 스케줄링 사용 안 함 (그래디언트 기반 업데이트)
    )
    
    # 멀티프로세싱 공유를 위해 네트워크 공유 메모리 설정
    global_agent.actor_critic.share_memory()
    
    # 공유 변수
    global_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)
    best_avg_reward = mp.Value('f', float('-inf'))  # 최고 평균 리워드 추적
    manager = mp.Manager()
    global_rewards = manager.list()  # deque 대신 list 사용
    lock = mp.Lock()
    
    # 워커 프로세스 시작
    processes = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker,
            args=(worker_id, global_agent, args,
                  global_step, global_episode, global_rewards, best_avg_reward, lock, device)
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)  # 순차 시작
    
    # 메인 프로세스: 주기적 저장 및 통계 출력
    try:
        last_save_step = 0
        last_stat_time = time.time()
        last_stat_step = 0
        
        print("\n🚀 학습 시작! 워커들이 경험을 수집하고 있습니다...\n", flush=True)
        
        while global_step.value < args.total_steps:
            time.sleep(5)  # 5초마다 체크
            
            current_time = time.time()
            elapsed_time = current_time - last_stat_time
            
            # 30초마다 통계 출력
            if elapsed_time >= 30:
                with lock:
                    current_step = global_step.value
                    current_episode = global_episode.value
                    step_diff = current_step - last_stat_step
                    steps_per_sec = step_diff / elapsed_time if elapsed_time > 0 else 0
                    
                    avg_reward = np.mean(list(global_rewards)) if global_rewards else 0.0
                    best_reward = best_avg_reward.value
                    progress = (current_step / args.total_steps * 100) if args.total_steps > 0 else 0
                    remaining_steps = args.total_steps - current_step
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    eta_hours = eta_seconds / 3600
                    
                    print(f"\n{'='*60}")
                    print(f"📊 학습 진행 상황")
                    print(f"{'='*60}")
                    print(f"총 스텝: {current_step:,} / {args.total_steps:,} ({progress:.1f}%)")
                    print(f"총 에피소드: {current_episode:,}")
                    print(f"평균 리워드 (100ep): {avg_reward:.2f}")
                    print(f"최고 평균 리워드: {best_reward:.2f}")
                    print(f"스텝/초: {steps_per_sec:.1f}")
                    print(f"예상 남은 시간: {eta_hours:.1f}시간")
                    print(f"{'='*60}\n", flush=True)
                
                last_stat_time = current_time
                last_stat_step = current_step
            
            # 주기적 모델 저장
            if global_step.value - last_save_step >= args.save_frequency:
                # 모델 저장 (PPOAgent와 동일한 형식)
                global_agent.save(args.save_path)
                print(f"💾 모델 저장: {args.save_path} (Step: {global_step.value:,})", flush=True)
                last_save_step = global_step.value
    
    except KeyboardInterrupt:
        print("\n\n⚠️  학습 중단됨 (Ctrl+C)", flush=True)
    
    finally:
        # 워커 종료
        print("\n워커 프로세스 종료 중...", flush=True)
        for p in processes:
            p.terminate()
            p.join()
        
        # 최종 통계
        with lock:
            final_step = global_step.value
            final_episode = global_episode.value
            final_avg_reward = np.mean(list(global_rewards)) if global_rewards else 0.0
        
        print(f"\n{'='*60}")
        print(f"✅ 학습 완료")
        print(f"{'='*60}")
        print(f"총 스텝: {final_step:,}")
        print(f"총 에피소드: {final_episode:,}")
        print(f"평균 리워드 (100ep): {final_avg_reward:.2f}")
        print(f"{'='*60}\n")
        
        # 최종 저장
        global_agent.save(args.save_path)
        print(f"💾 최종 모델 저장: {args.save_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method('spawn')  # Windows/Linux 호환
    main()

