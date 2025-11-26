#!/usr/bin/env python3
"""
PPO 강화학습 훈련 스크립트
RC Car 환경에서 PPO 에이전트를 학습
"""

import argparse
import numpy as np
import torch
import time
import sys
import os
from datetime import datetime
from collections import deque
from rc_car_sim_env import RCCarSimEnv
from car_racing_env import CarRacingEnvWrapper
from ppo_agent import PPOAgent

# TensorBoard 지원 (선택적)
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️  TensorBoard 미설치 - pip install tensorboard 로 설치하면 실시간 모니터링 가능")

# 실제 하드웨어 환경은 선택적 임포트
try:
    from rc_car_env import RCCarEnv
    HAS_REAL_ENV = True
except ImportError:
    HAS_REAL_ENV = False
    RCCarEnv = None


def train_ppo(
    env,
    agent,
    total_steps=100000,
    max_episode_steps=1000,
    update_frequency=2048,
    update_epochs=10,
    save_frequency=10000,
    save_path='ppo_model.pth',
    log_frequency=100,
    use_tensorboard=True,
    log_dir='runs',
    mc_update_on_done=False
):
    """
    PPO 학습 함수 (TRM-PPO 지원)
    
    Args:
        env: 환경 객체
        agent: PPO 에이전트 (TRM-PPO 또는 기존 PPO)
        total_steps: 총 학습 스텝 수
        max_episode_steps: 에피소드 최대 스텝 수
        update_frequency: 업데이트 주기 (버퍼 크기)
        update_epochs: 업데이트 에폭 수
        save_frequency: 모델 저장 주기
        save_path: 모델 저장 경로
        log_frequency: 로그 출력 주기
        use_tensorboard: TensorBoard 사용 여부
        log_dir: TensorBoard 로그 디렉토리
    """
    step_count = 0
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    
    # 이동 평균을 위한 deque
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    # TRM-PPO 모드 확인
    use_recurrent = getattr(agent, 'use_recurrent', False)
    
    # TensorBoard 설정
    writer = None
    if use_tensorboard and HAS_TENSORBOARD:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_str = f"TRM-PPO_n{agent.n_cycles}" if use_recurrent else "PPO"
        log_path = os.path.join(log_dir, f"{mode_str}_{timestamp}")
        writer = SummaryWriter(log_path)
        print(f"📊 TensorBoard 로그: {log_path}")
        print(f"   실행: tensorboard --logdir={log_dir}")
    
    print("=" * 60)
    print("PPO 강화학습 시작")
    if use_recurrent:
        print("  -> TRM-PPO 모드 (재귀 추론 + 잠재 상태 carry-over)")
        print(f"  -> n_cycles: {agent.n_cycles}, carry_latent: {agent.carry_latent}")
    print("=" * 60)
    print(f"총 학습 스텝: {total_steps}")
    print(f"업데이트 주기: {update_frequency} 스텝")
    print(f"에피소드 최대 길이: {max_episode_steps}")
    print("=" * 60)
    
    # Gymnasium vs Gym API 차이 처리
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        state, _ = reset_result  # Gymnasium
    else:
        state = reset_result  # Gym
    
    # TRM-PPO: 잠재 상태 초기화
    if use_recurrent:
        agent.reset_carry()
    
    episode_reward = 0
    episode_length = 0
    
    try:
        while step_count < total_steps:
            # 상태 정규화 [0, 255] -> [0, 1] (중요!)
            state_normalized = state.astype(np.float32) / 255.0
            
            # 액션 선택 (TRM-PPO 또는 기존 PPO)
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(agent.device)
            
            if use_recurrent:
                # TRM-PPO: get_action_with_carry 사용
                action, log_prob, value, latent_np = agent.get_action_with_carry(state_tensor)
            else:
                # 기존 PPO: actor_critic.get_action 사용
                action, log_prob, value = agent.actor_critic.get_action(state_tensor)
                latent_np = None
            
            # 이산 액션과 연속 액션 처리
            if agent.actor_critic.discrete_action:
                action_np = action.squeeze(0).cpu().detach().numpy().item()  # 정수로 변환
            else:
                action_np = action.squeeze(0).cpu().detach().numpy()
            log_prob_np = log_prob.squeeze(0).cpu().item() if log_prob is not None else 0.0
            value_np = value.squeeze(0).cpu().item()
            
            # 환경 스텝
            next_state, reward, done, info = env.step(action_np)
            
            # 에피소드 종료 조건 확인 (버퍼 저장 전에 계산)
            is_terminal = done or (episode_length + 1) >= max_episode_steps
            
            # 버퍼에 저장 (정규화된 상태 사용, TRM-PPO는 잠재 상태도 저장)
            # MC 모드에서는 truncation도 done으로 처리하여 리턴 계산 정확성 보장
            done_for_buffer = done if not mc_update_on_done else is_terminal
            
            if use_recurrent:
                agent.store_transition(
                    state_normalized.copy(),
                    action_np,
                    reward,
                    done_for_buffer,
                    log_prob_np,
                    value_np,
                    latent=latent_np
                )
            else:
                agent.store_transition(
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
            
            # 에피소드 종료 조건 (환경 done OR 최대 스텝 도달)
            episode_done = done or episode_length >= max_episode_steps
            
            # 에피소드 종료 또는 최대 스텝 도달
            if episode_done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                recent_rewards.append(episode_reward)
                recent_lengths.append(episode_length)
                
                # TensorBoard 로깅
                if writer:
                    writer.add_scalar('Episode/Reward', episode_reward, episode_count)
                    writer.add_scalar('Episode/Length', episode_length, episode_count)
                    if len(recent_rewards) >= 10:
                        writer.add_scalar('Episode/AvgReward_100', np.mean(recent_rewards), episode_count)
                        writer.add_scalar('Episode/AvgLength_100', np.mean(recent_lengths), episode_count)
                
                # 최고 성능 모델 저장
                if len(recent_rewards) >= 10:
                    current_avg = np.mean(recent_rewards)
                    if current_avg > best_avg_reward:
                        best_avg_reward = current_avg
                        best_path = save_path.replace('.pth', '_best.pth')
                        agent.save(best_path)
                        print(f"🏆 새로운 최고 기록! Avg Reward: {best_avg_reward:.2f}")
                
                # 에피소드 정보 출력 (매 에피소드)
                avg_reward = np.mean(recent_rewards) if recent_rewards else episode_reward
                avg_length = np.mean(recent_lengths) if recent_lengths else episode_length
                print(f"[Ep {episode_count}] "
                      f"R: {episode_reward:.2f} (Avg100: {avg_reward:.2f}), "
                      f"Len: {episode_length}, Steps: {step_count}")
                
                # 환경 리셋
                reset_result = env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    state, _ = reset_result  # Gymnasium
                else:
                    state = reset_result  # Gym
                
                # TRM-PPO: 에피소드 종료 시 잠재 상태 리셋
                if use_recurrent:
                    agent.reset_carry()
                
                episode_reward = 0
                episode_length = 0
            
            # 업데이트 조건 결정
            should_update = False
            if mc_update_on_done:
                # Monte Carlo 스타일: 에피소드 완료 시에만 업데이트
                # episode_done = 환경 done OR 최대 스텝 도달 (truncation 포함)
                should_update = episode_done and len(agent.buffer['states']) > 0
            else:
                # 일반 PPO: 버퍼가 충분히 찼을 때 업데이트
                should_update = len(agent.buffer['states']) >= update_frequency
            
            if should_update:
                loss_info = agent.update(epochs=update_epochs)
                
                if loss_info:
                    mc_tag = "[MC] " if mc_update_on_done else ""
                    print(f"{mc_tag}[Step {step_count}] "
                          f"Loss: {loss_info['loss']:.4f}, "
                          f"π: {loss_info['policy_loss']:.4f}, "
                          f"V: {loss_info['value_loss']:.4f}, "
                          f"H: {loss_info['entropy']:.3f}, "
                          f"Adv: {loss_info.get('adv_mean', 0):.2f}±{loss_info.get('adv_std', 0):.2f}, "
                          f"Ratio: {loss_info.get('ratio_mean', 1):.3f}")
                    
                    # TensorBoard 로깅
                    if writer:
                        writer.add_scalar('Train/Loss', loss_info['loss'], step_count)
                        writer.add_scalar('Train/PolicyLoss', loss_info['policy_loss'], step_count)
                        writer.add_scalar('Train/ValueLoss', loss_info['value_loss'], step_count)
                        writer.add_scalar('Train/Entropy', loss_info['entropy'], step_count)
                        writer.flush()
            
            # 정기 저장
            if step_count % save_frequency == 0 and step_count > 0:
                agent.save(save_path)
                print(f"Model saved at step {step_count}")
    
    except KeyboardInterrupt:
        print("\n학습 중단됨")
    
    finally:
        # TensorBoard 종료
        if writer:
            writer.close()
        
        # 최종 저장
        agent.save(save_path)
        env.close()
        
        # 최종 통계
        if episode_rewards:
            print("\n" + "=" * 60)
            print("학습 완료")
            print("=" * 60)
            print(f"총 에피소드: {episode_count}")
            print(f"총 스텝: {step_count}")
            print(f"평균 리워드: {np.mean(episode_rewards):.2f}")
            print(f"최고 리워드: {np.max(episode_rewards):.2f}")
            print(f"최고 평균(100ep) 리워드: {best_avg_reward:.2f}")
            print(f"평균 에피소드 길이: {np.mean(episode_lengths):.1f}")
            print(f"모델 저장 위치: {save_path}")
            if use_recurrent:
                print(f"모드: TRM-PPO (n_cycles={agent.n_cycles})")
            if writer:
                print(f"📊 TensorBoard: tensorboard --logdir={log_dir}")
            print("=" * 60)


def test_agent(env, agent, num_episodes=5, max_steps=1000):
    """
    학습된 에이전트 테스트 (TRM-PPO 지원)
    
    Args:
        env: 환경 객체
        agent: 학습된 PPO 에이전트 (TRM-PPO 또는 기존 PPO)
        num_episodes: 테스트 에피소드 수
        max_steps: 최대 스텝 수
    """
    # TRM-PPO 모드 확인
    use_recurrent = getattr(agent, 'use_recurrent', False)
    
    print("=" * 60)
    print("에이전트 테스트 시작")
    if use_recurrent:
        print(f"  -> TRM-PPO 모드 (n_cycles={agent.n_cycles})")
    print("=" * 60)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Gymnasium vs Gym API 차이 처리
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, _ = reset_result  # Gymnasium
        else:
            state = reset_result  # Gym
        
        # TRM-PPO: 에피소드 시작 시 잠재 상태 리셋
        if use_recurrent:
            agent.reset_carry()
        
        episode_reward = 0
        
        for step in range(max_steps):
            # 상태 정규화
            state_normalized = state.astype(np.float32) / 255.0
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(agent.device)
            
            if use_recurrent:
                # TRM-PPO: get_action_with_carry 사용 (deterministic)
                action, _, _, _ = agent.get_action_with_carry(state_tensor, deterministic=True)
            else:
                # 기존 PPO
                action, _, _ = agent.actor_critic.get_action(state_tensor, deterministic=True)
            
            # 이산 액션과 연속 액션 처리
            if agent.actor_critic.discrete_action:
                action_np = action.squeeze(0).cpu().detach().numpy().item()  # 정수로 변환
            else:
                action_np = action.squeeze(0).cpu().detach().numpy()
            next_state, reward, done, info = env.step(action_np)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step + 1}")
    
    print("=" * 60)
    print(f"평균 리워드: {np.mean(episode_rewards):.2f}")
    print("=" * 60)
    
    env.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='PPO 강화학습 훈련')
    
    # 환경 파라미터
    parser.add_argument('--max-episode-steps', type=int, default=1000,
                        help='에피소드 최대 스텝 수 (기본: 1000)')
    
    # 학습 파라미터
    parser.add_argument('--total-steps', type=int, default=100000,
                        help='총 학습 스텝 수 (기본: 100000)')
    parser.add_argument('--update-frequency', type=int, default=2048,
                        help='업데이트 주기 (기본: 2048)')
    parser.add_argument('--update-epochs', type=int, default=10,
                        help='업데이트 에폭 수 (기본: 10)')
    
    # 네트워크 파라미터
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='히든 레이어 차원 (기본: 256)')
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                        help='Actor 학습률 (기본: 3e-4)')
    parser.add_argument('--lr-critic', type=float, default=3e-4,
                        help='Critic 학습률 (기본: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='할인율 (기본: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE 람다 (기본: 0.95)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO 클립 범위 (기본: 0.2)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='엔트로피 계수 (기본: 0.01, 낮출수록 exploitation)')
    parser.add_argument('--value-coef', type=float, default=0.5,
                        help='가치 손실 계수 (기본: 0.5)')
    
    # TRM-PPO 파라미터
    parser.add_argument('--use-recurrent', action='store_true', default=True,
                        help='TRM-PPO 모드 사용 (재귀 추론, 기본: True)')
    parser.add_argument('--no-recurrent', dest='use_recurrent', action='store_false',
                        help='기존 PPO 모드 사용 (TRM 비활성화)')
    parser.add_argument('--n-cycles', type=int, default=4,
                        help='TRM-PPO 재귀 추론 반복 횟수 (기본: 4)')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='TRM-PPO 잠재 상태 차원 (기본: 256)')
    parser.add_argument('--carry-latent', action='store_true', default=True,
                        help='에피소드 내 잠재 상태 carry-over (기본: True)')
    parser.add_argument('--no-carry-latent', dest='carry_latent', action='store_false',
                        help='매 스텝 잠재 상태 초기화')
    
    # Monte Carlo 옵션
    parser.add_argument('--use-mc', action='store_true', default=False,
                        help='Monte Carlo 리턴 사용 (GAE 대신 순수 에피소드 리턴)')
    parser.add_argument('--mc-update-on-done', action='store_true', default=False,
                        help='에피소드 종료 시에만 업데이트 (MC 스타일)')
    
    # 저장/로드
    parser.add_argument('--save-path', type=str, default='ppo_model.pth',
                        help='모델 저장 경로 (기본: ppo_model.pth)')
    parser.add_argument('--load-path', type=str, default=None,
                        help='모델 로드 경로 (없으면 새로 학습)')
    parser.add_argument('--save-frequency', type=int, default=10000,
                        help='모델 저장 주기 (기본: 10000)')
    
    # 모니터링
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='TensorBoard 로깅 활성화 (기본: True)')
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                        help='TensorBoard 비활성화')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='TensorBoard 로그 디렉토리 (기본: runs)')
    
    # 모드
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='실행 모드: train(학습) 또는 test(테스트)')
    parser.add_argument('--test-episodes', type=int, default=5,
                        help='테스트 에피소드 수 (기본: 5)')
    
    # 환경 선택
    parser.add_argument('--env-type', choices=['real', 'sim', 'carracing'], default='carracing',
                        help='환경 타입: real(실제 하드웨어-추론전용), sim(시뮬레이션), carracing(Gym CarRacing 사전학습-권장)')
    parser.add_argument('--use-extended-actions', action='store_true', default=True,
                        help='확장된 액션 공간 사용 (전진/후진, 좌회전/우회전) - 연속 액션 모드')
    parser.add_argument('--use-discrete-actions', action='store_true', default=True,
                        help='이산 액션 공간 사용 (기본값, CarRacing: 0-4)')
    parser.add_argument('--use-continuous-actions', dest='use_discrete_actions', action='store_false',
                        help='연속 액션 공간 사용 (이산 액션 비활성화)')
    parser.add_argument('--render', action='store_true',
                        help='환경 렌더링 (시뮬레이션/CarRacing 모드에서만)')
    
    # 디바이스
    parser.add_argument('--device', type=str, default=None,
                        help='디바이스 (cuda/cpu, 기본: 자동 선택)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"사용 디바이스: {device}")
    print(f"환경 타입: {args.env_type}")
    print(f"확장된 액션 공간: {args.use_extended_actions}")
    
    # 환경 생성
    if args.env_type == 'carracing':
        # Gym CarRacing 환경 (사전학습용)
        try:
            env = CarRacingEnvWrapper(
                max_steps=args.max_episode_steps,
                use_extended_actions=args.use_extended_actions,
                use_discrete_actions=args.use_discrete_actions
            )
            print("=" * 60)
            print("Gym CarRacing 환경 사용 - 사전학습 권장")
            print("=" * 60)
            if args.render:
                print("렌더링 모드 활성화 - 학습 속도가 느려질 수 있습니다")
        except ImportError as e:
            print("=" * 60)
            print("❌ CarRacing 환경을 사용할 수 없습니다!")
            print("=" * 60)
            print(str(e))
            print("\n대안: 시뮬레이션 환경 사용")
            print("python train_ppo.py --env-type sim --use-extended-actions")
            print("=" * 60)
            sys.exit(1)
    elif args.env_type == 'sim':
        # 시뮬레이션 환경
        render_mode = 'human' if args.render else None
        env = RCCarSimEnv(
            max_steps=args.max_episode_steps,
            render_mode=render_mode,
            use_extended_actions=args.use_extended_actions
        )
        print("시뮬레이션 환경 사용 - 빠른 학습 가능")
        if args.render:
            print("렌더링 모드 활성화 - 학습 속도가 느려질 수 있습니다")
    else:
        # 실제 하드웨어 환경 (추론 전용)
        if not HAS_REAL_ENV:
            raise ImportError(
                "실제 하드웨어 환경을 사용할 수 없습니다.\n"
                "picamera 모듈이 설치되지 않았거나 라즈베리 파이 환경이 아닙니다.\n"
                "사전학습을 위해 CarRacing 환경을 사용하세요: --env-type carracing\n"
                "또는 시뮬레이션 환경: --env-type sim"
            )
        
        env = RCCarEnv(
            max_steps=args.max_episode_steps,
            use_extended_actions=args.use_extended_actions,
            use_discrete_actions=args.use_discrete_actions
        )
        print("=" * 60)
        print("⚠️  실제 하드웨어 환경 사용")
        print("⚠️  학습 모드에서는 사용하지 마세요!")
        print("⚠️  테스트/추론 전용입니다!")
        print("=" * 60)
    
    # 에이전트 생성
    # 이산 액션 모드인지 확인 (기본값: True)
    use_discrete = args.use_discrete_actions
    
    agent = PPOAgent(
        state_dim=256,
        action_dim=2 if not use_discrete else 5,  # 이산 액션: 5개
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_cycles=args.n_cycles,
        carry_latent=args.carry_latent,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        device=device,
        discrete_action=use_discrete,
        num_discrete_actions=5,
        use_recurrent=args.use_recurrent,
        use_monte_carlo=args.use_mc
    )
    
    # TRM-PPO 모드 출력
    if args.use_recurrent:
        print(f"TRM-PPO 모드: n_cycles={args.n_cycles}, latent_dim={args.latent_dim}, carry_latent={args.carry_latent}")
    
    # Monte Carlo 모드 출력
    if args.use_mc:
        print("=" * 60)
        print("📊 Monte Carlo 모드 활성화")
        print("  -> GAE 대신 순수 에피소드 리턴 사용")
        if args.mc_update_on_done:
            print("  -> 에피소드 종료 시에만 업데이트")
        print("=" * 60)
    
    # 모델 로드 (있는 경우)
    if args.load_path:
        agent.load(args.load_path)
    
    # 학습 또는 테스트
    if args.mode == 'train':
        train_ppo(
            env=env,
            agent=agent,
            total_steps=args.total_steps,
            max_episode_steps=args.max_episode_steps,
            update_frequency=args.update_frequency,
            update_epochs=args.update_epochs,
            save_frequency=args.save_frequency,
            save_path=args.save_path,
            use_tensorboard=args.tensorboard,
            log_dir=args.log_dir,
            mc_update_on_done=args.mc_update_on_done
        )
    elif args.mode == 'test':
        if not args.load_path:
            print("경고: 테스트 모드에서는 모델을 로드해야 합니다. --load-path를 지정하세요.")
        else:
            test_agent(env, agent, num_episodes=args.test_episodes, max_steps=args.max_episode_steps)


if __name__ == "__main__":
    main()

