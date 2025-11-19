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
from rc_car_sim_env import RCCarSimEnv
from car_racing_env import CarRacingEnvWrapper
from ppo_agent import PPOAgent

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
    log_frequency=100
):
    """
    PPO 학습 함수
    
    Args:
        env: 환경 객체
        agent: PPO 에이전트
        total_steps: 총 학습 스텝 수
        max_episode_steps: 에피소드 최대 스텝 수
        update_frequency: 업데이트 주기 (버퍼 크기)
        update_epochs: 업데이트 에폭 수
        save_frequency: 모델 저장 주기
        save_path: 모델 저장 경로
        log_frequency: 로그 출력 주기
    """
    step_count = 0
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    
    print("=" * 60)
    print("PPO 강화학습 시작")
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
    
    episode_reward = 0
    episode_length = 0
    
    try:
        while step_count < total_steps:
            # 액션 선택
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action, log_prob, value = agent.actor_critic.get_action(state_tensor)
            
            # 이산 액션과 연속 액션 처리
            if agent.actor_critic.discrete_action:
                action_np = action.squeeze(0).cpu().detach().numpy().item()  # 정수로 변환
            else:
                action_np = action.squeeze(0).cpu().detach().numpy()
            log_prob_np = log_prob.squeeze(0).cpu().item() if log_prob is not None else 0.0
            value_np = value.squeeze(0).cpu().item()
            
            # 환경 스텝
            next_state, reward, done, info = env.step(action_np)
            
            # 버퍼에 저장
            agent.store_transition(
                state.copy(),
                action_np,
                reward,
                done,
                log_prob_np,
                value_np
            )
            
            episode_reward += reward
            episode_length += 1
            step_count += 1
            state = next_state
            
            # 에피소드 종료 또는 최대 스텝 도달
            if done or episode_length >= max_episode_steps:
                episode_count += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # 에피소드 정보 출력
                if episode_count % log_frequency == 0:
                    avg_reward = np.mean(episode_rewards[-log_frequency:])
                    avg_length = np.mean(episode_lengths[-log_frequency:])
                    print(f"[Episode {episode_count}] "
                          f"Reward: {episode_reward:.2f} (Avg: {avg_reward:.2f}), "
                          f"Length: {episode_length} (Avg: {avg_length:.1f}), "
                          f"Total Steps: {step_count}")
                
                # 환경 리셋
                reset_result = env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    state, _ = reset_result  # Gymnasium
                else:
                    state = reset_result  # Gym
                episode_reward = 0
                episode_length = 0
            
            # 정기 업데이트
            if len(agent.buffer['states']) >= update_frequency:
                loss_info = agent.update(epochs=update_epochs)
                
                if loss_info:
                    print(f"[Step {step_count}] "
                          f"Loss: {loss_info['loss']:.4f}, "
                          f"Policy Loss: {loss_info['policy_loss']:.4f}, "
                          f"Value Loss: {loss_info['value_loss']:.4f}, "
                          f"Entropy: {loss_info['entropy']:.4f}")
            
            # 정기 저장
            if step_count % save_frequency == 0 and step_count > 0:
                agent.save(save_path)
                print(f"Model saved at step {step_count}")
    
    except KeyboardInterrupt:
        print("\n학습 중단됨")
    
    finally:
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
            print(f"평균 에피소드 길이: {np.mean(episode_lengths):.1f}")
            print(f"모델 저장 위치: {save_path}")
            print("=" * 60)


def test_agent(env, agent, num_episodes=5, max_steps=1000):
    """
    학습된 에이전트 테스트
    
    Args:
        env: 환경 객체
        agent: 학습된 PPO 에이전트
        num_episodes: 테스트 에피소드 수
        max_steps: 최대 스텝 수
    """
    print("=" * 60)
    print("에이전트 테스트 시작")
    print("=" * 60)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Gymnasium vs Gym API 차이 처리
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, _ = reset_result  # Gymnasium
        else:
            state = reset_result  # Gym
        
        episode_reward = 0
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
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
    
    # 저장/로드
    parser.add_argument('--save-path', type=str, default='ppo_model.pth',
                        help='모델 저장 경로 (기본: ppo_model.pth)')
    parser.add_argument('--load-path', type=str, default=None,
                        help='모델 로드 경로 (없으면 새로 학습)')
    parser.add_argument('--save-frequency', type=int, default=10000,
                        help='모델 저장 주기 (기본: 10000)')
    
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
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        device=device,
        discrete_action=use_discrete,
        num_discrete_actions=5
    )
    
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
            save_path=args.save_path
        )
    elif args.mode == 'test':
        if not args.load_path:
            print("경고: 테스트 모드에서는 모델을 로드해야 합니다. --load-path를 지정하세요.")
        else:
            test_agent(env, agent, num_episodes=args.test_episodes, max_steps=args.max_episode_steps)


if __name__ == "__main__":
    main()

