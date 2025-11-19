#!/usr/bin/env python3
"""
CarRacing 환경에서 사전학습 스크립트
사전학습된 모델을 실제 RC Car 환경으로 전이
"""

import argparse
from train_ppo import train_ppo, test_agent
from car_racing_env import CarRacingEnvWrapper
from ppo_agent import PPOAgent
import torch

# 실제 하드웨어 환경은 선택적 임포트
try:
    from rc_car_env import RCCarEnv
    HAS_REAL_ENV = True
except ImportError:
    HAS_REAL_ENV = False
    RCCarEnv = None


def main():
    parser = argparse.ArgumentParser(description='CarRacing 사전학습 및 전이 학습')
    
    # 단계 선택
    parser.add_argument('--stage', choices=['pretrain', 'transfer', 'test'], default='pretrain',
                        help='실행 단계: pretrain(사전학습), transfer(전이학습), test(테스트)')
    
    # 사전학습 파라미터
    parser.add_argument('--pretrain-steps', type=int, default=500000,
                        help='사전학습 스텝 수 (기본: 500000)')
    parser.add_argument('--pretrain-save-path', type=str, default='ppo_pretrained.pth',
                        help='사전학습 모델 저장 경로')
    
    # 전이학습 파라미터
    parser.add_argument('--transfer-steps', type=int, default=100000,
                        help='전이학습 스텝 수 (기본: 100000)')
    parser.add_argument('--transfer-save-path', type=str, default='ppo_transferred.pth',
                        help='전이학습 모델 저장 경로')
    
    # 공통 파라미터
    parser.add_argument('--max-episode-steps', type=int, default=1000,
                        help='에피소드 최대 스텝 수')
    parser.add_argument('--update-frequency', type=int, default=2048,
                        help='업데이트 주기')
    parser.add_argument('--update-epochs', type=int, default=10,
                        help='업데이트 에폭 수')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='히든 레이어 차원')
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                        help='Actor 학습률')
    parser.add_argument('--lr-critic', type=float, default=3e-4,
                        help='Critic 학습률')
    parser.add_argument('--device', type=str, default=None,
                        help='디바이스 (cuda/cpu)')
    parser.add_argument('--render', action='store_true',
                        help='렌더링 활성화')
    parser.add_argument('--use-discrete-actions', action='store_true', default=True,
                        help='이산 액션 공간 사용 (기본값, 5개 액션: 정지, 전진직진, 전진좌회전, 전진우회전, 후진)')
    parser.add_argument('--use-continuous-actions', dest='use_discrete_actions', action='store_false',
                        help='연속 액션 공간 사용 (이산 액션 비활성화)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 에이전트 생성
    # 이산 액션 모드인지 확인 (기본값: True)
    use_discrete = args.use_discrete_actions
    agent = PPOAgent(
        state_dim=256,
        action_dim=2 if not use_discrete else 5,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        device=device,
        discrete_action=use_discrete,
        num_discrete_actions=5
    )
    
    if args.stage == 'pretrain':
        print("=" * 60)
        print("Stage 1: CarRacing 환경에서 사전학습")
        print("=" * 60)
        
        # CarRacing 환경 생성
        env = CarRacingEnvWrapper(
            max_steps=args.max_episode_steps,
            use_extended_actions=True,
            use_discrete_actions=args.use_discrete_actions
        )
        
        print("CarRacing 환경에서 사전학습 시작...")
        print("이 단계는 실제 하드웨어 없이 빠르게 학습합니다.")
        print("=" * 60)
        
        # 사전학습
        train_ppo(
            env=env,
            agent=agent,
            total_steps=args.pretrain_steps,
            max_episode_steps=args.max_episode_steps,
            update_frequency=args.update_frequency,
            update_epochs=args.update_epochs,
            save_frequency=50000,
            save_path=args.pretrain_save_path
        )
        
        env.close()
        print("\n사전학습 완료! 모델 저장:", args.pretrain_save_path)
        print("다음 단계: --stage transfer 로 전이학습 실행")
        
    elif args.stage == 'transfer':
        print("=" * 60)
        print("Stage 2: 실제 RC Car 환경으로 전이학습")
        print("=" * 60)
        
        # 사전학습 모델 로드
        try:
            agent.load(args.pretrain_save_path)
            print(f"사전학습 모델 로드 완료: {args.pretrain_save_path}")
        except:
            print("⚠️  경고: 사전학습 모델을 찾을 수 없습니다.")
            print("⚠️  새로 학습을 시작합니다. --stage pretrain 을 먼저 실행하세요.")
        
        # 실제 하드웨어 환경 확인
        if not HAS_REAL_ENV:
            raise ImportError(
                "실제 하드웨어 환경을 사용할 수 없습니다.\n"
                "picamera 모듈이 설치되지 않았거나 라즈베리 파이 환경이 아닙니다.\n"
                "전이학습은 라즈베리 파이 환경에서만 가능합니다.\n"
                "일반 PC에서는 Stage 1 (사전학습)만 실행하세요."
            )
        
        # 실제 RC Car 환경 생성
        print("\n⚠️  실제 하드웨어 환경 사용")
        print("⚠️  RC Car가 준비되어 있는지 확인하세요!")
        print("=" * 60)
        
        env = RCCarEnv(
            max_steps=args.max_episode_steps,
            use_extended_actions=True,
            use_discrete_actions=args.use_discrete_actions
        )
        
        # 전이학습 (더 작은 학습률 권장)
        agent.optimizer = torch.optim.Adam(
            agent.actor_critic.parameters(),
            lr=args.lr_actor * 0.1  # 10배 작은 학습률
        )
        
        print("전이학습 시작 (작은 학습률 사용)...")
        
        train_ppo(
            env=env,
            agent=agent,
            total_steps=args.transfer_steps,
            max_episode_steps=args.max_episode_steps,
            update_frequency=args.update_frequency,
            update_epochs=args.update_epochs,
            save_frequency=10000,
            save_path=args.transfer_save_path
        )
        
        env.close()
        print("\n전이학습 완료! 모델 저장:", args.transfer_save_path)
        
    elif args.stage == 'test':
        print("=" * 60)
        print("Stage 3: 학습된 모델 테스트")
        print("=" * 60)
        
        # 모델 로드
        model_path = args.transfer_save_path
        try:
            agent.load(model_path)
            print(f"모델 로드 완료: {model_path}")
        except:
            print(f"⚠️  {model_path}를 찾을 수 없습니다.")
            print(f"⚠️  사전학습 모델 사용 시도: {args.pretrain_save_path}")
            try:
                agent.load(args.pretrain_save_path)
                model_path = args.pretrain_save_path
            except:
                print("❌ 모델을 찾을 수 없습니다!")
                return
        
        # 실제 하드웨어 환경 확인
        if not HAS_REAL_ENV:
            print("⚠️  실제 하드웨어 환경을 사용할 수 없습니다.")
            print("⚠️  CarRacing 환경에서 테스트를 진행합니다.")
            print("=" * 60)
            
            # CarRacing 환경으로 테스트
            env = CarRacingEnvWrapper(
                max_steps=args.max_episode_steps,
                use_extended_actions=True,
                use_discrete_actions=args.use_discrete_actions
            )
            print("\nCarRacing 환경에서 테스트 시작...")
        else:
            # 실제 RC Car 환경 생성
            env = RCCarEnv(
                max_steps=args.max_episode_steps,
                use_extended_actions=True,
                use_discrete_actions=args.use_discrete_actions
            )
            print("\n실제 RC Car 환경에서 테스트 시작...")
        
        test_agent(env, agent, num_episodes=5, max_steps=args.max_episode_steps)
        
        env.close()


if __name__ == "__main__":
    main()

