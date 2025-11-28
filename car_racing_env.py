#!/usr/bin/env python3
"""
Gymnasium CarRacing 환경 래퍼
사전학습을 위한 CarRacing-v3 환경 통합
Gymnasium 사용 (Gym의 후속 버전)
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    HAS_GYMNASIUM = False
import cv2


class CarRacingEnvWrapper(gym.Env):
    """
    Gymnasium/Gym CarRacing 환경을 RC Car 환경과 호환되도록 래핑
    이미지를 16x16 grayscale로 변환하여 동일한 상태 공간 제공
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, max_steps=1000, use_extended_actions=True, use_discrete_actions=True):
        """
        Args:
            max_steps: 최대 스텝 수
            use_extended_actions: 확장된 액션 공간 사용 여부 (연속 액션)
            use_discrete_actions: 이산 액션 공간 사용 여부
                - True: CarRacing의 이산 액션 (0-4) 사용
                - False: 연속 액션 사용
        """
        super(CarRacingEnvWrapper, self).__init__()
        
        self.use_discrete_actions = use_discrete_actions
        
        # Gymnasium/Gym CarRacing 환경 생성
        try:
            # CarRacing-v3 시도 (Gymnasium)
            if use_discrete_actions:
                # 이산 액션 모드: continuous=False
                self.env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=False)
            else:
                self.env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=True)
            self.is_gymnasium = True
        except:
            try:
                # CarRacing-v2 시도 (Gym)
                if use_discrete_actions:
                    self.env = gym.make('CarRacing-v2', render_mode='rgb_array', continuous=False)
                else:
                    self.env = gym.make('CarRacing-v2', render_mode='rgb_array', continuous=True)
                self.is_gymnasium = False
            except Exception as e:
                error_msg = str(e)
                install_guide = ""
                
                if "swig" in error_msg.lower() or "command 'swig' failed" in error_msg:
                    install_guide = (
                        "\n\n[해결 방법]\n"
                        "1. 시스템 패키지 설치:\n"
                        "   sudo apt-get update\n"
                        "   sudo apt-get install -y swig cmake build-essential python3-dev\n"
                        "\n"
                        "2. 그 다음:\n"
                        "   pip install gymnasium[box2d]\n"
                        "\n"
                        "3. 또는 Box2D 없이 시뮬레이션 환경 사용:\n"
                        "   python train_ppo.py --env-type sim\n"
                    )
                else:
                    install_guide = (
                        f"\n필요한 패키지 설치: pip install gymnasium[box2d] 또는 pip install gym[box2d]\n"
                        f"자세한 내용은 INSTALL_BOX2D.md 참조"
                    )
                
                raise ImportError(
                    f"CarRacing 환경을 생성할 수 없습니다.\n"
                    f"에러: {error_msg}"
                    f"{install_guide}"
                )
        
        self.use_extended_actions = use_extended_actions
        
        # 상태 공간: 28x28 grayscale 이미지 (784 차원)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(784,),
            dtype=np.uint8
        )
        
        # 액션 공간 (CarRacing-v3 호환)
        if use_discrete_actions:
            # 이산 액션: 5개 액션 (RC Car 통합)
            # 0: do nothing / 정지 (coast)
            # 1: steer right / 우회전 + 가스
            # 2: steer left / 좌회전 + 가스
            # 3: gas / 직진 가스
            # 4: brake / 급정지
            self.action_space = spaces.Discrete(5)
        elif use_extended_actions:
            # 확장된 액션: [전진/후진, 좌회전/우회전]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32
            )
        else:
            # 기본 액션: left_speed, right_speed
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32
            )
        
        self.max_steps = max_steps
        self.current_step = 0
        
    def _preprocess_image(self, img):
        """
        CarRacing 이미지를 RC Car 형식으로 변환
        96x96 RGB -> 28x28 grayscale
        """
        # Grayscale 변환
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # 28x28로 리사이즈 (더 많은 정보 보존)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # uint8로 변환
        resized = np.clip(resized, 0, 255).astype(np.uint8)
        
        # 784차원 벡터로 flatten
        return np.reshape(resized, 784)
    
    def _convert_action(self, action):
        """
        RC Car 액션을 CarRacing 액션으로 변환
        
        이산 액션 모드:
        - 0: do nothing
        - 1: steer right
        - 2: steer left
        - 3: gas
        - 4: brake
        
        연속 액션 모드:
        CarRacing 액션: [steering, gas, brake]
        steering: -1.0(왼쪽) ~ 1.0(오른쪽)
        gas: 0.0 ~ 1.0
        brake: 0.0 ~ 1.0
        """
        if self.use_discrete_actions:
            # 이산 액션은 그대로 사용 (0-4)
            return int(action)
        elif self.use_extended_actions:
            # 확장된 액션: [전진/후진, 좌회전/우회전]
            forward_backward = action[0]  # -1.0(후진) ~ 1.0(전진)
            left_right = action[1]  # -1.0(좌회전) ~ 1.0(우회전)
            
            # Steering: 좌회전/우회전
            steering = -left_right  # CarRacing은 반대 방향
            
            # Gas/Brake: 전진/후진
            if forward_backward >= 0:
                gas = forward_backward
                brake = 0.0
            else:
                gas = 0.0
                brake = -forward_backward
            
            return np.array([steering, gas, brake], dtype=np.float32)
        else:
            # 기본 액션: [left_speed, right_speed]
            left_speed = action[0]
            right_speed = action[1]
            
            # Steering: 좌우 속도 차이
            steering = (right_speed - left_speed) / 2.0
            
            # Gas: 평균 속도
            avg_speed = (left_speed + right_speed) / 2.0
            if avg_speed >= 0:
                gas = avg_speed
                brake = 0.0
            else:
                gas = 0.0
                brake = -avg_speed
            
            return np.array([steering, gas, brake], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """환경 리셋"""
        self.current_step = 0
        
        # Gymnasium vs Gym API 차이 처리
        if self.is_gymnasium:
            if seed is not None:
                obs, info = self.env.reset(seed=seed, options=options)
            else:
                obs, info = self.env.reset()
        else:
            # Gym (구버전)
            obs = self.env.reset()
            info = {}
        
        # 이미지 전처리
        state = self._preprocess_image(obs)
        
        return state, info if self.is_gymnasium else state
    
    def step(self, action):
        """
        환경 스텝 실행
        
        Args:
            action: 
                - 이산 액션 모드: 정수 (0-4)
                - 연속 액션 모드: numpy 배열
        """
        # 액션 변환
        if self.use_discrete_actions:
            # 이산 액션은 정수로 변환
            if isinstance(action, (list, np.ndarray)):
                action = int(action[0]) if len(action) > 0 else int(action)
            car_racing_action = self._convert_action(action)
        else:
            car_racing_action = self._convert_action(action)
        
        # CarRacing 환경 스텝
        if self.is_gymnasium:
            obs, reward, terminated, truncated, info = self.env.step(car_racing_action)
            done = terminated or truncated
        else:
            # Gym (구버전)
            obs, reward, done, info = self.env.step(car_racing_action)
            terminated = done
            truncated = False
        
        # 이미지 전처리
        next_state = self._preprocess_image(obs)
        
        # 종료 조건
        self.current_step += 1
        done = done or (self.current_step >= self.max_steps)
        
        # 리워드 조정 (CarRacing의 리워드를 RC Car 스타일로)
        # CarRacing은 보통 -0.1 ~ 1.0 범위의 리워드를 제공
        # 추가 보너스: 속도와 거리에 따라
        
        info.update({
            'step': self.current_step,
            'original_reward': reward,
            'action_type': 'extended' if self.use_extended_actions else 'basic'
        })
        
        if self.is_gymnasium:
            return next_state, reward, done, info
        else:
            return next_state, reward, done, info
    
    def render(self, mode='human'):
        """환경 렌더링"""
        return self.env.render()
    
    def close(self):
        """환경 종료"""
        self.env.close()


if __name__ == "__main__":
    # 테스트 코드
    env = CarRacingEnvWrapper(use_extended_actions=True)
    
    # Gymnasium vs Gym API 차이 처리
    if env.is_gymnasium:
        state, info = env.reset()
    else:
        state = env.reset()
    
    print(f"Initial state shape: {state.shape}")
    print(f"Using Gymnasium: {env.is_gymnasium}")
    
    for i in range(100):
        action = env.action_space.sample()
        result = env.step(action)
        
        if env.is_gymnasium:
            next_state, reward, done, info = result
        else:
            next_state, reward, done, info = result
        
        print(f"Step {i+1}: Reward={reward:.3f}, Done={done}")
        
        if i % 10 == 0:
            env.render()
        
        if done:
            if env.is_gymnasium:
                state, info = env.reset()
            else:
                state = env.reset()
            break
    
    env.close()
