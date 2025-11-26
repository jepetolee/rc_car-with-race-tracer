#!/usr/bin/env python3
"""
RC Car 강화학습 환경
rc_car_interface.py를 기반으로 Gym 스타일 환경 구현
"""

import numpy as np
import gym
from gym import spaces

# 실제 하드웨어 인터페이스는 선택적 임포트
try:
    from rc_car_interface import RC_Car_Interface
    HAS_REAL_HARDWARE = True
except ImportError:
    HAS_REAL_HARDWARE = False
    RC_Car_Interface = None


class RCCarEnv(gym.Env):
    """
    RC Car 강화학습 환경
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, max_steps=1000, use_extended_actions=True, use_discrete_actions=True):
        """
        환경 초기화
        
        Args:
            max_steps: 최대 스텝 수
            use_extended_actions: 확장된 액션 공간 사용 여부 (연속 액션 모드)
                - True: [전진/후진, 좌회전/우회전] (rc_car_controller.py 스타일)
                - False: [left_speed, right_speed] (기존 방식)
            use_discrete_actions: 이산 액션 공간 사용 여부
                - True: 이산 액션 (0-4) 사용
                - False: 연속 액션 사용
        """
        super(RCCarEnv, self).__init__()
        
        # 실제 하드웨어 확인
        if not HAS_REAL_HARDWARE:
            raise ImportError(
                "실제 하드웨어 환경을 사용할 수 없습니다.\n"
                "picamera 모듈이 설치되지 않았거나 라즈베리 파이 환경이 아닙니다.\n"
                "사전학습을 위해 CarRacing 환경을 사용하세요: --env-type carracing"
            )
        
        # RC Car 인터페이스 초기화
        self.rc_car = RC_Car_Interface()
        self.use_extended_actions = use_extended_actions
        self.use_discrete_actions = use_discrete_actions
        
        # 상태 공간: 16x16 grayscale 이미지 (256 차원)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(256,), 
            dtype=np.uint8
        )
        
        # 액션 공간
        if use_discrete_actions:
            # 이산 액션: 5개 액션 (CarRacing-v3 호환)
            # 0: 정지 (do nothing / coast)
            # 1: 우회전 + 가스 (steer right + gas)
            # 2: 좌회전 + 가스 (steer left + gas)
            # 3: 직진 가스 (gas only)
            # 4: 급정지 (brake)
            self.action_space = spaces.Discrete(5)
        elif use_extended_actions:
            # 확장된 액션: [전진/후진 속도, 좌회전/우회전 각도]
            # 전진/후진: -1.0(후진) ~ 1.0(전진)
            # 좌회전/우회전: -1.0(좌회전) ~ 1.0(우회전)
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
        self.state = None
        
        # 리워드 관련 변수
        self.last_image = None
        self.stability_reward = 0.0
        
    def reset(self):
        """환경 리셋"""
        self.current_step = 0
        self.last_image = None
        self.stability_reward = 0.0
        
        # 초기 상태 (카메라 이미지)
        img = self.rc_car.get_image_from_camera()
        self.state = np.reshape(img, img.shape[0]**2).astype(np.uint8)
        
        return self.state
    
    def _discrete_to_continuous(self, discrete_action):
        """
        이산 액션을 연속 액션으로 변환 (CarRacing-v3 호환)
        
        Args:
            discrete_action: 이산 액션 (0-4)
                0: 정지 / 코스팅 (do nothing) → RC Car: 정지
                1: 우회전 + 가스 (steer right + gas)
                2: 좌회전 + 가스 (steer left + gas)
                3: 직진 가스 (gas only)
                4: 브레이크 (brake) → RC Car: 정지 (0과 동일)
        
        Returns:
            continuous_action: [전진속도, 좌우회전] 형태의 연속 액션
                전진속도: 0.0 ~ 1.0 (후진 없음)
                좌우회전: -1.0(좌) ~ 1.0(우)
        
        Note:
            CarRacing에서는 0(coast)과 4(brake)가 다르지만,
            RC Car에서는 둘 다 정지로 동일하게 처리됨
        """
        speed = 0.8  # 기본 전진 속도
        turn_factor = 0.6  # 회전 강도
        
        if discrete_action == 0 or discrete_action == 4:
            # 정지 (coast와 brake 모두 RC Car에서는 동일)
            return np.array([0.0, 0.0], dtype=np.float32)
        elif discrete_action == 1:
            # 우회전 + 가스
            return np.array([speed, turn_factor], dtype=np.float32)
        elif discrete_action == 2:
            # 좌회전 + 가스
            return np.array([speed, -turn_factor], dtype=np.float32)
        elif discrete_action == 3:
            # 직진 가스
            return np.array([speed, 0.0], dtype=np.float32)
        else:
            # 기본값: 정지
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def step(self, action):
        """
        환경 스텝 실행
        
        Args:
            action: 액션
                - use_discrete_actions=True: 정수 (0-4)
                - use_extended_actions=True: [전진/후진, 좌회전/우회전]
                - use_extended_actions=False: [left_speed, right_speed]
        
        Returns:
            observation: 다음 상태
            reward: 리워드
            done: 종료 여부
            info: 추가 정보
        """
        # 이산 액션을 연속 액션으로 변환
        if self.use_discrete_actions:
            if isinstance(action, (list, np.ndarray)):
                discrete_action = int(action[0]) if len(action) > 0 else int(action)
            else:
                discrete_action = int(action)
            action = self._discrete_to_continuous(discrete_action)
            original_action = discrete_action
        else:
            original_action = action
        
        if self.use_extended_actions or self.use_discrete_actions:
            # 새로운 액션 체계 (후진 없음, CarRacing 호환)
            forward_speed = action[0]  # 0.0 ~ 1.0 (전진만)
            left_right = action[1]  # -1.0(좌회전) ~ 1.0(우회전)
            
            # 전진 속도 계산 (후진 없음)
            base_speed = max(0, forward_speed) * 255
            base_speed = int(np.clip(base_speed, 0, 255))
            
            # 좌회전/우회전에 따른 좌우 바퀴 속도 차이
            # 우회전(+): 왼쪽 빠르게, 오른쪽 느리게
            # 좌회전(-): 왼쪽 느리게, 오른쪽 빠르게
            turn_factor = left_right * 0.5  # 최대 50% 차이
            
            left_speed = int(base_speed * (1.0 + turn_factor))
            right_speed = int(base_speed * (1.0 - turn_factor))
            
            # 속도 범위 제한 (0-255)
            left_speed = int(np.clip(left_speed, 0, 255))
            right_speed = int(np.clip(right_speed, 0, 255))
        else:
            # 기본 액션 (left_speed, right_speed)
            left_speed = int(np.clip((action[0] + 1.0) * 127.5, 0, 255))
            right_speed = int(np.clip((action[1] + 1.0) * 127.5, 0, 255))
        
        # RC Car 제어
        self.rc_car.set_left_speed(left_speed)
        self.rc_car.set_right_speed(right_speed)
        
        # 다음 상태 관찰
        img = self.rc_car.get_image_from_camera()
        next_state = np.reshape(img, img.shape[0]**2).astype(np.uint8)
        
        # 리워드 계산
        reward = self._compute_reward(img, action)
        
        # 종료 조건
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # 상태 업데이트
        self.state = next_state
        self.last_image = img
        
        info = {
            'step': self.current_step,
            'left_speed': left_speed,
            'right_speed': right_speed,
            'action_type': 'discrete' if self.use_discrete_actions else ('extended' if self.use_extended_actions else 'basic'),
            'original_action': original_action if self.use_discrete_actions else None
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self, img, action):
        """
        리워드 계산
        
        Args:
            img: 현재 카메라 이미지 (16x16)
            action: 선택한 액션 (연속 액션 형태)
        
        Returns:
            reward: 계산된 리워드
        """
        reward = 0.0
        
        # 1. 차선 추적 리워드 (이미지 중앙 영역의 밝기 활용)
        center_region = img[6:10, 6:10]  # 중앙 4x4 영역
        center_brightness = np.mean(center_region) / 255.0
        
        # 중앙이 밝으면 (차선이 있으면) 리워드
        lane_reward = center_brightness * 0.5
        reward += lane_reward
        
        # 2. 속도 유지 리워드 (적당한 속도 유지)
        speed = np.mean([abs(action[0]), abs(action[1])])
        speed_reward = -abs(speed - 0.5) * 0.3  # 0.5 근처에서 최대
        reward += speed_reward
        
        # 3. 안정성 리워드 (이전 이미지와의 유사성)
        if self.last_image is not None:
            stability = 1.0 - np.mean(np.abs(img.astype(float) - self.last_image.astype(float))) / 255.0
            stability_reward = stability * 0.2
            reward += stability_reward
        
        # 4. 방향 일관성 리워드 (차량이 직진하는 것을 선호)
        direction_diff = abs(action[0] - action[1])
        direction_reward = (1.0 - direction_diff) * 0.1
        reward += direction_reward
        
        # 5. 페널티: 너무 느리거나 멈춤
        if speed < 0.1:
            reward -= 0.5
        
        # 6. 이산 액션 모드에서 전진 액션 보너스
        if self.use_discrete_actions and abs(action[0]) > 0.5:
            reward += 0.1  # 전진 액션에 작은 보너스
        
        return reward
    
    def render(self, mode='human'):
        """환경 렌더링 (선택사항)"""
        if mode == 'human':
            pass  # 필요시 구현
    
    def close(self):
        """환경 종료"""
        self.rc_car.stop()


if __name__ == "__main__":
    # 테스트 코드
    env = RCCarEnv(max_steps=10)
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for i in range(10):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.3f}, Done={done}")
        if done:
            break
    
    env.close()

