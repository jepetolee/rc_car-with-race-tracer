#!/usr/bin/env python3
"""
RC Car 시뮬레이션 환경
Pygame을 사용한 가상 트랙에서 RC Car 학습
실제 하드웨어 없이 빠르게 학습 가능
"""

import numpy as np
import pygame
import math
import gym
from gym import spaces


class RCCarSimEnv(gym.Env):
    """
    RC Car 시뮬레이션 환경
    Pygame으로 가상 트랙을 렌더링하고 물리 시뮬레이션 수행
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 track_width=800, 
                 track_height=600,
                 render_mode=None,
                 max_steps=2000,
                 use_extended_actions=True,
                 use_discrete_actions=True):
        """
        Args:
            track_width: 트랙 너비
            track_height: 트랙 높이
            render_mode: 렌더링 모드 ('human' or 'rgb_array')
            max_steps: 최대 스텝 수
            use_extended_actions: 확장된 액션 공간 사용 여부 (연속 액션)
            use_discrete_actions: 이산 액션 공간 사용 여부
        """
        super(RCCarSimEnv, self).__init__()
        
        self.track_width = track_width
        self.track_height = track_height
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.use_extended_actions = use_extended_actions
        self.use_discrete_actions = use_discrete_actions
        
        # 상태 공간: 28x28 grayscale 이미지 (784 차원)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(784,),
            dtype=np.uint8
        )
        
        # 액션 공간 (CarRacing-v3 호환)
        if use_discrete_actions:
            # 이산 액션: 5개 액션
            # 0: 정지/코스팅, 1: 우회전+가스, 2: 좌회전+가스, 3: 직진, 4: 브레이크
            self.action_space = spaces.Discrete(5)
        elif use_extended_actions:
            # 확장된 액션: [전진 속도, 좌회전/우회전 각도] (후진 없음)
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0]),
                high=np.array([1.0, 1.0]),
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
        
        # 카 상태
        self.car_x = track_width // 2
        self.car_y = track_height // 2
        self.car_angle = 0  # 각도 (라디안)
        self.car_speed = 0
        self.car_angular_velocity = 0
        
        # 물리 파라미터
        self.max_speed = 5.0
        self.max_angular_velocity = 0.1
        self.friction = 0.95
        self.angular_friction = 0.9
        
        # 트랙 생성
        self.track = None
        self.track_mask = None
        
        # 렌더링
        self.screen = None
        self.clock = None
        self.font = None
        
        # 에피소드 정보
        self.current_step = 0
        self.distance_traveled = 0
        self.last_position = None
        
        # 트랙 생성
        self._generate_track()
    
    def _discrete_to_continuous(self, discrete_action):
        """
        이산 액션을 연속 액션으로 변환 (CarRacing-v3 호환)
        
        Args:
            discrete_action: 이산 액션 (0-4)
                0: 정지/코스팅 → 정지
                1: 우회전 + 가스
                2: 좌회전 + 가스
                3: 직진 가스
                4: 브레이크 → 정지 (0과 동일)
        
        Returns:
            continuous_action: [전진속도, 좌우회전] 형태의 연속 액션
        
        Note:
            RC Car에서는 action 0(coast)과 4(brake) 모두 정지로 처리
        """
        speed = 0.8
        turn_factor = 0.6
        
        if discrete_action == 0 or discrete_action == 4:
            # 정지 (coast와 brake 모두 동일)
            return np.array([0.0, 0.0], dtype=np.float32)
        elif discrete_action == 1:
            return np.array([speed, turn_factor], dtype=np.float32)
        elif discrete_action == 2:
            return np.array([speed, -turn_factor], dtype=np.float32)
        elif discrete_action == 3:
            return np.array([speed, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)
        
    def _generate_track(self):
        """가상 트랙 생성 (타원형 트랙)"""
        # 트랙 마스크 생성 (타원형)
        center_x = self.track_width // 2
        center_y = self.track_height // 2
        radius_x = self.track_width // 3
        radius_y = self.track_height // 3
        
        # 타원형 트랙 경로 생성
        self.track_points = []
        for angle in np.linspace(0, 2 * math.pi, 100):
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            self.track_points.append((int(x), int(y)))
        
        # 트랙 내부/외부 영역 계산을 위한 마스크
        self.track_mask = np.zeros((self.track_height, self.track_width), dtype=bool)
        for y in range(self.track_height):
            for x in range(self.track_width):
                # 타원 내부인지 확인
                dx = (x - center_x) / radius_x
                dy = (y - center_y) / radius_y
                dist = dx * dx + dy * dy
                # 트랙 경로 (너비 고려)
                if 0.7 < dist < 1.3:
                    self.track_mask[y, x] = True
        
    def _get_camera_view(self):
        """
        카메라 시야 시뮬레이션
        카의 위치와 각도를 기반으로 28x28 이미지 생성
        """
        img = np.zeros((28, 28), dtype=np.uint8)
        
        # 카의 앞쪽 방향 시뮬레이션
        view_distance = 50
        view_angle_range = math.pi / 3  # 60도 시야각
        
        for i in range(28):
            for j in range(28):
                # 카메라 좌표계로 변환
                cam_x = (j - 14) / 14.0 * view_distance
                cam_y = i / 14.0 * view_distance
                
                # 월드 좌표계로 변환
                cos_a = math.cos(self.car_angle)
                sin_a = math.sin(self.car_angle)
                world_x = self.car_x + cam_x * cos_a - cam_y * sin_a
                world_y = self.car_y + cam_x * sin_a + cam_y * cos_a
                
                # 트랙 경계 확인
                world_x = int(np.clip(world_x, 0, self.track_width - 1))
                world_y = int(np.clip(world_y, 0, self.track_height - 1))
                
                # 트랙 위에 있으면 밝게
                if world_y < self.track_height and world_x < self.track_width:
                    if self.track_mask[world_y, world_x]:
                        img[i, j] = 255
                    else:
                        # 트랙 밖이면 어둡게
                        img[i, j] = 50
                else:
                    img[i, j] = 50
        
        return img
    
    def reset(self):
        """환경 리셋"""
        # 카 초기 위치 (트랙 시작점)
        self.car_x = self.track_width // 2
        self.car_y = self.track_height // 2 - self.track_height // 3
        self.car_angle = 0
        self.car_speed = 0
        self.car_angular_velocity = 0
        
        self.current_step = 0
        self.distance_traveled = 0
        self.last_position = (self.car_x, self.car_y)
        
        # 초기 상태 (28x28 = 784)
        img = self._get_camera_view()
        state = np.reshape(img, 784).astype(np.uint8)
        
        return state
    
    def step(self, action):
        """
        환경 스텝 실행
        
        Args:
            action: 액션
                - use_discrete_actions=True: 정수 (0-4)
                - use_extended_actions=True: [전진속도, 좌회전/우회전]
                - use_extended_actions=False: [left_speed, right_speed]
        
        Returns:
            observation, reward, done, info
        """
        # 이산 액션을 연속 액션으로 변환
        if self.use_discrete_actions:
            if isinstance(action, (list, np.ndarray)):
                discrete_action = int(action[0]) if len(np.atleast_1d(action)) > 0 else int(action)
            else:
                discrete_action = int(action)
            action = self._discrete_to_continuous(discrete_action)
        
        if self.use_extended_actions or self.use_discrete_actions:
            # 확장된 액션 해석 (후진 없음)
            forward_speed = max(0, action[0])  # 0.0 ~ 1.0 (전진만)
            left_right = action[1]  # -1.0(좌회전) ~ 1.0(우회전)
            
            # 속도 업데이트
            target_speed = forward_speed * self.max_speed
            self.car_speed = self.car_speed * 0.8 + target_speed * 0.2
            
            # 각속도 업데이트
            target_angular = left_right * self.max_angular_velocity
            self.car_angular_velocity = self.car_angular_velocity * 0.8 + target_angular * 0.2
        else:
            # 기본 액션 (left_speed, right_speed)
            left_speed = action[0]
            right_speed = action[1]
            
            # 속도와 각속도 계산
            speed = (left_speed + right_speed) / 2.0 * self.max_speed
            angular = (right_speed - left_speed) / 2.0 * self.max_angular_velocity
            
            self.car_speed = self.car_speed * 0.8 + speed * 0.2
            self.car_angular_velocity = self.car_angular_velocity * 0.8 + angular * 0.2
        
        # 마찰 적용
        self.car_speed *= self.friction
        self.car_angular_velocity *= self.angular_friction
        
        # 위치 업데이트
        self.car_angle += self.car_angular_velocity
        self.car_x += self.car_speed * math.cos(self.car_angle)
        self.car_y += self.car_speed * math.sin(self.car_angle)
        
        # 경계 체크
        self.car_x = np.clip(self.car_x, 0, self.track_width)
        self.car_y = np.clip(self.car_y, 0, self.track_height)
        
        # 거리 계산
        if self.last_position:
            dx = self.car_x - self.last_position[0]
            dy = self.car_y - self.last_position[1]
            self.distance_traveled += math.sqrt(dx*dx + dy*dy)
        
        self.last_position = (self.car_x, self.car_y)
        
        # 다음 상태 (28x28 = 784)
        img = self._get_camera_view()
        next_state = np.reshape(img, 784).astype(np.uint8)
        
        # 리워드 계산
        reward = self._compute_reward(img, action)
        
        # 종료 조건
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # 트랙 이탈 체크
        car_x_int = int(self.car_x)
        car_y_int = int(self.car_y)
        if (car_y_int >= 0 and car_y_int < self.track_height and 
            car_x_int >= 0 and car_x_int < self.track_width):
            if not self.track_mask[car_y_int, car_x_int]:
                reward -= 10.0  # 트랙 이탈 페널티
                done = True
        
        info = {
            'step': self.current_step,
            'distance': self.distance_traveled,
            'speed': abs(self.car_speed),
            'x': self.car_x,
            'y': self.car_y
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self, img, action):
        """리워드 계산 (CarRacing 스타일)"""
        reward = 0.0
        
        # 1. 속도 리워드 (전진 시)
        if self.use_extended_actions or self.use_discrete_actions:
            forward_speed = max(0, action[0])
            reward += forward_speed * 0.5
        else:
            speed = np.mean([abs(action[0]), abs(action[1])])
            reward += speed * 0.5
        
        # 2. 거리 리워드 (이동 거리에 비례)
        reward += self.distance_traveled * 0.01
        
        # 3. 차선 추적 리워드 (이미지 중앙 밝기)
        center_region = img[6:10, 6:10]
        center_brightness = np.mean(center_region) / 255.0
        reward += center_brightness * 1.0
        
        # 4. 안정성 리워드 (직진 유지)
        if self.use_extended_actions or self.use_discrete_actions:
            turning = abs(action[1])
            reward += (1.0 - turning) * 0.3
        
        # 5. 페널티: 정지
        if abs(self.car_speed) < 0.1:
            reward -= 0.2
        
        return reward
    
    def render(self, mode='human'):
        """환경 렌더링"""
        if mode == 'human' and self.render_mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.track_width, self.track_height))
                pygame.display.set_caption("RC Car 시뮬레이션")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 24)
            
            # 화면 지우기
            self.screen.fill((40, 40, 40))  # 어두운 배경
            
            # 트랙 그리기
            for point in self.track_points:
                pygame.draw.circle(self.screen, (255, 255, 255), point, 3)
            
            # 트랙 영역 그리기
            track_surface = pygame.surfarray.make_surface(
                self.track_mask.astype(np.uint8) * 255
            )
            self.screen.blit(track_surface, (0, 0))
            
            # 카 그리기
            car_size = 10
            car_points = [
                (self.car_x + car_size * math.cos(self.car_angle),
                 self.car_y + car_size * math.sin(self.car_angle)),
                (self.car_x + car_size * math.cos(self.car_angle + 2.5),
                 self.car_y + car_size * math.sin(self.car_angle + 2.5)),
                (self.car_x - car_size * math.cos(self.car_angle),
                 self.car_y - car_size * math.sin(self.car_angle)),
                (self.car_x + car_size * math.cos(self.car_angle - 2.5),
                 self.car_y + car_size * math.sin(self.car_angle - 2.5))
            ]
            pygame.draw.polygon(self.screen, (255, 0, 0), car_points)
            
            # 정보 표시
            info_text = [
                f"Step: {self.current_step}",
                f"Distance: {self.distance_traveled:.1f}",
                f"Speed: {abs(self.car_speed):.2f}"
            ]
            for i, text in enumerate(info_text):
                text_surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, 10 + i * 25))
            
            pygame.display.flip()
            self.clock.tick(60)
            
        elif mode == 'rgb_array':
            # RGB 배열 반환 (학습 시 사용)
            if self.screen is None:
                self.render(mode='human')
            return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
    
    def close(self):
        """환경 종료"""
        if self.screen is not None:
            pygame.quit()


if __name__ == "__main__":
    # 테스트 코드
    env = RCCarSimEnv(render_mode='human', use_extended_actions=True)
    state = env.reset()
    
    running = True
    while running:
        # 랜덤 액션
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        env.render()
        
        if done:
            state = env.reset()
        
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    env.close()

