#!/usr/bin/env python3
"""
수동 피드백 기반 PPO 강화학습 훈련 스크립트
라즈베리 파이에서 실행하여 Human-in-the-Loop 학습 수행

사용법:
    python3 train_manual_feedback.py

키보드 조작:
    - [SPACE] 또는 [+]: 긍정 피드백 (선로 유지 중)
    - [-] 또는 [n]: 부정 피드백 (선로 이탈)
    - [r]: 에피소드 리셋 (차량 위치 재조정 후)
    - [s]: 모델 저장
    - [q]: 학습 종료
    - [p]: 일시정지/재개
"""

import sys
import os
import time
import threading
import queue
import argparse
import json
from datetime import datetime
from collections import deque
import numpy as np
import torch

# 키보드 입력 처리
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

from ppo_agent import PPOAgent


class TrainingMetrics:
    """
    학습 진행 상황 추적 및 모니터링 클래스
    """
    
    def __init__(self, window_size: int = 20, log_file: str = None):
        """
        Args:
            window_size: 이동 평균 계산을 위한 윈도우 크기
            log_file: 로그 파일 경로 (None이면 자동 생성)
        """
        self.window_size = window_size
        
        # 에피소드 메트릭
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_positive_rates = []  # 긍정 피드백 비율
        
        # 이동 평균용 deque
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_lengths = deque(maxlen=window_size)
        self.recent_positive_rates = deque(maxlen=window_size)
        
        # 현재 에피소드 추적
        self.current_positive = 0
        self.current_negative = 0
        self.current_neutral = 0
        
        # PPO 손실 추적
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        # 베스트 기록
        self.best_reward = float('-inf')
        self.best_positive_rate = 0.0
        self.best_episode = 0
        
        # 학습 개선 추적
        self.improvement_streak = 0  # 연속 개선 횟수
        self.no_improvement_count = 0  # 개선 없는 에피소드 수
        
        # 로그 파일
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"training_log_{timestamp}.json"
        self.log_file = log_file
        self.log_data = []
    
    def reset_episode(self):
        """에피소드 시작 시 리셋"""
        self.current_positive = 0
        self.current_negative = 0
        self.current_neutral = 0
    
    def record_feedback(self, feedback_type: str):
        """
        피드백 기록
        
        Args:
            feedback_type: 'positive', 'negative', 'neutral'
        """
        if feedback_type == 'positive':
            self.current_positive += 1
        elif feedback_type == 'negative':
            self.current_negative += 1
        else:
            self.current_neutral += 1
    
    def end_episode(self, episode: int, reward: float, length: int):
        """
        에피소드 종료 시 메트릭 업데이트
        
        Args:
            episode: 에피소드 번호
            reward: 총 리워드
            length: 에피소드 길이
        """
        # 긍정 피드백 비율 계산
        total_feedback = self.current_positive + self.current_negative
        if total_feedback > 0:
            positive_rate = self.current_positive / total_feedback
        else:
            positive_rate = 0.5  # 피드백 없으면 중립
        
        # 기록 추가
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_positive_rates.append(positive_rate)
        
        self.recent_rewards.append(reward)
        self.recent_lengths.append(length)
        self.recent_positive_rates.append(positive_rate)
        
        # 베스트 기록 업데이트
        improved = False
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_episode = episode
            improved = True
        if positive_rate > self.best_positive_rate:
            self.best_positive_rate = positive_rate
            improved = True
        
        # 개선 추적
        if improved:
            self.improvement_streak += 1
            self.no_improvement_count = 0
        else:
            self.improvement_streak = 0
            self.no_improvement_count += 1
        
        # 로그 저장
        log_entry = {
            'episode': episode,
            'reward': reward,
            'length': length,
            'positive_rate': positive_rate,
            'positive_count': self.current_positive,
            'negative_count': self.current_negative,
            'neutral_count': self.current_neutral,
            'avg_reward': self.get_avg_reward(),
            'avg_positive_rate': self.get_avg_positive_rate(),
            'timestamp': datetime.now().isoformat()
        }
        self.log_data.append(log_entry)
    
    def record_loss(self, policy_loss: float, value_loss: float, entropy: float):
        """PPO 손실 기록"""
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
    
    def get_avg_reward(self) -> float:
        """최근 평균 리워드"""
        if not self.recent_rewards:
            return 0.0
        return np.mean(self.recent_rewards)
    
    def get_avg_positive_rate(self) -> float:
        """최근 평균 긍정 피드백 비율"""
        if not self.recent_positive_rates:
            return 0.0
        return np.mean(self.recent_positive_rates)
    
    def get_learning_status(self) -> str:
        """
        학습 상태 판단
        
        Returns:
            상태 문자열: '🚀 급성장', '📈 개선 중', '➡️ 안정', '📉 정체', '⚠️ 악화'
        """
        if len(self.recent_rewards) < 5:
            return "📊 데이터 수집 중"
        
        # 최근 트렌드 분석
        recent = list(self.recent_rewards)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        improvement = (second_half - first_half) / (abs(first_half) + 1e-8)
        
        # 긍정 피드백 비율 확인
        avg_positive_rate = self.get_avg_positive_rate()
        
        if improvement > 0.2 and avg_positive_rate > 0.7:
            return "🚀 급성장"
        elif improvement > 0.05 or self.improvement_streak >= 3:
            return "📈 개선 중"
        elif abs(improvement) <= 0.05 and avg_positive_rate > 0.5:
            return "➡️ 안정"
        elif improvement < -0.1 or self.no_improvement_count > 10:
            return "⚠️ 악화"
        else:
            return "📉 정체"
    
    def get_summary(self) -> str:
        """학습 요약 문자열 생성"""
        status = self.get_learning_status()
        avg_reward = self.get_avg_reward()
        avg_positive_rate = self.get_avg_positive_rate()
        
        lines = [
            f"┌{'─'*56}┐",
            f"│ 학습 상태: {status:40} │",
            f"├{'─'*56}┤",
            f"│ 평균 리워드 (최근 {self.window_size}): {avg_reward:+8.2f}              │",
            f"│ 긍정 피드백 비율:         {avg_positive_rate*100:5.1f}%                │",
            f"│ 베스트 리워드:           {self.best_reward:+8.2f} (Ep {self.best_episode:3d})     │",
            f"├{'─'*56}┤",
        ]
        
        if self.policy_losses:
            recent_policy = np.mean(self.policy_losses[-5:])
            recent_value = np.mean(self.value_losses[-5:])
            recent_entropy = np.mean(self.entropies[-5:])
            lines.extend([
                f"│ Policy Loss:    {recent_policy:8.4f}                        │",
                f"│ Value Loss:     {recent_value:8.4f}                        │",
                f"│ Entropy:        {recent_entropy:8.4f}                        │",
            ])
        
        # 학습 조언
        advice = self._get_advice()
        lines.extend([
            f"├{'─'*56}┤",
            f"│ 💡 {advice:52} │",
            f"└{'─'*56}┘",
        ])
        
        return '\n'.join(lines)
    
    def _get_advice(self) -> str:
        """학습 상황에 맞는 조언"""
        avg_positive_rate = self.get_avg_positive_rate()
        status = self.get_learning_status()
        
        if "급성장" in status:
            return "훌륭합니다! 현재 설정을 유지하세요."
        elif avg_positive_rate < 0.3:
            return "부정 피드백이 많습니다. 속도를 낮추세요."
        elif avg_positive_rate > 0.8 and self.no_improvement_count > 5:
            return "쉬운 구간입니다. 더 어려운 코스를 시도하세요."
        elif "정체" in status or "악화" in status:
            return "학습이 멈췄습니다. 하이퍼파라미터 조정 필요."
        elif len(self.recent_rewards) < 10:
            return "더 많은 에피소드가 필요합니다. 계속하세요."
        else:
            return "꾸준히 학습 중입니다. 계속 진행하세요."
    
    def save_log(self):
        """로그 파일 저장"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def print_progress_bar(self, current: int, total: int, width: int = 30):
        """진행률 바 출력"""
        progress = current / total
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {progress*100:5.1f}%"


class KeyboardListener:
    """비동기 키보드 입력 리스너"""
    
    def __init__(self):
        self.input_queue = queue.Queue()
        self.running = False
        self.thread = None
        
    def start(self):
        """리스너 시작"""
        if not HAS_TERMIOS:
            print("경고: termios 모듈을 찾을 수 없습니다. 키보드 입력이 제한됩니다.")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        
    def stop(self):
        """리스너 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _listen(self):
        """키보드 입력 대기 (별도 스레드)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            while self.running:
                if sys.stdin in [sys.stdin]:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.input_queue.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def get_input(self):
        """큐에서 입력 가져오기 (non-blocking)"""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None


class ManualFeedbackTrainer:
    """
    수동 피드백 기반 훈련 클래스
    Human-in-the-Loop 강화학습
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        save_path: str = 'ppo_manual.pth',
        positive_reward: float = 1.0,
        negative_reward: float = -2.0,
        neutral_reward: float = 0.1,
        feedback_timeout: float = 0.5,
        update_frequency: int = 64,
        update_epochs: int = 4
    ):
        """
        Args:
            agent: PPO 에이전트 (TRM-PPO 권장)
            save_path: 모델 저장 경로
            positive_reward: 긍정 피드백 리워드 (선로 유지)
            negative_reward: 부정 피드백 리워드 (선로 이탈)
            neutral_reward: 중립 리워드 (피드백 없음 - 기본 전진 보상)
            feedback_timeout: 피드백 대기 시간 (초)
            update_frequency: PPO 업데이트 주기 (스텝 수)
            update_epochs: PPO 업데이트 에폭 수
        """
        self.agent = agent
        self.save_path = save_path
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.neutral_reward = neutral_reward
        self.feedback_timeout = feedback_timeout
        self.update_frequency = update_frequency
        self.update_epochs = update_epochs
        
        # RC Car 인터페이스
        self.rc_car = None
        self._init_rc_car()
        
        # 키보드 리스너
        self.keyboard = KeyboardListener()
        
        # 상태 변수
        self.paused = False
        self.running = False
        self.episode_count = 0
        self.step_count = 0
        self.total_steps = 0
        
        # 통계
        self.episode_rewards = []
        self.positive_count = 0
        self.negative_count = 0
        
        # 학습 메트릭 추적
        self.metrics = TrainingMetrics(window_size=20)
        
    def _init_rc_car(self):
        """RC Car 인터페이스 초기화"""
        try:
            from rc_car_interface import RC_Car_Interface
            self.rc_car = RC_Car_Interface()
            print("✓ RC Car 인터페이스 초기화 완료")
        except ImportError as e:
            print(f"✗ RC Car 인터페이스 로드 실패: {e}")
            print("  시뮬레이션 모드로 전환합니다.")
            self.rc_car = None
    
    def get_state(self) -> np.ndarray:
        """현재 상태 (카메라 이미지) 가져오기"""
        if self.rc_car:
            img = self.rc_car.get_image_from_camera()
            return np.reshape(img, -1).astype(np.float32) / 255.0
        else:
            # 시뮬레이션: 랜덤 상태
            return np.random.rand(256).astype(np.float32)
    
    def execute_action(self, action: int):
        """
        액션 실행
        
        Args:
            action: 이산 액션 (0-4)
                0: 정지
                1: 전진 직진
                2: 전진 좌회전
                3: 전진 우회전
                4: 후진
        """
        if not self.rc_car:
            return
        
        speed = 180  # 기본 속도 (0-255)
        turn_diff = 60  # 회전 시 속도 차이
        
        if action == 0:
            # 정지
            self.rc_car.set_left_speed(0)
            self.rc_car.set_right_speed(0)
        elif action == 1:
            # 전진 직진
            self.rc_car.set_left_speed(speed)
            self.rc_car.set_right_speed(speed)
        elif action == 2:
            # 전진 좌회전
            self.rc_car.set_left_speed(speed - turn_diff)
            self.rc_car.set_right_speed(speed + turn_diff)
        elif action == 3:
            # 전진 우회전
            self.rc_car.set_left_speed(speed + turn_diff)
            self.rc_car.set_right_speed(speed - turn_diff)
        elif action == 4:
            # 후진
            self.rc_car.set_left_speed(0)  # 후진은 별도 처리 필요
            self.rc_car.set_right_speed(0)
    
    def stop_car(self):
        """차량 정지"""
        if self.rc_car:
            self.rc_car.stop()
    
    def process_feedback(self, key: str) -> tuple:
        """
        키보드 입력 처리
        
        Args:
            key: 입력된 키
        
        Returns:
            (reward, done, action): 리워드, 종료 여부, 특수 액션
        """
        if key is None:
            # 피드백 없음 - 중립 리워드
            return self.neutral_reward, False, None
        
        key = key.lower()
        
        if key == ' ' or key == '+' or key == '=':
            # 긍정 피드백: 선로 유지 중 → 계속 진행
            self.positive_count += 1
            self.metrics.record_feedback('positive')
            print("  ✓ 선로 유지 (+) - 계속 진행", end='\r')
            return self.positive_reward, False, None
        
        elif key == '-' or key == 'n':
            # 부정 피드백: 선로 이탈 → 즉시 정지!
            self.negative_count += 1
            self.metrics.record_feedback('negative')
            self.stop_car()  # 즉시 정지
            print("\n  ✗ 선로 이탈! 차량 정지됨")
            print("    → 차량을 선로에 다시 올려놓으세요")
            print("    → [SPACE] 누르면 재개, [r] 누르면 에피소드 리셋")
            return self.negative_reward, False, 'wait_reposition'
        
        elif key == 'r':
            # 에피소드 리셋
            self.stop_car()
            print("\n  ↺ 에피소드 리셋")
            return self.negative_reward, True, 'reset'
        
        elif key == 's':
            # 모델 저장
            self.agent.save(self.save_path)
            print(f"\n  💾 모델 저장됨: {self.save_path}")
            return None, False, 'save'
        
        elif key == 'p':
            # 일시정지/재개
            self.paused = not self.paused
            status = "일시정지" if self.paused else "재개"
            print(f"\n  ⏸ {status}")
            return None, False, 'pause'
        
        elif key == 'q' or key == '\x03':  # q 또는 Ctrl+C
            # 종료
            print("\n  ⏹ 학습 종료")
            return None, True, 'quit'
        
        else:
            # 알 수 없는 키
            return self.neutral_reward, False, None
    
    def print_status(self, episode_reward: float):
        """현재 상태 출력"""
        print(f"\r[Ep {self.episode_count}] "
              f"Step: {self.step_count:4d} | "
              f"Total: {self.total_steps:6d} | "
              f"Reward: {episode_reward:+.2f} | "
              f"(+): {self.positive_count} (-): {self.negative_count}    ", end='')
    
    def train(self, max_episodes: int = 1000, max_steps_per_episode: int = 500):
        """
        수동 피드백 기반 학습 실행
        
        Args:
            max_episodes: 최대 에피소드 수
            max_steps_per_episode: 에피소드 당 최대 스텝 수
        """
        print("=" * 60)
        print("수동 피드백 기반 PPO 학습")
        print("=" * 60)
        print()
        print("키보드 조작:")
        print("  [SPACE] / [+]: 긍정 피드백 (선로 유지)")
        print("  [-] / [n]   : 부정 피드백 (선로 이탈)")
        print("  [r]         : 에피소드 리셋")
        print("  [s]         : 모델 저장")
        print("  [p]         : 일시정지/재개")
        print("  [q]         : 학습 종료")
        print()
        print("=" * 60)
        print()
        
        # TRM-PPO 모드 확인
        use_recurrent = getattr(self.agent, 'use_recurrent', False)
        if use_recurrent:
            print(f"TRM-PPO 모드 활성화 (n_cycles={self.agent.n_cycles})")
        
        # 키보드 리스너 시작
        self.keyboard.start()
        self.running = True
        
        try:
            for episode in range(max_episodes):
                if not self.running:
                    break
                
                self.episode_count = episode + 1
                self.step_count = 0
                episode_reward = 0.0
                
                # 상태 초기화
                state = self.get_state()
                
                # TRM-PPO: 잠재 상태 리셋
                if use_recurrent:
                    self.agent.reset_carry()
                
                print(f"\n--- 에피소드 {self.episode_count} 시작 ---")
                print("차량을 선로 위에 올려놓고 학습을 시작하세요.")
                print("준비되면 아무 키나 누르세요...")
                
                # 시작 대기
                while self.running:
                    key = self.keyboard.get_input()
                    if key:
                        break
                    time.sleep(0.1)
                
                if not self.running:
                    break
                
                # 에피소드 실행
                done = False
                while not done and self.step_count < max_steps_per_episode and self.running:
                    # 일시정지 체크
                    while self.paused and self.running:
                        self.stop_car()
                        time.sleep(0.1)
                        key = self.keyboard.get_input()
                        if key:
                            self.process_feedback(key)
                    
                    if not self.running:
                        break
                    
                    # 상태 텐서 변환
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    
                    # 액션 선택
                    if use_recurrent:
                        action, log_prob, value, latent_np = self.agent.get_action_with_carry(state_tensor)
                    else:
                        action, log_prob, value = self.agent.actor_critic.get_action(state_tensor)
                        latent_np = None
                    
                    # 이산 액션 추출
                    action_int = action.squeeze().cpu().item()
                    if isinstance(action_int, float):
                        action_int = int(action_int)
                    
                    # 액션 실행
                    self.execute_action(action_int)
                    
                    # 피드백 대기
                    time.sleep(self.feedback_timeout)
                    
                    # 다음 상태
                    next_state = self.get_state()
                    
                    # 키보드 피드백 처리
                    key = self.keyboard.get_input()
                    reward, done, special_action = self.process_feedback(key)
                    
                    if special_action == 'quit':
                        self.running = False
                        break
                    elif special_action in ['save', 'pause']:
                        continue
                    elif special_action == 'wait_reposition':
                        # 선로 이탈! 차량이 멈추고 재배치 대기
                        # 사용자가 SPACE를 누르면 재개, 'r'을 누르면 에피소드 리셋
                        while self.running:
                            reposition_key = self.keyboard.get_input()
                            if reposition_key == ' ' or reposition_key == '+':
                                print("    ▶ 재개!")
                                break
                            elif reposition_key == 'r':
                                print("    ↺ 에피소드 리셋")
                                done = True
                                break
                            elif reposition_key == 'q':
                                self.running = False
                                break
                            time.sleep(0.1)
                        
                        if done or not self.running:
                            break
                        
                        # 새 상태 갱신 (재배치 후)
                        next_state = self.get_state()
                    
                    if reward is None:
                        reward = self.neutral_reward
                    
                    # 버퍼에 저장
                    log_prob_val = log_prob.cpu().item() if log_prob is not None else 0.0
                    value_val = value.squeeze().cpu().item()
                    
                    if use_recurrent:
                        self.agent.store_transition(
                            state, action_int, reward, done,
                            log_prob_val, value_val, latent=latent_np
                        )
                    else:
                        self.agent.store_transition(
                            state, action_int, reward, done,
                            log_prob_val, value_val
                        )
                    
                    episode_reward += reward
                    self.step_count += 1
                    self.total_steps += 1
                    state = next_state
                    
                    # 상태 출력
                    self.print_status(episode_reward)
                    
                    # PPO 업데이트
                    if len(self.agent.buffer['states']) >= self.update_frequency:
                        self.stop_car()
                        print(f"\n  🔄 PPO 업데이트 중...")
                        loss_info = self.agent.update(epochs=self.update_epochs)
                        if loss_info:
                            print(f"     Loss: {loss_info['loss']:.4f}, "
                                  f"Policy: {loss_info['policy_loss']:.4f}, "
                                  f"Value: {loss_info['value_loss']:.4f}")
                            # 메트릭에 손실 기록
                            self.metrics.record_loss(
                                loss_info['policy_loss'],
                                loss_info['value_loss'],
                                loss_info['entropy']
                            )
                
                # 에피소드 종료
                self.stop_car()
                self.episode_rewards.append(episode_reward)
                
                # 메트릭 기록
                self.metrics.end_episode(self.episode_count, episode_reward, self.step_count)
                self.metrics.reset_episode()
                
                print(f"\n--- 에피소드 {self.episode_count} 종료 ---")
                print(f"    총 스텝: {self.step_count}")
                print(f"    에피소드 리워드: {episode_reward:.2f}")
                
                # 학습 상태 요약 (5 에피소드마다)
                if self.episode_count % 5 == 0:
                    print()
                    print(self.metrics.get_summary())
                elif len(self.episode_rewards) > 1:
                    avg_reward = self.metrics.get_avg_reward()
                    pos_rate = self.metrics.get_avg_positive_rate()
                    status = self.metrics.get_learning_status()
                    print(f"    {status} | 평균 리워드: {avg_reward:+.2f} | 긍정률: {pos_rate*100:.1f}%")
                
                # 에피소드마다 자동 저장
                if episode % 5 == 0:
                    self.agent.save(self.save_path)
                    print(f"    💾 자동 저장됨")
        
        except KeyboardInterrupt:
            print("\n\n학습 중단됨 (Ctrl+C)")
        
        finally:
            self.stop_car()
            self.keyboard.stop()
            
            # 최종 저장
            self.agent.save(self.save_path)
            
            # 로그 파일 저장
            self.metrics.save_log()
            
            # 최종 학습 요약
            print("\n")
            print("=" * 60)
            print("📊 학습 완료 - 최종 요약")
            print("=" * 60)
            print()
            print(self.metrics.get_summary())
            print()
            print(f"총 에피소드: {self.episode_count}")
            print(f"총 스텝: {self.total_steps}")
            print(f"긍정 피드백 수: {self.positive_count}")
            print(f"부정 피드백 수: {self.negative_count}")
            
            if self.episode_rewards:
                print()
                print("리워드 통계:")
                print(f"  - 평균: {np.mean(self.episode_rewards):.2f}")
                print(f"  - 최고: {np.max(self.episode_rewards):.2f}")
                print(f"  - 최저: {np.min(self.episode_rewards):.2f}")
                print(f"  - 표준편차: {np.std(self.episode_rewards):.2f}")
            
            print()
            print(f"모델 저장 위치: {self.save_path}")
            print(f"로그 저장 위치: {self.metrics.log_file}")
            print()
            
            # 학습 결과 판정
            final_status = self.metrics.get_learning_status()
            avg_pos_rate = self.metrics.get_avg_positive_rate()
            
            if avg_pos_rate > 0.8:
                print("🎉 훌륭합니다! 모델이 선로를 잘 따라갑니다.")
            elif avg_pos_rate > 0.6:
                print("👍 좋은 진전입니다. 더 학습하면 개선될 것입니다.")
            elif avg_pos_rate > 0.4:
                print("📈 학습이 진행 중입니다. 더 많은 피드백이 필요합니다.")
            else:
                print("💡 더 많은 학습이 필요합니다. 하이퍼파라미터 조정을 고려하세요.")
            
            print("=" * 60)
            
            # RC Car 정리
            if self.rc_car:
                self.rc_car.close()


def main():
    parser = argparse.ArgumentParser(description='수동 피드백 기반 PPO 학습')
    
    # 학습 파라미터
    parser.add_argument('--max-episodes', type=int, default=100,
                        help='최대 에피소드 수 (기본: 100)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='에피소드 당 최대 스텝 수 (기본: 500)')
    parser.add_argument('--update-frequency', type=int, default=64,
                        help='PPO 업데이트 주기 (기본: 64)')
    parser.add_argument('--update-epochs', type=int, default=4,
                        help='PPO 업데이트 에폭 수 (기본: 4)')
    
    # 리워드 파라미터
    parser.add_argument('--positive-reward', type=float, default=1.0,
                        help='긍정 피드백 리워드 (기본: 1.0)')
    parser.add_argument('--negative-reward', type=float, default=-2.0,
                        help='부정 피드백 리워드 (기본: -2.0)')
    parser.add_argument('--neutral-reward', type=float, default=0.1,
                        help='중립 리워드 (기본: 0.1)')
    parser.add_argument('--feedback-timeout', type=float, default=0.3,
                        help='피드백 대기 시간 초 (기본: 0.3)')
    
    # TRM-PPO 파라미터
    parser.add_argument('--use-recurrent', action='store_true', default=True,
                        help='TRM-PPO 모드 사용 (기본: True)')
    parser.add_argument('--no-recurrent', dest='use_recurrent', action='store_false',
                        help='기존 PPO 모드 사용')
    parser.add_argument('--n-cycles', type=int, default=4,
                        help='TRM-PPO 재귀 추론 반복 횟수 (기본: 4)')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='TRM-PPO 잠재 상태 차원 (기본: 256)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='히든 레이어 차원 (기본: 256)')
    
    # 저장/로드
    parser.add_argument('--save-path', type=str, default='ppo_manual.pth',
                        help='모델 저장 경로 (기본: ppo_manual.pth)')
    parser.add_argument('--load-path', type=str, default=None,
                        help='사전학습 모델 로드 경로')
    
    # 디바이스
    parser.add_argument('--device', type=str, default=None,
                        help='디바이스 (cuda/cpu, 기본: 자동)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"디바이스: {device}")
    
    # 에이전트 생성
    agent = PPOAgent(
        state_dim=256,
        action_dim=5,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_cycles=args.n_cycles,
        carry_latent=True,
        discrete_action=True,
        num_discrete_actions=5,
        use_recurrent=args.use_recurrent,
        device=device,
        # 수동 피드백에 맞는 하이퍼파라미터
        lr_actor=1e-4,
        lr_critic=1e-4,
        gamma=0.95,
        gae_lambda=0.9,
        clip_epsilon=0.1,
        entropy_coef=0.05  # 탐색 강화
    )
    
    # 사전학습 모델 로드
    if args.load_path and os.path.exists(args.load_path):
        agent.load(args.load_path)
        print(f"사전학습 모델 로드됨: {args.load_path}")
    
    # 트레이너 생성 및 실행
    trainer = ManualFeedbackTrainer(
        agent=agent,
        save_path=args.save_path,
        positive_reward=args.positive_reward,
        negative_reward=args.negative_reward,
        neutral_reward=args.neutral_reward,
        feedback_timeout=args.feedback_timeout,
        update_frequency=args.update_frequency,
        update_epochs=args.update_epochs
    )
    
    trainer.train(
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps
    )


if __name__ == "__main__":
    main()

