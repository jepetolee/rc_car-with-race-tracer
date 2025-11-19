# CarRacing 환경 학습 파이프라인

## 전체 학습 흐름

```
CarRacing 환경 → 이미지 전처리 → PPO 에이전트 → 액션 선택 → 환경 실행 → 리워드/다음 상태
     ↑                                                                              ↓
     └─────────────────────────── 버퍼 저장 → PPO 업데이트 ←─────────────────────────┘
```

## 상세 단계별 설명

### 1단계: 환경 초기화 및 상태 관찰

```python
# CarRacing 환경 생성
env = CarRacingEnvWrapper(max_steps=1000, use_extended_actions=True)

# 환경 리셋 → 초기 상태 받기
state, info = env.reset()
```

**무슨 일이 일어나는가:**
- CarRacing-v3 환경이 시작됨
- 초기 화면(96x96 RGB 이미지)을 렌더링
- `_preprocess_image()` 함수가 이미지를 처리:
  - RGB → Grayscale 변환
  - 96x96 → 16x16 리사이즈
  - 256차원 벡터로 flatten
- 결과: `state` = [256] 크기의 numpy 배열 (0~255 값)

### 2단계: PPO 에이전트가 상태를 받아 액션 선택

```python
# 상태를 텐서로 변환
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
# shape: [1, 256] (배치 크기 1)

# Actor-Critic 네트워크로 액션 생성
action, log_prob, value = agent.actor_critic.get_action(state_tensor)
```

**네트워크 내부 동작:**

```
입력: [1, 256] (16x16 이미지 벡터)
    ↓
[Linear(256 → 256) + ReLU]
    ↓
[Linear(256 → 256) + ReLU]
    ↓
[Linear(256 → 128) + ReLU]  ← 공통 특징 추출
    ↓
    ├──────────┬──────────┐
    ↓          ↓          ↓
[Actor]    [Actor]    [Critic]
Mean       Log_Std    Value
(128→2)    (2)        (128→1)
    ↓
[정규분포 샘플링] → [Tanh] → 액션 [-1, 1]
```

**출력:**
- `action`: [전진/후진, 좌회전/우회전] (각각 -1.0 ~ 1.0)
- `log_prob`: 이 액션을 선택할 확률의 로그
- `value`: 현재 상태의 예상 가치 (V(s))

### 3단계: 액션을 CarRacing 환경에 전달

```python
# 액션을 numpy로 변환
action_np = action.squeeze(0).cpu().detach().numpy()
# shape: [2] (예: [0.8, -0.3] = 전진 80%, 좌회전 30%)

# 환경에 액션 전달
next_state, reward, done, info = env.step(action_np)
```

**액션 변환 과정 (`_convert_action`):**

```python
# RC Car 액션: [전진/후진, 좌회전/우회전]
forward_backward = 0.8   # 전진 80%
left_right = -0.3        # 좌회전 30%

# CarRacing 액션으로 변환: [steering, gas, brake]
steering = -(-0.3) = 0.3      # 오른쪽으로 조향
gas = 0.8                     # 가속
brake = 0.0                    # 브레이크 없음

car_racing_action = [0.3, 0.8, 0.0]
```

**CarRacing 환경 실행:**
- 물리 엔진(Box2D)이 차량 움직임 시뮬레이션
- 차량이 트랙을 따라 이동
- 새로운 화면 렌더링 (96x96 RGB)
- 리워드 계산 (차선 유지, 속도 등)

### 4단계: 다음 상태 전처리 및 리워드 받기

```python
# CarRacing이 반환한 이미지 (96x96 RGB)
obs = env.env.render()  # CarRacing 내부 렌더링

# 전처리: 96x96 RGB → 16x16 Grayscale → 256차원
next_state = _preprocess_image(obs)

# 리워드: CarRacing이 계산한 리워드
reward = env.env.reward  # 예: 0.1 (차선 유지), -0.1 (이탈)
```

**리워드의 의미:**
- 양수: 좋은 행동 (차선 유지, 빠른 속도)
- 음수: 나쁜 행동 (트랙 이탈, 충돌)
- CarRacing은 보통 -0.1 ~ 1.0 범위

### 5단계: 경험(Experience) 버퍼에 저장

```python
agent.store_transition(
    state.copy(),      # 현재 상태
    action_np,         # 선택한 액션
    reward,            # 받은 리워드
    done,              # 종료 여부
    log_prob_np,       # 액션 선택 확률
    value_np           # 상태 가치 예측
)
```

**버퍼 구조:**
```python
buffer = {
    'states': [state1, state2, ..., state2048],
    'actions': [action1, action2, ..., action2048],
    'rewards': [reward1, reward2, ..., reward2048],
    'dones': [False, False, ..., True],
    'log_probs': [log_prob1, ..., log_prob2048],
    'values': [value1, value2, ..., value2048]
}
```

### 6단계: PPO 업데이트 (버퍼가 가득 찰 때)

```python
# 2048개 경험이 쌓이면 업데이트
if len(agent.buffer['states']) >= 2048:
    loss_info = agent.update(epochs=10)
```

**PPO 업데이트 과정:**

1. **GAE (Generalized Advantage Estimation) 계산**
   ```python
   # 어드밴티지 계산: A(s,a) = Q(s,a) - V(s)
   advantages = compute_gae(rewards, values, dones)
   returns = advantages + values  # 실제 리턴
   ```

2. **정책 손실 계산 (PPO 클리핑)**
   ```python
   # 새 정책과 옛 정책의 비율
   ratio = exp(new_log_prob - old_log_prob)
   
   # 클리핑: 너무 큰 변화 방지
   clipped_ratio = clip(ratio, 0.8, 1.2)
   
   policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
   ```

3. **가치 함수 손실**
   ```python
   value_loss = MSE(predicted_value, actual_return)
   ```

4. **엔트로피 보너스 (탐험 장려)**
   ```python
   entropy_loss = -entropy  # 높은 엔트로피 = 더 다양한 액션
   ```

5. **총 손실 및 역전파**
   ```python
   total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
   optimizer.step()  # 네트워크 가중치 업데이트
   ```

## 전체 학습 루프

```python
for step in range(total_steps):  # 예: 500,000 스텝
    # 1. 상태 관찰
    state = env.get_current_state()  # 16x16 이미지
    
    # 2. 액션 선택
    action = agent.select_action(state)  # [전진, 회전]
    
    # 3. 환경 실행
    next_state, reward, done = env.step(action)
    
    # 4. 경험 저장
    agent.store(state, action, reward, done)
    
    # 5. 주기적 업데이트 (2048 스텝마다)
    if len(buffer) >= 2048:
        agent.update()  # PPO 학습
        buffer.clear()
    
    state = next_state
```

## 학습이 진행되면서 일어나는 일

### 초기 (0~50K 스텝)
- 랜덤 액션 선택
- 트랙 이탈 빈번
- 리워드: -100 ~ -50

### 중기 (50K~200K 스텝)
- 차선을 따라가는 패턴 학습
- 가끔 트랙 유지
- 리워드: -50 ~ 0

### 후기 (200K~500K 스텝)
- 안정적인 주행
- 트랙을 따라 빠르게 이동
- 리워드: 0 ~ 100

## 실제 코드 흐름 예시

```python
# train_ppo.py의 실제 코드
state = env.reset()  # [256] 배열

for step in range(500000):
    # 1. 상태 → 텐서 변환
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, 256]
    
    # 2. 네트워크 순전파
    action_mean, _, value = actor_critic.forward(state_tensor)
    # action_mean: [1, 2] = [[0.7, -0.2]] (전진 70%, 좌회전 20%)
    # value: [1, 1] = [[15.3]] (예상 가치)
    
    # 3. 액션 샘플링
    action = sample_from_normal(action_mean)  # [0.65, -0.18]
    action = tanh(action)  # [-1, 1] 범위로 제한
    
    # 4. 환경 실행
    next_state, reward, done = env.step(action)
    # reward: 0.1 (차선 유지)
    
    # 5. 저장
    buffer.append(state, action, reward, log_prob, value)
    
    # 6. 업데이트 (2048개마다)
    if len(buffer) == 2048:
        advantages = compute_gae(buffer.rewards, buffer.values)
        loss = compute_ppo_loss(buffer, advantages)
        optimizer.step()  # 네트워크 학습!
        buffer.clear()
    
    state = next_state
```

## 핵심 포인트

1. **이미지 전처리**: CarRacing의 96x96 RGB → 16x16 Grayscale로 변환하여 RC Car와 동일한 입력 형식 유지

2. **액션 변환**: RC Car 스타일 액션 [전진/후진, 좌회전/우회전] → CarRacing 액션 [steering, gas, brake]

3. **학습 방식**: PPO는 경험을 버퍼에 모아서 배치로 학습 (온라인 학습)

4. **전이 가능성**: CarRacing에서 학습한 특징(차선 인식, 속도 조절)이 실제 RC Car 환경에도 적용 가능

## 시각화

```
CarRacing 화면 (96x96 RGB)
    ↓ [전처리]
16x16 Grayscale (256차원)
    ↓ [Actor-Critic]
액션: [0.8, -0.3] (전진 80%, 좌회전 30%)
    ↓ [액션 변환]
CarRacing: [0.3, 0.8, 0.0] (steering, gas, brake)
    ↓ [물리 시뮬레이션]
새 화면 + 리워드
    ↓ [버퍼 저장]
경험 축적 → PPO 업데이트
```

이렇게 CarRacing 환경에서 학습한 모델은 실제 RC Car 환경에서도 사용할 수 있습니다!

