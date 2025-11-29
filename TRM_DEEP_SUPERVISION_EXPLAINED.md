# TRM 구조의 Deep Supervision 원리 설명

## 현재 구조 분석

### 1. TRM (Tiny Reasoning Model) 구조

현재 코드에서 TRM 구조는 다음과 같이 구현되어 있습니다:

```
State (256차원)
    ↓
Encoder (state → state_embedding, 256차원)
    ↓
Initial Latent (학습 가능한 초기 잠재 상태)
    ↓
┌─────────────────────────────────────┐
│ RecurrentReasoningBlock 반복 (n_cycles=4) │
│                                      │
│  Cycle 1: [state_emb + latent] → MLP → latent₁ │
│  Cycle 2: [state_emb + latent₁] → MLP → latent₂ │
│  Cycle 3: [state_emb + latent₂] → MLP → latent₃ │
│  Cycle 4: [state_emb + latent₃] → MLP → latent₄ │
└─────────────────────────────────────┘
    ↓
최종 Latent (latent₄)
    ↓
┌─────────────┬─────────────┐
│   Actor     │   Critic    │
│  (정책)      │  (가치)      │
└─────────────┴─────────────┘
```

### 2. 현재 구현의 문제점: Deep Supervision 부재

**현재 코드 (`ppo_agent.py:143-195`):**
```python
# HRM 스타일 근사 그래디언트: 내부 반복은 no_grad
if n_cycles > 1:
    with torch.no_grad():
        for _ in range(n_cycles - 1):
            latent = self.reasoning_block(state_emb, latent)

# 마지막 단계만 gradient 계산
latent = self.reasoning_block(state_emb, latent.detach())

# Actor/Critic 출력 (최종 latent만 사용)
value = self.critic(latent)
action_logits = self.actor(latent)
```

**문제점:**
- ❌ 중간 reasoning cycle의 출력을 활용하지 않음
- ❌ 각 cycle마다의 학습 신호가 없음
- ❌ 초기 cycle들이 제대로 학습되지 않을 수 있음

### 3. Deep Supervision이 필요한 이유

TRM 구조에서 Deep Supervision은 다음과 같은 이유로 중요합니다:

1. **그래디언트 흐름 개선**: 
   - 초기 reasoning cycle에도 직접적인 학습 신호 제공
   - 깊은 네트워크에서 그래디언트 소실 문제 완화

2. **중간 표현 학습 강화**:
   - 각 cycle이 점진적으로 더 나은 추론을 수행하도록 유도
   - 초기 cycle: 기본 패턴 학습
   - 후기 cycle: 세밀한 추론 수행

3. **학습 안정성**:
   - 여러 단계에서의 손실을 가중합하여 더 안정적인 학습
   - 일부 cycle이 실패해도 다른 cycle이 보완

## 추론 시 동작 원리

### 현재 추론 흐름 (`run_ai_agent.py:286-294`)

```python
# 1. 상태 입력
state_normalized = state.astype(np.float32) / 255.0
state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0)

# 2. TRM-PPO: 재귀 추론 수행
action, _, value, _ = self.agent.get_action_with_carry(
    state_tensor, deterministic=True
)
```

**내부 동작 (`ppo_agent.py:143-195`):**

1. **상태 인코딩**: `state → state_embedding (256차원)`
2. **잠재 상태 초기화**: `init_latent` 또는 이전 에피소드의 `carry`
3. **재귀 추론 반복** (n_cycles=4):
   ```
   Cycle 1: [state_emb, latent₀] → MLP → latent₁
   Cycle 2: [state_emb, latent₁] → MLP → latent₂  (no_grad)
   Cycle 3: [state_emb, latent₂] → MLP → latent₃  (no_grad)
   Cycle 4: [state_emb, latent₃] → MLP → latent₄  (grad 계산)
   ```
4. **최종 출력**: `latent₄`에서만 actor/critic 출력 계산

**특징:**
- 추론 시에는 모든 cycle이 실행됨 (no_grad는 학습 시에만 의미)
- 최종 `latent₄`만 사용하여 액션 결정
- 에피소드 내에서 `carry`로 잠재 상태 유지

## 학습 시 동작 원리

### PPO 학습 흐름 (`train_ppo.py:119-128`)

```python
# 1. 액션 샘플링 (데이터 수집)
action, log_prob, value, latent_np = agent.get_action_with_carry(state_tensor)

# 2. 환경 스텝
next_state, reward, done, info = env.step(action_np)

# 3. 버퍼 저장 (잠재 상태도 저장)
agent.store_transition(
    state_normalized.copy(),
    action_np,
    reward,
    done,
    log_prob_np,
    value_np,
    latent_np  # 재현을 위해 저장
)
```

### PPO 업데이트 (`ppo_agent.py:742-889`)

```python
# 1. 저장된 잠재 상태로 재현
if self.use_recurrent:
    log_probs, values, entropy = self.actor_critic.evaluate(
        states, actions, latent=latents, n_cycles=self.n_cycles
    )

# 2. PPO 손실 계산 (최종 출력만 사용)
policy_loss = -torch.min(surr1, surr2).mean()
value_loss = F.mse_loss(values, returns)
total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

**문제점:**
- `evaluate()`에서도 최종 latent만 사용
- 중간 cycle의 출력에 대한 손실이 없음

## Deep Supervision 구현 방법

### 1. 구조 변경: 중간 출력 저장

```python
def forward_with_deep_supervision(
    self,
    state: torch.Tensor,
    carry: Optional[LatentCarry] = None,
    n_cycles: Optional[int] = None,
    return_intermediates: bool = False
):
    """Deep Supervision을 포함한 forward"""
    # ... 상태 인코딩 ...
    
    intermediate_latents = []  # 중간 출력 저장
    
    # 모든 cycle을 grad 계산 (Deep Supervision을 위해)
    for cycle in range(n_cycles):
        latent = self.reasoning_block(state_emb, latent)
        intermediate_latents.append(latent)
    
    # 최종 출력
    final_latent = intermediate_latents[-1]
    final_value = self.critic(final_latent)
    final_action_logits = self.actor(final_latent)
    
    if return_intermediates:
        # 중간 출력들도 계산
        intermediate_outputs = []
        for i, latent_i in enumerate(intermediate_latents):
            value_i = self.critic(latent_i)
            action_logits_i = self.actor(latent_i)
            intermediate_outputs.append({
                'latent': latent_i,
                'value': value_i,
                'action_logits': action_logits_i
            })
        return final_action_logits, final_value, intermediate_outputs
    
    return final_action_logits, None, final_value, new_carry
```

### 2. 손실 함수에 Deep Supervision 추가

```python
def compute_deep_supervision_loss(
    self,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    intermediate_outputs: List[Dict],
    weights: List[float] = None  # 각 cycle의 가중치
):
    """Deep Supervision 손실 계산"""
    if weights is None:
        # 후기 cycle에 더 높은 가중치
        weights = [0.1, 0.2, 0.3, 0.4]  # n_cycles=4인 경우
    
    total_policy_loss = 0
    total_value_loss = 0
    
    # 각 중간 출력에 대한 손실 계산
    for i, output in enumerate(intermediate_outputs):
        # Policy loss
        dist_i = Categorical(logits=output['action_logits'])
        log_prob_i = dist_i.log_prob(actions.squeeze(-1))
        ratio_i = torch.exp(log_prob_i - old_log_probs)
        surr1_i = ratio_i * advantages
        surr2_i = torch.clamp(ratio_i, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss_i = -torch.min(surr1_i, surr2_i).mean()
        
        # Value loss
        value_loss_i = F.mse_loss(output['value'].squeeze(), returns)
        
        # 가중합
        w = weights[i]
        total_policy_loss += w * policy_loss_i
        total_value_loss += w * value_loss_i
    
    return total_policy_loss, total_value_loss
```

### 3. 학습 시 적용

```python
# evaluate()에서 중간 출력 반환
log_probs, values, entropy, intermediate_outputs = self.actor_critic.evaluate_with_deep_supervision(
    states, actions, latent=latents, n_cycles=self.n_cycles
)

# Deep Supervision 손실 계산
policy_loss, value_loss = self.compute_deep_supervision_loss(
    states, actions, returns, advantages, intermediate_outputs
)
```

## 요약

### 현재 상태
- ✅ TRM 구조 구현됨 (재귀 추론 블록)
- ✅ HRM 스타일 근사 그래디언트 적용
- ❌ Deep Supervision 미구현
- ❌ 중간 cycle 출력 미활용

### Deep Supervision 추가 시 기대 효과
1. **학습 안정성 향상**: 여러 단계에서의 손실로 더 안정적인 학습
2. **초기 cycle 학습 강화**: 직접적인 학습 신호 제공
3. **표현력 향상**: 각 cycle이 점진적으로 더 나은 추론 수행

### 구현 우선순위
1. **High**: `forward()`에 중간 출력 저장 기능 추가
2. **High**: `evaluate()`에 Deep Supervision 손실 계산 추가
3. **Medium**: 가중치 스케줄링 (학습 초기에는 모든 cycle 동등, 후기에는 최종 cycle에 더 높은 가중치)
4. **Low**: 추론 시에도 중간 출력 활용 (ensemble 등)

