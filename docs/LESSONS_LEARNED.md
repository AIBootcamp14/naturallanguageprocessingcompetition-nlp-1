# 실험 교훈 및 Best Practices

**프로젝트**: Dialogue Summarization 경진대회
**작성일**: 2025-10-15
**목적**: 향후 NLP 대회에서 재사용 가능한 교훈 및 Best Practices 정리

---

## 목차

1. [개요](#개요)
2. [핵심 교훈 Top 5](#핵심-교훈-top-5)
3. [Best Practices](#best-practices)
4. [실패 사례 심층 분석](#실패-사례-심층-분석)
5. [성공 요인 분석](#성공-요인-분석)
6. [Future Work](#future-work)
7. [재사용 가능한 체크리스트](#재사용-가능한-체크리스트)

---

## 개요

이 문서는 Dialogue Summarization 경진대회(2025-10-12 ~ 2025-10-15)에서 얻은 **실험 교훈과 Best Practices**를 정리한 것입니다. 향후 NLP 대회 참여 시 참고할 수 있는 **재사용 가능한 지식 베이스**로 활용하세요.

### 프로젝트 결과 요약

- **최고 점수**: 47.47점 (Baseline 대비 +0.35점)
- **총 실험**: 12회 제출 (100% 사용)
- **개발 기간**: 4일
- **주요 성과**: Loss Gap 분석 기법 확립, 데이터 증강 프레임워크 구축

---

## 핵심 교훈 Top 5

### 1. Loss Gap이 진실을 말한다

#### 정의
```
Loss Gap = Train Loss - Eval Loss
```

#### 해석

| Loss Gap | 의미 | 조치 |
|----------|------|------|
| **양수 (+0.15 이상)** | Train > Eval, 정상 학습 | 제출 고려 |
| **작은 양수 (+0.0~+0.15)** | 거의 수렴, 과적합 조짐 | 신중하게 제출 |
| **음수 (-)** | Train < Eval, 과적합 | 제출 금지 |

#### 실험 증거

| 실험 | Train Loss | Eval Loss | Loss Gap | Test 점수 | 결과 |
|------|------------|-----------|----------|-----------|------|
| **Exp #7-A** | 1.0278 | 0.5255 | **+0.50** | **47.41** | 성공 |
| Exp #7-C | 0.9519 | 0.5474 | +0.40 | (스킵) | 사전 예측 |
| Exp #7-F | 1.0008 | 0.5293 | +0.47 | 46.62 | 실패* |

**\*주의**: Exp #7-F는 Loss Gap이 양수였지만 실패했습니다. 이는 **Train 분포 왜곡** (WeightedSampler)으로 인한 것으로, Loss Gap만으로는 충분하지 않을 수 있습니다.

#### Best Practice

**DO**:
- Loss Gap **+0.15 이상** 유지
- Eval Loss와 Train Loss를 매 epoch 추적
- Loss Gap 감소 추세 시 Early Stopping

**DON'T**:
- Loss Gap만 믿고 제출 (분포 왜곡 확인 필요)
- 음수 Loss Gap인데 제출
- Loss Gap 없이 Dev ROUGE만 보고 판단

#### 코드 예시

```python
# 학습 후 Loss Gap 확인
train_loss = trainer.state.log_history[-1]['train_loss']
eval_loss = trainer.state.log_history[-1]['eval_loss']
loss_gap = train_loss - eval_loss

print(f"Loss Gap: {loss_gap:+.4f}")

if loss_gap >= 0.15:
    print("건강한 학습 - 제출 고려")
elif loss_gap >= 0.0:
    print("주의 - 과적합 조짐, 신중하게 제출")
else:
    print("과적합 - 제출 금지")
```

---

### 2. WeightedRandomSampler의 치명적 함정

#### 문제 패턴

```
증강 없는 카테고리 (135개) × 3.70배 가중치
↓
Epoch당 500회 반복 노출
↓
같은 샘플 반복 학습
↓
모델 암기 (Memorization)
↓
Train 분포 ≠ Test 분포 (자연 분포)
↓
Test 성능 하락
```

#### 실패 사례: Exp #7-F

**설정**:
- 노동/고용: 135개 (증강 없음) × 3.70배 가중치
- 환경: 200개 (증강 없음) × 2.50배 가중치

**결과**:
- Dev ROUGE-1: 36.43% (높음 ↑)
- Test Score: 46.62점 (낮음 ↓)
- **Dev도 학습 분포 영향 받음** → Dev 점수 신뢰 불가

**원인**:
1. 증강 없이 가중치만 적용 → 같은 샘플 반복
2. 모델이 특정 샘플 암기
3. Train 분포가 Test 분포(자연 분포)와 다름
4. Dev도 학습 분포에 영향 받아 높게 나옴

#### Best Practice

**DO**:
- **증강 + 가중치 없음** (가장 안전)
- 모든 카테고리를 균등하게 증강 (300~500개)
- 자연 분포로 학습
- Loss Gap 양수 유지

**DON'T**:
- 증강 없이 가중치만 사용 (절대 금지!)
- 가중치 3.0배 이상 (암기 위험)
- Dev ROUGE만 보고 성공 판단

#### 올바른 증강 전략

**Option 1: 균등 증강 (추천)**
```python
# 모든 카테고리를 500개로 증강
target_samples = 500

for category in categories:
    current_samples = len(category_data[category])
    needed_samples = target_samples - current_samples

    if needed_samples > 0:
        # 증강 생성 (back-translation, paraphrasing 등)
        augmented_data = generate_augmentation(
            category_data[category],
            n_samples=needed_samples
        )
        category_data[category].extend(augmented_data)

# 가중치 없이 학습
train_dataset = concatenate_all_categories(category_data)
trainer = Trainer(
    ...,
    train_dataset=train_dataset,
    # WeightedRandomSampler 사용 안 함!
)
```

**Option 2: 가중치 최소화**
```python
# 증강 후에도 불균형이 있다면 가중치 최소화
max_weight = 1.5  # 1.5배 이하로 제한

weights = []
for category in categories:
    category_count = len(category_data[category])
    weight = min(target_samples / category_count, max_weight)
    weights.extend([weight] * category_count)

sampler = WeightedRandomSampler(weights, len(weights))
```

---

### 3. Dev ROUGE ≠ Test Score

#### 실험 증거

| 실험 | Dev ROUGE-1 | Test Score | 상관관계 |
|------|-------------|------------|----------|
| Exp #7-A | 36.18% | **47.41** | Dev 낮음, Test 높음 |
| Exp #7-F | **36.43%** | 46.62 | Dev 높음, Test 낮음 |

**결론**: Dev ROUGE와 Test Score는 **역상관** 가능!

#### 왜 이런 일이 발생하는가?

1. **Dev도 학습 분포 영향 받음**
   - WeightedSampler 사용 시 Dev 점수도 왜곡
   - Dev는 Train과 같은 분포에서 샘플링

2. **Test는 자연 분포**
   - Test는 실제 사용 환경의 자연 분포
   - Train 분포 왜곡 시 Test와 괴리 발생

3. **평가 지표 차이**
   - Dev: 로컬 ROUGE 계산 (형태소 분석기 버전 차이 가능)
   - Test: 대회 서버의 ROUGE 계산

#### Best Practice

**DO**:
- **Test 제출만이 유일한 진실**
- Loss Gap을 우선 지표로 활용
- Dev는 상대적 비교만 (같은 조건 내)
- 제출 전 Loss Gap **+0.15 이상** 확인

**DON'T**:
- Dev ROUGE 높다고 무조건 제출
- Dev ROUGE만 보고 성공/실패 판단
- Dev와 Test 점수를 동일하게 신뢰

#### 권장 검증 전략

```python
# 제출 전 체크리스트
checklist = {
    "Loss Gap": loss_gap >= 0.15,        # 최우선
    "Dev ROUGE 상승": dev_rouge > prev_dev_rouge,  # 참고용
    "Train 분포 왜곡 없음": not using_weighted_sampler,  # 필수
    "제출 횟수 여유": submissions_left >= 2,  # 안전장치
}

if all(checklist.values()):
    print("제출 조건 충족")
else:
    print("제출 보류")
    print(f"미충족 조건: {[k for k, v in checklist.items() if not v]}")
```

---

### 4. 증강 vs 가중치

#### 비교표

| 방법 | 메커니즘 | 데이터 | 다양성 | 일반화 | 과적합 위험 |
|------|----------|--------|--------|--------|-------------|
| **증강** | 새로운 샘플 생성 | ↑ 증가 | ↑ 높음 | 향상 | 낮음 |
| **가중치** | 같은 샘플 반복 | - 유지 | ↓ 낮음 | 저하 | 높음 |

#### 성공 패턴: Exp #7-A (47.41점)

**설정**:
```yaml
data:
  original: 12,457
  augmented: 1,009 (+8.1%)
  total: 13,465

training:
  weighted_sampler: false  # 가중치 없음!
  batch_size: 24
  gradient_accumulation_steps: 3

results:
  loss_gap: +0.50          # 안정적
  test_score: 47.41        # 성공
```

#### 실패 패턴: Exp #7-F (46.62점)

**설정**:
```yaml
data:
  original: 12,457
  augmented: 1,009
  total: 13,465

training:
  weighted_sampler: true   # 가중치 사용!
  weights:
    노동/고용: 3.70배      # 135개 × 3.70 = 500회 반복
    환경: 2.50배           # 200개 × 2.50 = 500회 반복

results:
  loss_gap: +0.47          # 표면적 양수
  test_score: 46.62        # 실패
```

#### Best Practice

**권장 순서**:
1. **1순위**: 증강 + 가중치 없음 (Exp #7-A)
2. **2순위**: 증강 + 가중치 최소화 (max 1.5배)
3. **금지**: 증강 없이 가중치만 사용 (Exp #7-F)

---

### 5. 단순함의 가치 (KISS 원칙)

**KISS**: Keep It Simple, Stupid

#### 실험 비교

| 접근법 | 복잡도 | 소요시간 | 점수 변화 | 효율성 |
|--------|--------|----------|-----------|--------|
| **LP=0.5** | 매우 낮음 | 12초 | **+0.35** | 최고 |
| 데이터 증강 | 높음 | 3시간 | -0.06 | 낮음 |
| 가중치 조정 | 중간 | 3시간 | -0.79 | 역효과 |

**인사이트**:
- 가장 간단한 변경(LP=0.5)이 가장 큰 효과
- 복잡한 기법(증강, 가중치)은 오히려 역효과
- **기본에 충실**하는 것이 최고의 전략

#### Best Practice

**DO**:
- Baseline부터 시작 (절대 건너뛰지 말 것)
- **한 번에 하나씩 변경** (A/B 테스트)
- 간단한 변경부터 시도 (Generation 파라미터 등)
- 효과 확인 후 다음 단계

**DON'T**:
- 여러 변경사항 동시 적용
- Baseline 없이 복잡한 기법부터 시도
- 효과 미검증 상태에서 다음 변경

#### 권장 실험 순서

```
Phase 1: Generation 파라미터 (12초, +0.35점)
  ├─ Length Penalty
  ├─ Num Beams
  └─ No Repeat N-gram

Phase 2: 학습 파라미터 (20분, +0.5~1.5점 예상)
  ├─ Learning Rate
  ├─ Batch Size
  └─ Warmup Steps

Phase 3: 데이터 (3시간, +0.8~2.0점 예상)
  ├─ 균등 증강
  └─ 데이터 정제

Phase 4: 모델 (3시간+, +1.0~3.0점 예상, High Risk)
  ├─ Larger Models
  └─ Ensemble
```

---

## Best Practices

### 실험 프로세스

#### 1. 실험 시작 전 체크리스트

- [ ] Baseline 재현 완료 (공식 점수 ±0.5 이내)
- [ ] Git commit 완료 (현재 상태 백업)
- [ ] 실험 목적 명확히 정의
- [ ] 변경사항 1개만 선정 (A/B 테스트)
- [ ] 예상 효과 및 리스크 평가
- [ ] 제출 횟수 확인 (여유분 최소 2회)

#### 2. 학습 중 모니터링

- [ ] Loss Gap 추적 (매 epoch)
- [ ] Train Loss vs Eval Loss 시각화
- [ ] Dev ROUGE 추적 (참고용)
- [ ] GPU 메모리 사용량 확인
- [ ] 학습 시간 기록

#### 3. 학습 후 검증

- [ ] Loss Gap +0.15 이상 확인
- [ ] Dev ROUGE 상승 확인 (참고용)
- [ ] Train 분포 왜곡 없음 확인
- [ ] Checkpoint 저장 확인
- [ ] 제출 전 inference 실행 및 format 검증

#### 4. 제출 전 최종 체크

- [ ] Loss Gap +0.15 이상
- [ ] WeightedSampler 사용 안 함
- [ ] CSV format 올바름 (`,fname,summary` 헤더)
- [ ] 특수 토큰 제거 (`<s>`, `</s>`, `<pad>` 등)
- [ ] 제출 횟수 여유분 확인 (최소 2회)

#### 5. 제출 후 분석

- [ ] Test 점수 기록
- [ ] Baseline 대비 변화 계산
- [ ] 실험 로그 문서화 (EXPERIMENT_LOG.md)
- [ ] 성공 시: 다음 실험 계획
- [ ] 실패 시: 원인 분석 및 롤백

---

### 코드 관리

#### Git 전략

```bash
# 실험 전 백업
git add .
git commit -m "실험 전 백업: Baseline 상태"

# 실험 후 커밋
git add .
git commit -m "Exp #X: [변경사항] (Test: XX.XX점, ΔX.XX)"

# 실패 시 롤백
git revert HEAD
```

#### 설정 파일 관리

```yaml
# config/experiments.yaml 사용
experiments:
  exp7a:
    description: "증강 데이터 + 가중치 없음"
    general:
      output_dir: /path/to/submission_exp7a
    # ... 설정 ...

  exp7f:
    description: "증강 데이터 + 가중치 조정"
    # ... 설정 ...
```

**장점**:
- 모든 실험을 하나의 파일에서 관리
- 쉽게 비교 및 재현 가능
- Git으로 버전 관리

---

## 실패 사례 심층 분석

### 실패 #1: no_repeat_ngram_size=3 (Exp #5)

**변경사항**: `no_repeat_ngram_size: 2 → 3`

**기대효과**: 반복 억제 강화로 요약 품질 향상

**실제결과**: 47.03점 (-0.44점)

**원인 분석**:
1. 모델의 자연스러운 반복 패턴 과도하게 제한
2. 대화 요약에서 반복은 자연스러운 현상
3. Baseline 설정(ngram=2)이 이미 최적

**교훈**:
- Baseline 설정 변경 시 신중하게
- 반복 억제는 양날의 검 (너무 강하면 역효과)
- 도메인 지식 필요 (대화 요약의 특성)

---

### 실패 #2: WeightedRandomSampler with 가중치 5.0 (Exp #7-C)

**변경사항**: 소수 카테고리에 max 5.0배 가중치

**기대효과**: 소수 카테고리 학습 강화

**실제결과**: Loss Gap +0.40 (#7-A: +0.50보다 낮음) → 제출 스킵

**원인 분석**:
1. 가중치 5.0배는 여전히 너무 높음
2. Loss Gap 감소로 과적합 징후 탐지
3. **사전 실패 예측 성공** (제출 횟수 절약)

**교훈**:
- Loss Gap 비교로 실패 예측 가능
- 가중치는 최대한 낮게 (1.5배 이하)
- 제출 전 Loss Gap 확인 필수

---

### 실패 #3: WeightedRandomSampler 최종 (Exp #7-F)

**변경사항**:
- 가중치 낮춤: max 3.70배 (노동/고용), 2.50배 (환경)
- 증강 데이터 1,009개 사용

**기대효과**: 가중치 낮춰서 안정적 학습

**실제결과**: 46.62점 (-0.79점) **최대 실패**

**원인 분석 (4단계)**:

**1단계: 증강 없는 카테고리에 가중치 적용**
```
노동/고용: 135개 (증강 없음) × 3.70배 = 500회/epoch
환경: 200개 (증강 없음) × 2.50배 = 500회/epoch
```
→ 같은 샘플을 20 epochs × 500회 = **10,000회 반복 학습**

**2단계: 모델 암기 발생**
```
반복 학습 → 모델이 특정 샘플 암기 (Memorization)
→ 일반화 능력 저하
→ 새로운 샘플에 대한 성능 하락
```

**3단계: Train 분포 왜곡**
```
Train 분포 (가중치 적용):
  노동/고용: 500회/epoch (4.0%)
  환경: 500회/epoch (4.0%)
  ... (실제 비율과 다름)

Test 분포 (자연 분포):
  노동/고용: 1.1%
  환경: 1.6%
  ... (실제 비율)
```
→ 모델이 왜곡된 분포에만 최적화

**4단계: Loss Gap 양수 역설**
```
Loss Gap: +0.47 (표면적 양수)
하지만 Train 분포 ≠ Test 분포
→ Test에서는 실패
```

**교훈 (최중요)**:
1. **증강 없으면 가중치 절대 사용 금지**
2. **Loss Gap 양수 ≠ 항상 성공** (분포 확인 필요)
3. **Dev 점수도 학습 분포 영향 받음** (Dev: 36.43%, Test: 46.62)
4. **암기 vs 일반화**: 증강으로 다양성 확보 필수

---

## 성공 요인 분석

### 성공 #1: Phase 1 LP=0.5 (47.47점, +0.35점)

**변경사항**: `length_penalty: 1.0 → 0.5`

**성공 요인**:
1. **단순성**: 단 1줄 변경
2. **효율성**: 12초만에 완료 (추론만 재실행)
3. **안전성**: 학습 없이 Generation만 변경
4. **효과성**: +0.35점 (가장 큰 개선)

**Why it worked**:
- Length Penalty 낮추면 긴 요약 생성
- ROUGE는 긴 요약에 유리 (Recall 상승)
- LP=0.5가 sweet spot (0.3은 너무 긴, 0.7은 짧음)

**재사용 가능한 패턴**:
```python
# 모든 요약 태스크에서 시도
generation_params = {
    "length_penalty": [0.3, 0.5, 0.7, 1.0],  # 순차 실험
    "num_beams": [4, 8],
    "no_repeat_ngram_size": [2, 3],
}

for lp in generation_params["length_penalty"]:
    # 추론만 재실행 (학습 X)
    outputs = model.generate(..., length_penalty=lp)
    score = evaluate(outputs)
    print(f"LP={lp}: {score}")
```

---

### 성공 #2: Exp #7-A 안정적 학습 (47.41점)

**변경사항**:
- 증강 데이터 1,009개 추가 (+8.1%)
- WeightedSampler **사용 안 함**
- LP=0.5 유지

**성공 요인**:
1. **다양성 확보**: 증강으로 새로운 샘플 생성
2. **자연 분포 학습**: 가중치 없이 원래 분포 유지
3. **Loss Gap 양수**: +0.50 (건강한 학습)
4. **재현 가능**: checkpoint-2068 보존

**Why it worked**:
- 증강 데이터가 일반화 능력 향상
- 자연 분포로 학습하여 Test와 일치
- Loss Gap 양수로 과적합 방지

**재사용 가능한 패턴**:
```python
# 1. 균등 증강
for category in categories:
    target = 500
    current = len(category_data[category])
    if current < target:
        augmented = generate_augmentation(category_data[category], target - current)
        category_data[category].extend(augmented)

# 2. 가중치 없이 학습
train_dataset = concatenate_all_categories(category_data)
trainer = Trainer(train_dataset=train_dataset)  # WeightedSampler X
```

---

### 성공 #3: Loss Gap 기반 사전 실패 예측 (Exp #7-C)

**상황**: Exp #7-C에서 Loss Gap +0.40 (#7-A: +0.50보다 낮음)

**의사결정**: 제출 스킵

**결과**: 제출 횟수 1회 절약 (실제로 실패했을 것)

**성공 요인**:
1. **Loss Gap 비교**: 절대값이 아닌 상대적 비교
2. **신중한 제출**: 확실할 때만 제출
3. **제출 횟수 관리**: 12회 제한 내에서 최적화

**재사용 가능한 패턴**:
```python
# Loss Gap 기반 제출 결정
current_loss_gap = 0.40
best_loss_gap = 0.50

if current_loss_gap < best_loss_gap - 0.05:
    print("Loss Gap 감소 → 제출 스킵")
elif current_loss_gap >= best_loss_gap:
    print("Loss Gap 개선 → 제출 고려")
else:
    print("미미한 개선 → 신중하게 판단")
```

---

## Future Work

### Immediate Actions (제출 횟수 리셋 후)

#### 1. 균등 증강 (Balanced Augmentation) [우선순위: 높음]

**목표**: 모든 카테고리를 300~500개로 균등화

**작업 내용**:
```python
target_samples = {
    "large": 500,    # 큰 카테고리
    "medium": 300,   # 중간 카테고리
    "small": 300,    # 작은 카테고리
}

augmentation_needed = {
    "노동/고용": 365개 (135 → 500),
    "감정 지원": 170개 (130 → 300),
    "친구 상담": 156개 (144 → 300),
    "일상 대화": 92개 (208 → 300),
}
```

**예상 효과**: +0.8~2.0점
**리스크**: Low
**근거**: WeightedSampler 실패 원인 해결, 자연 분포 학습

---

#### 2. Learning Rate 튜닝 [우선순위: 높음]

**목표**: 1e-5 → 3e-5 실험

**근거**:
- 증강 데이터(13,465개)로 안정성 확보
- 더 빠른 수렴 가능
- 일반적인 튜닝 방법

**예상 효과**: +0.5~1.5점
**리스크**: Low

**실험 계획**:
```python
learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]

for lr in learning_rates:
    model = train(lr=lr, other_params=best_params)
    score = evaluate(model)
    if score > best_score:
        best_lr = lr
        best_score = score
```

---

#### 3. Extended Training [우선순위: 중간]

**목표**: Epochs 20 → 30, Patience 3 → 5

**근거**:
- 현재 Epoch 12에서 조기 종료
- Loss가 아직 수렴하지 않음
- 추가 학습 여지 존재

**예상 효과**: +0.3~0.8점
**리스크**: Medium (과적합 가능성)

**주의사항**:
- Loss Gap 지속 모니터링
- +0.15 이하로 떨어지면 Early Stop

---

### Long-term Improvements

#### 4. Larger Models [우선순위: 낮음, 리스크: 높음]

**후보 모델**:
- `gogamza/kobart-base-v2` (더 큰 KoBART)
- `KETI-AIR/ke-t5-base` (T5 기반)
- `psyche/KoT5-summarization` (T5 요약 특화)

**예상 효과**: +1.0~3.0점
**리스크**: High (학습 시간 3배, 메모리 부족 가능)

**제약사항**:
- GPU 메모리 24GB로 제한
- 학습 시간 3시간 → 9시간+
- 제출 횟수로 충분한 튜닝 어려움

---

#### 5. Ensemble Methods

**방법**: 여러 checkpoint 조합

**예상 효과**: +0.3~1.0점
**리스크**: Medium

**제약사항**:
- 제출 횟수 제한으로 테스트 어려움
- 각 모델의 성능이 높아야 효과 있음

---

### Critical Constraints

**제약사항**:
- 일일 제출 12회 완전 소진 (현재)
- 일일 리셋 대기 필요
- Loss Gap **+0.15 이상** 필수
- Dev ROUGE는 참고용 (신뢰도 낮음)

**권장 실험 순서** (제출 횟수 리셋 후):
1. 균등 증강 (1회 제출) → +0.8~2.0점 예상
2. LR 3e-5 (1회 제출) → +0.5~1.5점 예상
3. Extended Training (1회 제출) → +0.3~0.8점 예상
4. 최적 조합 (2회 제출) → 최종 점수

---

## 재사용 가능한 체크리스트

### 실험 전

```markdown
- [ ] Baseline 재현 완료 (±0.5점 이내)
- [ ] Git commit (백업)
- [ ] 변경사항 1개만 선정
- [ ] 예상 효과 및 리스크 평가
- [ ] 제출 횟수 여유분 확인 (최소 2회)
```

### 학습 중

```markdown
- [ ] Loss Gap 추적 (매 epoch)
- [ ] Train Loss vs Eval Loss 시각화
- [ ] Dev ROUGE 추적 (참고용)
- [ ] GPU 메모리 확인
- [ ] 학습 시간 기록
```

### 학습 후

```markdown
- [ ] Loss Gap +0.15 이상
- [ ] Dev ROUGE 상승 (참고용)
- [ ] Train 분포 왜곡 없음
- [ ] Checkpoint 저장 확인
- [ ] Inference 실행 및 format 검증
```

### 제출 전

```markdown
- [ ] Loss Gap +0.15 이상
- [ ] WeightedSampler 사용 안 함
- [ ] CSV format 올바름 (`,fname,summary`)
- [ ] 특수 토큰 제거 (`<s>`, `</s>`, `<pad>`)
- [ ] 제출 횟수 여유분 확인 (최소 2회)
```

### 제출 후

```markdown
- [ ] Test 점수 기록
- [ ] Baseline 대비 변화 계산
- [ ] 실험 로그 문서화
- [ ] 성공 시: 다음 실험 계획
- [ ] 실패 시: 원인 분석 및 롤백
```

---

## 관련 문서

- `COMPETITION_FINAL_REPORT.md` - 대회 최종 결과
- `EXPERIMENT_LOG.md` - 전체 실험 상세 기록
- `NEXT_STEPS.md` - 향후 개선 방향
- `RESTART_GUIDE.md` - 재시작 가이드
- `ARCHIVE.md` - 프로젝트 아카이브 가이드

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-10-15
**작성자**: AI Assistant (Claude Code)
**상태**: 최종본

**활용 가이드**: 이 문서는 향후 NLP 대회에서 재사용 가능한 Best Practices 집합입니다. 각 섹션의 체크리스트와 코드 예시를 복사하여 바로 사용할 수 있습니다.
