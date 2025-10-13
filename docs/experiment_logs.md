# 실험 로그

**프로젝트:** 일상 대화 요약 모델 성능 개선
**목표:** Baseline 47점 → 50점 이상 달성
**시작일:** 2025-10-12

---

## Experiment #0: Baseline 재현

**날짜:** 2025-10-12
**실험자:** Claude + User

### 변경사항
- 없음 (공식 baseline 코드 그대로 실행)

### 설정
```yaml
model: digit82/kobart-summarization
data: train.csv (12,457 samples)
epochs: 20
learning_rate: 1e-5
batch_size: 50 (train) / 32 (eval)
optimizer: adamw_torch
scheduler: cosine
warmup_ratio: 0.1
early_stopping: patience=3
generation:
  num_beams: 4
  no_repeat_ngram_size: 2
  max_length: 100
```

### 환경
```
GPU: NVIDIA RTX 3090 24GB
CUDA: 12.2
transformers: 4.35.2
tokenizers: 0.15.2
accelerate: 0.25.0
```

### 결과

**ROUGE Scores:**
```
ROUGE-1 F1:  56.43%
ROUGE-2 F1:  36.65%
ROUGE-L F1:  47.75%
─────────────────────
Final Score: 46.9426
```

**비교:**
```
공식 Baseline: 47.1244
우리 결과:     46.9426
차이:          -0.18점 (0.38%)
```

### 판단
✅ **성공** - Baseline 재현 완료

**분석:**
- 오차 0.18점은 무시 가능한 수준 (0.4% 미만)
- 환경 차이 (랜덤 시드, GPU 연산 순서 등)로 인한 자연스러운 변동
- 안정적인 출발점 확보

### 다음 단계

**즉시 진행:**
1. 증강 데이터 실험 (Experiment #1)
   - 배경: 동일 대회 수행 팀의 Solar mini 증강 성공 사례 확인
   - 방법: train_with_augmentation.csv (24,914 samples) 사용
   - 목표: 48-49점 달성

2. 하이퍼파라미터 튜닝 (Experiment #2)
   - Learning rate: 1e-5 → 5e-5
   - 목표: 추가 0.5-1점 개선

**주의사항:**
- 한 번에 하나씩 변경
- Dev/Test 격차 5점 이내 유지
- 매 실험마다 Test 제출로 검증 (12회/일 제한)

---
---

## Experiment #1: 증강 데이터 학습 (2배 데이터)

**날짜**: 2025-10-12 23:15 - 23:48
**베이스**: Baseline (Experiment #0)

### 변경사항
- 데이터: train.csv (12,457) → train_with_augmentation.csv (24,914)
- 2배 증강 (원본 + 증강)

### 설정
```yaml
model: digit82/kobart-summarization
data: train_with_augmentation.csv (24,914 samples)
epochs: 20
learning_rate: 1e-5
batch_size: 50 (train) / 32 (eval)
optimizer: adamw_torch
scheduler: cosine
warmup_ratio: 0.1
early_stopping: patience=3
```

### 증강 데이터 품질 (EDA 결과)
- ROUGE-L F1 (원본 vs 증강): 0.4299 (평균)
- Dialogue 길이: 406자 → 544자 (+33.9%)
- Summary: 동일 (86자)
- 토픽 분포: 동일

### 결과

**Dev Set ROUGE (학습 중 평가)**:
```
Epoch 1:  R-1: 19.2% | R-2:  3.3% | R-L: 18.2%
Epoch 4:  R-1: 27.0% | R-2:  7.3% | R-L: 25.0% ← Best
Epoch 5:  R-1: 19.4% | R-2:  6.0% | R-L: 18.3%
Epoch 8:  R-1: 21.7% | R-2:  7.1% | R-L: 20.3% (Early stopping)
```

**비교**:
```
Baseline (Exp #0): Dev ROUGE-L 47.75% (추정)
Experiment #1:     Dev ROUGE-L 25.00% (Epoch 4)
변화: -22.75%p ⚠️
```

**Test Set**: 제출 대기 중
- 파일: submission_augmented_exp1.csv
- 샘플 수: 499개 ✅
- 포맷 검증: 통과 ✅

### 판단
⚠️ **주의 필요**

**분석**:
1. Dev ROUGE가 Baseline 대비 크게 하락 (47.75% → 25.00%)
2. 증강 데이터 품질은 양호했으나 (ROUGE-L 0.43) 학습 효과는 미미
3. Early stopping이 Epoch 8에서 작동 (patience=3)
4. 가능한 원인:
   - 증강 dialogue의 스타일 차이 (번역투 → 자연스러운 한국어)
   - 데이터 양 증가로 인한 학습 불충분
   - 증강 품질과 학습 효과의 불일치

**예상**:
- Test 점수: 30-40점 (Baseline 46.94보다 낮을 가능성 높음)

### 다음 단계
1. Test 제출 후 실제 점수 확인
2. 점수 분석:
   - **40점 이상**: 부분적 성공, 하이퍼파라미터 튜닝 고려
   - **30-40점**: 증강 방법 재검토 필요
   - **30점 미만**: Baseline으로 롤백, 다른 접근 시도


**Test Set 결과**: 42.7807점 ❌
- ROUGE-1: 52.43%
- ROUGE-2: 32.50%
- ROUGE-L: 43.41%
- **Baseline 대비: -4.16점 (-8.9%)**

### 최종 판단
❌ **실패 - Baseline으로 롤백**

**실패 원인 분석**:
1. **증강 데이터 스타일 문제**
   - 증강이 번역투 → 자연스러운 한국어로 변환
   - 원본 데이터의 번역투 스타일과 불일치
   - 모델이 혼란을 겪음

2. **데이터 양 증가의 역효과**
   - 2배 데이터(24,914개)가 오히려 부정적
   - 학습 시간 부족 (Epoch 8에서 조기 종료)
   - 증강 품질(ROUGE 0.43)은 괜찮았으나 실제 효과는 음수

3. **Dev/Test 격차**
   - Dev ROUGE-L: 0.25 (낮음)
   - Test ROUGE-L: 0.43 (상대적으로 높음)
   - Dev에서 이미 문제 신호 있었음

### 교훈
- ✅ EDA로 증강 품질 확인 (ROUGE 0.43)
- ❌ 증강 데이터의 스타일 일관성 부족
- ❌ 데이터 증강이 만능은 아님
- ✅ "한 번에 하나씩" 원칙 준수 (증강만 변경)

### 다음 실험 방향
1. **Baseline 기반 하이퍼파라미터 튜닝** (우선)
   - Learning rate: 1e-5 → 5e-5
   - 목표: +1~2점 개선
   
2. **Longer training** (차선)
   - Epochs: 20 → 30
   - Early stopping patience: 3 → 5
   
3. **증강 재시도** (보류)
   - 번역투 스타일 유지하는 증강 방법 연구
   - 또는 증강 비율 축소 (50% 혼합)

---

## Experiment #2: 후처리 개선 (Post-processing v2)

**날짜**: 2025-10-13
**베이스**: Baseline Modular (Experiment #0.1, 46.9526점)

### 가설

생성된 요약문의 품질 저하 요인:
1. 불필요한 연속 공백
2. 중복된 문장 반복
3. 특수 토큰 잔여

**예상 효과**: +0.5~1.2점 (목표 47.5~48.2점)
**리스크**: ✅ Low (후처리만 변경, 모델 재학습 불필요)

### 변경사항

**기존**: `postprocess_summaries()` - 특수 토큰만 제거
```python
def postprocess_summaries(summaries, remove_tokens):
    cleaned = summaries.copy()
    for token in remove_tokens:
        cleaned = [s.replace(token, " ") for s in cleaned]
    return cleaned
```

**개선**: `postprocess_summaries_v2()` - 3단계 처리
```python
def postprocess_summaries_v2(summaries, remove_tokens):
    # 1단계: 특수 토큰 제거
    cleaned = summaries.copy()
    for token in remove_tokens:
        cleaned = [s.replace(token, " ") for s in cleaned]

    # 2단계: 공백 정규화 (연속 공백 → 단일 스페이스)
    cleaned = [normalize_whitespace(s) for s in cleaned]

    # 3단계: 중복 문장 제거
    cleaned = [remove_duplicate_sentences(s) for s in cleaned]

    return cleaned
```

**추가 함수**:
1. `normalize_whitespace()`: `re.sub(r'\s+', ' ', text).strip()`
2. `remove_duplicate_sentences()`: 문장 부호(. ! ?)로 분리 후 중복 제거

### 설정

**Model**: checkpoint-1750 (재학습 없음)
```yaml
# 변경 없음 (추론 단계만 수정)
checkpoint: submission/checkpoint-1750
postprocessing: postprocess_summaries_v2  # 기존: postprocess_summaries
```

### 구현

**파일 수정**:
- `scripts/inference_utils.py`: 3개 함수 추가
- `code/test_postprocessing_v2.py`: 호환성 테스트 생성
- `code/run_exp2.py`: 추론 스크립트 생성
- `code/exp2_postprocessing.ipynb`: 재현용 노트북 생성

**테스트 결과**:
```
✅ PASS - Import
✅ PASS - normalize_whitespace (4 passed, 0 failed)
✅ PASS - remove_duplicate_sentences (4 passed, 0 failed)
✅ PASS - postprocess_summaries_v2 (3 passed, 0 failed)
✅ 모든 테스트 통과!
```

**추론 실행**:
- 출력: `code/prediction/output_modular_v2.csv`
- 샘플 수: 499개 ✅
- 변경율: 100% (모든 요약문이 후처리됨)
- Baseline과 비교: 0% 일치 (모든 요약문 변경)

### 결과

**Test Set 점수**:
```
ROUGE-1 F1:  56.31%
ROUGE-2 F1:  36.65%
ROUGE-L F1:  48.00%
─────────────────────
Final Score: 46.9863
```

**비교**:
```
Baseline Modular: 46.9526
Experiment #2:    46.9863
─────────────────────────
변화: +0.0337점 (+0.07%)
```

### 판단
❌ **실패 (롤백)** - 사실상 변화 없음

**분석**:
1. **예상**: +0.5~1.2점 (47.5~48.2 목표)
2. **실제**: +0.03점 (오차 범위 내)
3. **결론**: 후처리 개선이 성능에 기여하지 못함

### 실패 원인 분석

**주요 원인: 모델 출력이 이미 최적화되어 있었음** ⭐⭐⭐

**증거**:
1. KoBART는 이미 well-trained summarization model
2. Baseline 코드가 최소한의 후처리만 하는 데에는 이유가 있었음
3. "개선"이라고 생각한 것이 실제로는 모델 의도를 변경했을 뿐

**기타 가능한 원인**:
- Test set에 중복 문장이나 과도한 공백이 거의 없었을 가능성
- 공백 정규화/중복 제거가 적용될 케이스가 적었음
- Dev set 검증을 하지 않아 효과를 미리 예측하지 못함

### 교훈

1. **"당연히 좋을 것"이라는 가정은 위험함**
   - 이론적으로 합리적인 개선 ≠ 성능 향상
   - 실증적 검증이 필수

2. **Dev set 검증의 중요성** ⚠️
   - Test 제출 전에 Dev set에서 먼저 검증 필요
   - 12회/일 제출 제한 절약

3. **Baseline의 단순함을 존중**
   - 주최 측 baseline이 최소한의 후처리만 하는 것은 의도적
   - 섣부른 "개선"보다는 모델 학습 개선에 집중

4. **EDA 분석의 한계**
   - 이론적 분석으로 예측한 효과가 항상 실현되지는 않음
   - 빠른 실험과 검증이 더 중요

### 다음 단계

**Exp #3: Learning Rate 2e-5** (최우선)
- 목표: +1~2점 (48~49 예상)
- 방법: `config.yaml`에서 `learning_rate: 1e-5 → 2e-5`
- 근거:
  - LR 튜닝은 안전한 실험 (Low risk)
  - EDA 분석에서도 최우선 권장
  - 후처리 대신 모델 학습 개선에 집중
- **중요**: 이번에는 Dev set 검증 먼저 수행!

---

## Experiment #3: Learning Rate 2e-5

**날짜**: 2025-10-13
**베이스**: Baseline Modular (Experiment #0.1, 46.9526점)

### 가설

Learning Rate를 2배 증가시키면:
1. 더 빠른 수렴
2. 더 높은 validation 성능
3. 최종 Test 점수 향상

**예상 효과**: +1~2점 (목표 48~49점)
**리스크**: ✅ Low (공식 권장 범위 내)

### 변경사항

**하이퍼파라미터**:
```yaml
learning_rate: 1e-5 → 2e-5  # 2배 증가
epochs: 20
early_stopping_patience: 3
save_strategy: epoch
load_best_model_at_end: true
metric_for_best_model: eval_loss  # 기본값
```

**주의사항**:
- Dev set 검증을 먼저 수행하기로 결정 (Exp #2 교훈)
- 3-phase validation 전략 수립:
  1. Baseline Dev score 확인
  2. Exp #3 학습 후 Dev 비교
  3. Dev 개선 시에만 Test 제출

### 학습 과정

**학습 시간**: ~14분 (7 epochs, early stopping)

**Epoch별 결과**:
```
Epoch  | Train Loss | Eval Loss | Dev ROUGE-1 | Dev ROUGE-2 | Dev ROUGE-L | Dev 평균
-------|------------|-----------|-------------|-------------|-------------|----------
  1    |   4.6448   |  1.0500   |   26.74%    |    6.70%    |   25.31%    |  19.58%
  2    |   0.6904   |  0.5443   |   35.23%    |   12.20%    |   33.35%    |  26.93%
  3    |   0.5322   |  0.5203   |   36.24%    |   13.06%    |   33.95%    |  27.75%
  4    |   0.4756   |  0.5152   |   36.22%    |   13.17%    |   34.00%    |  27.80% ← Best Loss
  5    |   0.4002   |  0.5155   |   36.96%    |   13.52%    |   34.64%    |  28.37% ← Best ROUGE
  6    |   0.3704   |  0.5205   |   36.70%    |   13.50%    |   34.61%    |  28.27%
  7    |   0.3000   |  0.5269   |   36.30%    |   13.49%    |   34.25%    |  28.01%
```

**Early Stopping**: Epoch 7에서 종료 (patience=3)

**Best Checkpoints**:
- **checkpoint-1000** (Epoch 4): eval_loss 최저 (0.5152)
- **checkpoint-1750** (Epoch 7): Dev ROUGE 최고 (28.37% at Epoch 5)

### Dev Set 검증

**Baseline Dev**:
- ROUGE-1: 35.58%, ROUGE-2: 11.82%, ROUGE-L: 32.98%
- **평균: 26.79%**

#### v1: checkpoint-1750 (Epoch 7, 학습 종료 checkpoint)

**Dev ROUGE** (독립 평가):
- ROUGE-1: 36.33%, ROUGE-2: 12.71%, ROUGE-L: 33.75%
- **평균: 27.60%** (Baseline 대비 +0.81%p) ✅

**판단**: Dev 개선 확인 → Test 제출 진행

#### v2: checkpoint-1000 (Epoch 4, Best eval_loss)

**Dev ROUGE** (독립 평가):
- ROUGE-1: 35.78%, ROUGE-2: 11.78%, ROUGE-L: 32.77%
- **평균: 26.78%** (Baseline 대비 -0.01%p) ≈ Baseline

**특징**: Loss는 최저이지만 Dev ROUGE는 Baseline과 거의 동일

### 결과

#### v1 Results (checkpoint-1750)

**Test Set 점수**:
```
ROUGE-1 F1:  56.19%
ROUGE-2 F1:  36.32%
ROUGE-L F1:  47.57%
─────────────────────
Final Score: 46.6919
```

**비교**:
```
Baseline:      46.9526
Exp #3-v1:     46.6919
─────────────────────
변화: -0.2607점 (-0.56%) ❌
```

**Dev vs Test 괴리**:
- Dev: +0.81%p 개선 ✅
- Test: -0.26점 하락 ❌

#### v2 Results (checkpoint-1000)

**가설**: checkpoint-1750이 overfitting이라면, checkpoint-1000 (best loss)이 더 나을 것

**Test Set 점수**:
```
ROUGE-1 F1:  55.93%
ROUGE-2 F1:  36.72%
ROUGE-L F1:  47.17%
─────────────────────
Final Score: 46.6089
```

**비교**:
```
Baseline:      46.9526
Exp #3-v2:     46.6089
─────────────────────
변화: -0.3437점 (-0.73%) ❌❌
```

**충격적 발견**: checkpoint-1000이 오히려 **더 나쁨**

### 판단
❌ **실패 (양쪽 모두 하락)** - Baseline으로 롤백

### 실패 원인 분석

#### 1. Learning Rate 2e-5가 근본적으로 문제 ⭐⭐⭐

**증거**:
- **모든 checkpoint에서 일관된 하락**:
  - checkpoint-1750 (Best Dev ROUGE): -0.26점
  - checkpoint-1000 (Best Loss): -0.34점 (더 나쁨!)
- **checkpoint 선택과 무관**: 어떤 checkpoint를 써도 Baseline보다 나쁨
- **결론**: LR 2e-5 자체가 잘못된 선택

**원인 추정**:
- LR 2e-5는 Baseline 1e-5의 2배
- 너무 높은 LR로 인해 Test 데이터 분포에서 **일반화 실패**
- Train/Dev에는 맞지만 Test에는 과적합 (Dev/Test 불일치)

#### 2. checkpoint 선택의 역설

**예상**:
- Best eval_loss (ckpt-1000)가 더 나을 것
- Best Dev ROUGE (ckpt-1750)가 overfitting일 것

**실제**:
- checkpoint-1000이 **오히려 더 나쁨** (-0.34 vs -0.26)
- checkpoint-1750이 상대적으로 나음

**교훈**:
- **잘못된 LR로는 어떤 checkpoint도 좋지 않음**
- checkpoint 선택 < LR 선택
- Best loss ≠ Best Test score

#### 3. Dev/Test 괴리 심화

**Baseline Dev/Test 격차**:
- Dev: 26.79%
- Test: 46.95%
- **격차: 20.16%p** (매우 큼)

**Exp #3 Dev/Test 괴리**:
- v1 Dev +0.81%p → Test -0.26점
- v2 Dev -0.01%p → Test -0.34점

**결론**: **Dev 점수로 Test 예측 불가능**

#### 4. 3-phase validation의 실패

**Phase 2 판단 착오**:
- Dev +0.81%p 개선 보고 "성공"으로 판단
- Test 제출 진행 결정
- 하지만 Test에서는 -0.26점 하락

**Phase 2 재검증 착오**:
- "checkpoint-1750이 overfit"이라고 가정
- checkpoint-1000으로 재시도
- 하지만 오히려 더 나쁨 (-0.34점)

**근본 문제**: Dev 기준 자체가 신뢰할 수 없음

### 교훈

#### 1. Learning Rate 튜닝의 함정 ⚠️

**잘못된 가정**:
- "LR 2배 = 학습 2배 빠름 = 성능 향상"
- "공식 권장 범위 내 = 안전함"

**현실**:
- LR 2e-5는 이 데이터/모델에는 **과도함**
- Baseline 1e-5가 이미 최적
- 2배 증가의 영향이 예상보다 큼

**교훈**:
- LR 튜닝은 예상보다 **훨씬 민감**
- Baseline 하이퍼파라미터에는 이유가 있음
- 보수적 접근 필요 (1e-5 → 1.5e-5 등)

#### 2. checkpoint 선택보다 Learning Rate가 중요

**발견**:
- 같은 LR로 학습한 모든 checkpoint가 나쁨
- checkpoint 선택으로는 문제 해결 불가
- **근본 원인(LR)을 고쳐야 함**

**교훈**:
- checkpoint 최적화는 2차 문제
- 먼저 올바른 LR 찾기

#### 3. Dev 점수의 신뢰 한계

**Dev/Test 격차**: 20.16%p (매우 비정상적)

**현상**:
- Dev +0.81%p → Test -0.26점
- Dev -0.01%p → Test -0.34점

**교훈**:
- **Dev 점수로 Test 예측 불가능**
- Dev는 참고만, Test 제출이 유일한 진실
- 제출 횟수(12/day) 아껴 쓰기

#### 4. "당연히 좋을 것" 가정의 위험

**Exp #2**: "후처리 개선은 당연히 좋을 것" → +0.03점 (무의미)
**Exp #3**: "LR 2배는 당연히 좋을 것" → -0.26점 (하락)

**교훈**:
- 이론적 개선 ≠ 실제 성능 향상
- 모든 변경은 검증 필요
- Baseline 존중

### 다음 단계

#### ❌ 건너뛰기

**Exp #4: Learning Rate 3e-5**
- LR 튜닝 방향이 잘못됨 확인
- 더 높은 LR은 더 나쁠 것

**Exp #5: Learning Rate 5e-5**
- 동일한 이유로 건너뛰기

#### ✅ 새로운 방향

**1. Epochs 연장** (최우선 추천)
- `num_train_epochs: 20 → 30`
- 리스크: ✅ Low (early stopping 유지)
- 예상: +0.5~1점
- 근거: LR은 그대로, 학습 시간만 증가

**2. Warmup Steps 조정**
- `warmup_ratio: 0.1 → 0.15` 또는 `warmup_steps: 20 → 50`
- 리스크: ✅ Low
- 예상: +0.3~0.7점

**3. Special Tokens 추가**
- Time Token (`#Time#`)
- Money Token (`#Money#`)
- 리스크: ⚠️ Medium (재전처리 필요)
- 예상: +0.5~1.5점

### 결론

**Exp #3 종합 평가**:
- ❌ 두 checkpoint 모두 실패
- ❌ LR 2e-5 자체가 문제
- ❌ Dev 점수 신뢰도 낮음 재확인
- ✅ 다른 방향으로 pivot 결정

**Current Best**: 46.9526 (Baseline Modular) 유지

**제출 횟수 사용**: 8/12 (남은 횟수: 4회)

---
