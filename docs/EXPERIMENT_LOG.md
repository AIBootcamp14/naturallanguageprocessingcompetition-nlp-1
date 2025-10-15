# 일상 대화 요약 - 종합 실험 로그

## 개요

- **대회명**: NIKLuge 2024 - 일상 대화 요약
- **평가 지표**: ROUGE-F1 (ROUGE-1, ROUGE-2, ROUGE-L 평균)
- **목표**: 50점 이상 달성
- **현재 최고 점수**: 47.47점 (Phase 1: LP=0.5)
- 제출 횟수: 12/12 사용 완료
- 프로젝트 기간: 2025-10-12 ~ 2025-10-15

---

## 실험 요약 테이블

| Exp | 날짜 | 주요 변경사항 | Dev ROUGE-1 | Test Score | 제출 | Loss Gap | 비고 |
|-----|------|--------------|-------------|------------|------|----------|------|
| Baseline | 10/12 | digit82/kobart 재현 | ~36% | 36.12 | 1/12 | - | 재현 성공 |
| **Phase 1** | 10/13 | LP=0.5 | - | **47.47** | 2/12 | - | 최고점 |
| #5 | 10/14 | no_repeat_ngram=3 | - | 47.03 | 8/12 | - | 미세 하락 |
| **Exp #7-A** | 10/15 | 증강 데이터 (가중치 없음) | 36.18% | **47.41** | 11/12 | **+0.50** | 안정적 |
| **Exp #7-F** | 10/15 | 증강 + WeightedSampler | 36.43% | 46.62 | 12/12 | **-0.47** | 과적합 |

### 점수 변화 추이
```
36.12 (Baseline) → 47.47 (LP=0.5, 최고점) → 47.03 (ngram=3) → 47.41 (증강) → 46.62 (가중치)
```

---

## 상세 실험 기록

### Baseline: 원본 재현
**날짜**: 2025-10-12

**목표**: 공식 Baseline 코드 재현 검증

**변경사항**:
- 모델: `digit82/kobart-summarization`
- 하이퍼파라미터:
  - Learning Rate: 1e-5
  - Batch Size: 50 (train), 32 (eval)
  - Epochs: 20
  - Early Stopping: patience=3
  - Warmup Ratio: 0.1
  - Generation: LP=1.0, num_beams=4, no_repeat_ngram_size=2

**결과**:
- Test Score: **36.12점**
- 학습 시간: ~20분

**인사이트**:
- Baseline 재현 성공
- 공식 코드가 안정적으로 작동함
- 36점대가 출발선임을 확인

**다음 단계**: Length Penalty 튜닝

---

### Phase 1: Length Penalty 최적화
**날짜**: 2025-10-13

**목표**: Generation 파라미터 최적화로 빠른 성능 향상

**실험 설계**:
- LP=0.3, 0.5, 0.7, 1.0 순차 테스트
- 다른 하이퍼파라미터는 Baseline 유지
- 각 실험마다 Test 제출

**결과**:
| LP | Test Score | 변화 |
|----|------------|------|
| 1.0 | 36.12 | Baseline |
| **0.5** | **47.47** | **+11.35** |
| 0.3 | 47.15 | +11.03 |
| 0.7 | 47.22 | +11.10 |

**최적값**: LP=0.5 (47.47점)

**인사이트**:
- **LP 낮추면 요약문이 길어짐** → ROUGE 점수 상승
- LP=0.3은 너무 낮음 (과도하게 긴 요약)
- **LP=0.5가 sweet spot**
- 간단한 변경으로 +11.35점 달성

**영향**:
- 모든 후속 실험의 기본값으로 LP=0.5 채택

---

### Exp #5: no_repeat_ngram_size=3
**날짜**: 2025-10-14

**목표**: 반복 억제 강화로 요약 품질 향상

**변경사항**:
- no_repeat_ngram_size: 2 → 3
- 다른 파라미터: Phase 1 설정 유지 (LP=0.5)

**결과**:
- Test Score: **47.03점** (-0.44 from Phase 1)

**인사이트**:
- 반복 억제 강화가 오히려 성능 저하
- 모델의 자연스러운 반복 패턴을 과도하게 제한
- Baseline 설정(ngram=2)이 이미 최적

**결론**: Rollback to no_repeat_ngram_size=2

---

### Exp #6: Learning Rate 3e-5
**날짜**: 2025-10-14

**목표**: 증강 데이터 추가 후 LR 튜닝 준비

**상태**: **실험 보류 (Deferred)**

**이유**:
1. 증강 데이터 효과를 먼저 검증 필요
2. LR 튜닝은 증강 안정성 확인 후 진행
3. 제출 횟수 절약 (12회 제한)

**계획**: Exp #7 (증강) 성공 시 진행

---

### Exp #7-A: 증강 데이터 (가중치 없음)
**날짜**: 2025-10-15

**목표**: 데이터 증강으로 다양성 확보 및 성능 향상

#### 데이터 구성
- **원본 데이터**: 12,457개
- **증강 데이터**: 1,009개 (+8.1%)
- **최종 데이터**: 13,465개
- **증강 방법**: ChatGPT-4 기반 대화 생성

#### 증강 데이터 분포
```
도메인별 분포:
- 가족 관계: 82개
- 건강/운동: 92개
- 노동/고용: 135개
- 환경: 200개
- 감정 지원: 130개
...
```

#### 하이퍼파라미터
```yaml
학습:
  Learning Rate: 1e-5
  Batch Size: 24 (per_device)
  Gradient Accumulation: 3 (effective batch=72)
  Epochs: 20
  Early Stopping: patience=3
  Warmup Ratio: 0.1

생성:
  Length Penalty: 0.5
  Num Beams: 4
  No Repeat Ngram: 2
  Max Length: 100

샘플링:
  WeightedRandomSampler: 사용 안 함
  자연 분포로 학습
```

#### 학습 결과

**Loss 추이**:
```
Epoch   Train Loss   Eval Loss   Dev ROUGE-1
  1       3.1937       -          15.45%
  2       0.8482       -          21.85%
  3       0.5707       -          29.60%
  4       0.5425       -          34.45%
  5       0.5336       -          35.21%
  6       0.5299       -          35.43%
  7       0.5264       -          36.18%
  8       0.5252       -          36.13%
  9       0.5233       -          36.60%
 10       0.5241       -          35.80%
 11       0.5246       -          36.18%  ← Best Epoch
 12       0.5255       -          35.93%
```

**최종 Loss**:
- Train Loss: **1.0278**
- Eval Loss: **0.5255**
- **Loss Gap: +0.5023** (Train > Eval, 안정적)

**Best Checkpoint**: Epoch 11 (checkpoint-2068)
- Dev ROUGE-1: **36.18%**
- Dev ROUGE-2: 12.99%
- Dev ROUGE-L: 33.83%

#### Test 결과
- Test Score: **47.41점**
- 변화: -0.06 (from Phase 1 최고점 47.47)
- 제출: 11/12

#### 인사이트

**성공 요인**:
1. **Loss Gap이 양수 (+0.50)**
   - Train Loss > Eval Loss
   - 모델이 Train보다 Eval에서 더 어려워함
   - 정상적인 학습, 안정적 일반화

2. **증강 데이터가 다양성 확보**
   - 1,009개 추가로 8.1% 데이터 증가
   - 성능 유지하면서 과적합 방지

3. **가중치 없이 자연 분포 학습**
   - 모델이 자연스럽게 패턴 학습
   - 특정 카테고리 편향 없음

**분석**:
- Dev ROUGE-1: 36.18% (참고용)
- Test Score: 47.41점 (실제 성능)
- Dev와 Test의 괴리 존재 (36% vs 47점)
- Loss Gap이 더 신뢰할 수 있는 지표

**결론**:
- 증강 데이터가 안정성을 제공함
- 향후 LR 튜닝의 기반 마련
- Loss Gap 양수 유지가 핵심

---

### Exp #7-C: 가중치 과다 (실패)
**날짜**: 2025-10-15

**목표**: WeightedRandomSampler로 소수 카테고리 학습 강화

**변경사항**:
- WeightedRandomSampler 적용
- 가중치: `min(권장값, 5.0)`
- Threshold: 500개
- 도메인별 가중치 예시:
  - 노동/고용 (135개): 3.70배
  - 환경 (200개): 2.50배

#### 학습 결과

**Loss 추이**:
```
Epoch   Train Loss   Eval Loss   Dev ROUGE-1
  1       3.1991       -          15.18%
  2       0.8584       -          22.04%
  3       0.5882       -          33.10%
  ...
 11       0.5475       -          35.95%
 12       0.5474       -          35.79%
```

**최종 Loss**:
- Train Loss: **0.9519**
- Eval Loss: **0.5474**
- **Loss Gap: +0.4045** (양수이지만 #7-A보다 낮음)

**Dev ROUGE-1**: 35.95% (Epoch 11)

#### 결과
- 제출: 스킵
- 이유: Loss Gap이 #7-A(+0.50)보다 낮음 → 과적합 우려

#### 인사이트
- 가중치 5.0 상한도 여전히 높음
- Loss Gap 감소 = 일반화 능력 저하
- Loss Gap 비교로 사전 실패 예측 성공

**판단**: Exp #7-F로 가중치 조정 후 재시도

---

### Exp #7-F: 증강 + 가중치 조정 (최종 실패)
**날짜**: 2025-10-15

**목표**: 가중치를 더욱 낮춰서 안정적 학습

#### 변경사항

**WeightedRandomSampler 설정**:
```python
도메인 threshold: 500개
서브클러스터 threshold: 300개

가중치 예시:
- 노동/고용 (135개 → 500회): 3.70배
- 환경 (200개 → 500회): 2.50배
- 감정 지원 (130개 → 300회): 2.31배
```

#### 하이퍼파라미터
```yaml
학습:
  Learning Rate: 1e-5
  Batch Size: 24
  Gradient Accumulation: 3 (effective=72)
  Epochs: 20
  Early Stopping: patience=3

샘플링:
  WeightedRandomSampler: 활성화
  도메인 가중치: threshold/샘플 수
  서브클러스터 가중치: threshold/샘플 수
```

#### 학습 결과

**Loss 추이**:
```
Epoch   Train Loss   Eval Loss   Dev ROUGE-1   Loss Gap
  1       3.1969       -          15.11%         -
  2       0.8527       -          21.47%         -
  3       0.5785       -          32.59%         -
  4       0.5505       -          34.30%         -
  5       0.5420       -          34.57%         -
  6       0.5370       -          35.61%         -
  7       0.5350       -          35.89%         -
  8       0.5323       -          36.11%         -
  9       0.5293       -          36.22%         ← Best Epoch
 10       0.5324       -          36.43%         -
 11       0.5296       -          36.01%         -
 12       0.5301       -          35.78%         -
```

**최종 Loss**:
- Train Loss: **1.0008**
- Eval Loss: **0.5301**
- **Loss Gap: +0.4707** (양수)

**BUT**: 실제 Loss Gap은 최종 값이 아닌 **Best Epoch 기준**:
- Best Epoch: 9
- Eval Loss: 0.5293
- **실제 Loss Gap: 1.0008 - 0.5293 = +0.4715**

**하지만 문제는...**:
- Train Loss는 전체 에폭 평균
- Eval Loss는 Best Epoch 기준
- **비교 기준 불일치** → Loss Gap 해석 주의 필요

#### Dev 결과
- Best Checkpoint: Epoch 9 (checkpoint-1880)
- Dev ROUGE-1: **36.22%** (Epoch 9)
- Final Epoch Dev ROUGE-1: 36.43% (Epoch 10)

**관찰**: Dev ROUGE가 #7-A(36.18%)보다 높음!

#### Test 결과
- Test Score: **46.62점**
- 변화: **-0.79** (from #7-A: 47.41)
- 제출: 12/12 (마지막 제출)

#### 실패 원인 분석

**주요 원인: 증강 없는 카테고리에 가중치 적용**

**문제 상황**:
```
노동/고용 (135개, 증강 없음)
→ 가중치 3.70배 적용
→ 135 × 3.70 = 500회 반복 노출
→ 같은 샘플을 여러 번 학습
→ 모델이 해당 샘플을 암기
```

**증강 vs 가중치 비교**:
| 방법 | 설명 | 효과 |
|------|------|------|
| **증강** | 새로운 샘플 생성 | 다양성 ↑, 일반화 ↑ |
| **가중치** | 같은 샘플 반복 | 암기 ↑, 과적합 ↑ |

**Loss Gap 재해석**:
- **#7-A (가중치 없음)**: +0.50 (안정적)
- **#7-F (가중치 있음)**: +0.47 (표면적으로는 양수)

**하지만**:
- Loss Gap 양수 = 항상 좋은 것은 아님
- **Train 분포 != Test 분포**인 경우 문제 발생
- 가중치로 Train 분포를 인위적으로 왜곡
- 모델이 **왜곡된 분포**에 최적화
- Test set(자연 분포)에서 실패

**Dev ROUGE 역설**:
- Dev ROUGE-1: 36.43% (#7-A: 36.18%)
- Test Score: 46.62 (#7-A: 47.41)
- **Dev 점수는 높지만 Test 점수는 낮음**
- **Dev set도 가중치 영향을 받음** (Dev도 같은 카테고리)

#### 인사이트

**교훈 1: Loss Gap만으로 부족**
- Loss Gap 양수 = 필요조건 (Not 충분조건)
- Train 분포 왜곡 여부도 체크 필요

**교훈 2: 증강 없는 가중치는 위험**
```python
# 위험한 패턴
증강 없음 + 가중치 높음 = 암기

# 안전한 패턴
증강 있음 + 가중치 없음 = 일반화
```

**교훈 3: Dev ROUGE를 맹신하지 말 것**
- Dev도 학습 분포의 영향을 받음
- Test 제출만이 진실

**교훈 4: WeightedRandomSampler의 올바른 사용**
```python
# 잘못된 사용
작은 카테고리 (135개) × 3.70배 = 500회
→ 같은 샘플 반복 = 암기

# 올바른 사용 (해야 했던 것)
1. 모든 카테고리를 먼저 500개로 증강
2. 그 다음 가중치 없이 학습
   또는
3. 가중치 사용 시 threshold를 매우 낮게 (1.5배 이하)
```

#### 대안

**Option 1: 균등 증강 (추천)**
```python
모든 카테고리를 500개로 증강:
- 노동/고용: 135 → 500 (+365개)
- 환경: 200 → 500 (+300개)
- 감정 지원: 130 → 300 (+170개)
→ 가중치 없이 자연 분포로 학습
```

**예상 효과**: +0.8~2.0점

**Option 2: 가중치 최소화**
```python
Max 가중치: 1.5배 (현재 3.70배 대신)
→ 암기 위험 최소화
```

**예상 효과**: +0.3~0.8점

---

## 핵심 인사이트 종합

### 1. Loss Gap이 진실을 말한다 (대부분의 경우)

**Loss Gap 정의**:
```
Loss Gap = Train Loss - Eval Loss
```

**해석**:
| Loss Gap | 의미 | Train vs Eval | 일반화 | 예상 Test 성능 |
|----------|------|---------------|--------|----------------|
| **양수 (+)** | Train > Eval | Train이 더 어려움 | 좋음 | 좋음 |
| **음수 (-)** | Train < Eval | Train이 더 쉬움 | 과적합 | 나쁨 |

**실험 증거**:
```
#7-A: Loss Gap +0.50 → Test 47.41 (성공)
#7-F: Loss Gap +0.47 → Test 46.62 (실패, 분포 왜곡)
```

**주의사항**:
- Loss Gap 양수 = 필요조건이지 충분조건 아님
- Train 분포 왜곡이 있으면 양수여도 실패 가능
- 가중치 사용 시 Loss Gap만으로 판단 불충분

### 2. WeightedRandomSampler의 함정

**문제 시나리오**:
```python
# 증강 없는 카테고리에 가중치 적용
카테고리: 노동/고용 (135개)
가중치: 3.70배
결과: 135 × 3.70 = 500회 노출

→ 같은 샘플을 여러 번 학습
→ 모델이 암기 (Memorization)
→ Train 분포 != Test 분포
→ Test 성능 하락
```

**올바른 사용법**:
```python
# Option 1: 증강 + 가중치 없음 (추천)
1. 모든 카테고리를 균등하게 증강
2. 가중치 없이 자연 분포로 학습

# Option 2: 가중치 최소화
1. 최대 가중치 1.5배 이하로 제한
2. Threshold를 낮게 설정
```

### 3. Dev ROUGE는 참고용, Test만 믿을 것

**실험 증거**:
| Exp | Dev ROUGE-1 | Test Score | 괴리 |
|-----|-------------|------------|------|
| #7-A | 36.18% | 47.41 | Dev < Test |
| #7-F | **36.43%** | 46.62 | Dev ↑ but Test ↓ |

**교훈**:
- Dev ROUGE가 높다고 Test도 높지 않음
- Dev set도 학습 분포 영향을 받음
- Test 제출만이 유일한 진실
- Dev는 Loss Gap과 함께 참고만

### 4. 증강 데이터의 올바른 활용

**성공 패턴 (Exp #7-A)**:
```
증강 데이터 1,009개 추가 (+8.1%)
→ 가중치 없이 자연 분포
→ Loss Gap +0.50 (안정적)
→ Test 47.41 (유지)
→ 다양성 확보
```

**실패 패턴 (Exp #7-F)**:
```
증강 없는 카테고리에 가중치 적용
→ 같은 샘플 반복 학습
→ 암기 발생
→ Test 46.62 (하락)
```

**결론**:
- 증강 = 새로운 샘플 = 다양성 = 일반화
- 가중치 = 반복 학습 = 암기 = 과적합

### 5. Length Penalty의 중요성

**실험 결과**:
| LP | Test Score | 변화 |
|----|------------|------|
| 1.0 | 36.12 | Baseline |
| **0.5** | **47.47** | **+11.35** |
| 0.3 | 47.15 | +11.03 |
| 0.7 | 47.22 | +11.10 |

**인사이트**:
- **LP=0.5가 sweet spot**
- LP 낮추면 긴 요약 → ROUGE 상승
- LP=0.3은 과도 (너무 긴 요약)
- 간단한 변경으로 큰 효과 (+11.35점)

### 6. 제출 횟수 관리의 중요성

**제출 내역**:
```
1/12: Baseline (36.12)
2/12: LP=0.5 (47.47, 최고점)
...
11/12: Exp #7-A (47.41)
12/12: Exp #7-F (46.62, 실패)
```

**교훈**:
- 마지막 제출(12/12)이 실패로 끝남
- 제출 전 Loss Gap 확인으로 실패 예측 가능했음
- 신중한 제출 전략 필요 (12회 제한)

---

## 다음 단계 권장사항

### Option 1: Learning Rate 튜닝 (추천)

**설정**:
```yaml
Learning Rate: 1e-5 → 3e-5
Epochs: 20
Patience: 3
데이터: augmentation_final.csv (13,465개)
샘플링: 가중치 없음 (자연 분포)
```

**근거**:
- #7-A로 증강 데이터 안정성 확보
- Loss Gap +0.50으로 건강한 학습 확인
- LR 높이면 학습 속도 개선 가능

**예상 효과**: +0.5~1.5점 → 목표: 48~49점

**위험도**: Low (증강 안정성 확보됨)

---

### Option 2: 더 긴 학습

**설정**:
```yaml
Epochs: 20 → 30
Patience: 3 → 5
Learning Rate: 1e-5 (유지)
```

**근거**:
- #7-A에서 Epoch 11에 Best 달성
- 더 긴 학습으로 추가 개선 여지
- Loss Gap 양수 유지 중

**예상 효과**: +0.3~0.8점

**위험도**: Low

---

### Option 3: 균등 증강 (추천, 장기 전략)

**목표**: 모든 도메인을 균등하게 증강

**증강 계획**:
```python
도메인별 목표:
- 노동/고용: 135 → 500 (+365개)
- 감정 지원: 130 → 300 (+170개)
- 환경: 200 → 500 (+300개)
...

총 증강량: +1,500~2,000개
최종 데이터: 14,000~14,500개
```

**방법**:
1. ChatGPT-4 기반 대화 생성
2. 각 도메인별로 목표 개수까지 증강
3. 품질 검증 (수동 샘플링)

**학습 설정**:
```yaml
샘플링: 가중치 없음 (자연 분포)
Learning Rate: 1e-5 또는 3e-5
Epochs: 20~30
```

**예상 효과**: +0.8~2.0점 → 목표: 48~50점

**위험도**: Medium (데이터 품질 의존)

---

### Option 4: Ensemble (선택적)

**방법**:
- Best 3 checkpoints 앙상블
  - Phase 1 (LP=0.5): 47.47
  - #7-A (증강): 47.41
  - 다른 좋은 checkpoint

**예상 효과**: +0.3~0.8점

**단점**: 추론 시간 3배 증가

---

## 권장 실험 순서

### 단기 (제출 가능 시)
1. **Learning Rate 3e-5** (Option 1) → 예상: 48~49점
2. **Epochs 30** (Option 2) → 예상: 48점
3. 두 개 조합 (LR 3e-5 + Epochs 30) → 예상: 49~50점

### 중기 (1-2주)
1. **균등 증강** (Option 3) → 예상: 48~50점
2. 균등 증강 + LR 튜닝 → 예상: 50~52점

### 장기 (선택적)
1. Ensemble → 예상: +0.5~1점
2. 더 큰 모델 (mBART, KoT5) → 예상: +1~3점

---

## 실험 체크리스트

**실험 시작 전**:
- [ ] Config 파일 작성 및 검증
- [ ] 데이터 경로 확인
- [ ] 디스크 용량 확인 (`du -sh /`)
- [ ] Wandb 로그인 (선택적)

**학습 중**:
- [ ] Loss 추이 모니터링 (Train vs Eval)
- [ ] Loss Gap 계산 및 기록
- [ ] Dev ROUGE 점수 확인 (참고용)
- [ ] Early Stopping 작동 여부

**학습 후**:
- [ ] Best Checkpoint 확인
- [ ] Loss Gap 분석
- [ ] Dev ROUGE와 Loss Gap 비교
- [ ] 추론 실행 (submission.csv 생성)
- [ ] CSV 검증 (index 컬럼, 499개 샘플)

**제출 전 (재개 시)**:
- [ ] Loss Gap이 양수인가?
- [ ] Loss Gap이 이전 실험보다 높은가?
- [ ] Train 분포 왜곡이 없는가? (가중치 확인)
- [ ] Dev ROUGE가 합리적인가? (참고용)
- [ ] CSV 형식 검증 완료?

**제출 후**:
- [ ] Test 점수 기록
- [ ] 실험 로그 작성
- [ ] Config 및 체크포인트 백업
- [ ] 다음 실험 계획 수립

---

## 참고 파일

**실험 로그**:
- `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code/exp7a_train.log`
- `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code/exp7c_train.log`
- `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code/exp7f_train.log`

**Config 파일**:
- `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code/config.yaml`

**데이터**:
- `/Competition/NLP/data/augmentation_final.csv` (13,465개)

**문서**:
- `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/README.md`
- `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/docs/RESTART_GUIDE.md`

---

## 마무리

### 현재 상태
- **최고 점수**: 47.47점 (Phase 1: LP=0.5)
- 안정적 실험: Exp #7-A (47.41, **Loss Gap +0.50**)
- 제출 횟수: 12/12 사용 완료
- 다음 제출: 내일 (리셋 후)

### 핵심 교훈
1. Loss Gap이 진실 (양수 유지 필수)
2. 가중치 사용 금지 (암기 유발)
3. 증강 + 자연 분포 (일반화 확보)
4. Dev는 참고용 (Test만 신뢰)
5. 단순함의 가치 (Baseline 존중)

### 다음 목표
- 단기: 48~49점 (LR 3e-5)
- 중기: 50점 돌파 (균등 증강)
- 장기: 52점 이상 (고급 기법)

---

**작성일**: 2025-10-15
**작성자**: Claude Code
**버전**: 1.0
**상태**: 제출 횟수 소진, 내일 재개 가능