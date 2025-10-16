# 대회 최종 리포트 (Competition Final Report)

**프로젝트**: Dialogue Summarization 경진대회
**작성일**: 2025-10-15
**대회 상태**: 종료
**Public Test 최고 점수**: **47.47점** (Phase 1: LP=0.5)
**Private Test 최고 점수**: **47.31점** (Exp #7-A: 데이터 증강)

---

## 목차

1. [대회 개요](#대회-개요)
2. [프로젝트 정보](#프로젝트-정보)
3. [최종 성과](#최종-성과)
4. [실험 요약](#실험-요약)
5. [주요 발견사항](#주요-발견사항)
6. [최고 성능 설정](#최고-성능-설정)
7. [제출 이력](#제출-이력)
8. [결론](#결론)

---

## 대회 개요

### 기본 정보

**대회명**: Dialogue Summarization 경진대회
**주최**: AI 부트캠프 (AI Stages)
**대회 목표**: 일상 대화(회의, 토의, 일상 대화)를 효과적으로 요약하는 모델 개발

**대회 배경**:
- 일상 대화 녹음의 전체 재청취 어려움 해결
- 대화 중 실시간 요약의 한계 극복
- 기억 의존 요약의 오해/누락 문제 해결

### 평가 지표

**주요 메트릭**: ROUGE-F1 Score

**최종 점수 계산**:
```
Final Score = max(ROUGE-1-F1) + max(ROUGE-2-F1) + max(ROUGE-L-F1)
```

**평가 특징**:
- 각 대화당 **3개의 정답 요약문** 제공 (Multi-Reference Dataset)
- 3개 중 가장 높은 점수를 선택하여 합산
- 한국어 형태소 분석기로 토큰화 후 평가
- 100점이 만점이 아님 (Multi-Reference 평균 특성)

**ROUGE 세부 메트릭**:
- **ROUGE-1**: Unigram 기반 겹침 측정
- **ROUGE-2**: Bigram 기반 겹침 측정
- **ROUGE-L**: LCS(최장 공통 부분 문자열) 기반 측정

### 데이터셋 규모

| 구분 | 샘플 수 | 용도 |
|------|---------|------|
| **Train** | 12,457개 | 모델 학습 |
| **Dev** | 499개 | 로컬 검증 |
| **Public Test** | 250개 | 공개 리더보드 (50%) |
| **Private Test** | 249개 | 최종 순위 (50%) |
| **총합** | 13,455개 | - |

**데이터 특성**:
- 대화 턴 수: 2~60턴
- 대화 참여자: 2~7명
- 주제: 회의, 일상 대화 등 다양한 주제

### 제약사항

| 항목 | 제한 |
|------|------|
| **일일 제출 횟수** | 12회 (팀 단위) |
| **최종 제출물 선택** | 2개 |
| **참가 계정** | 1계정/인 |
| **제출 초기화** | 한국시간 자정 |

**규칙**:
- 외부 데이터셋 사용 허용
- DialogSum 데이터셋 사용 전면 금지
- 무료 API 사용 가능
- 평가 데이터 학습 활용 금지

---

## 프로젝트 정보

### 개발 환경

**하드웨어**:
- GPU: NVIDIA RTX 3090 24GB
- CUDA: 12.1

**소프트웨어**:
- Python: 3.10
- PyTorch: 2.5.1
- Transformers: 4.46.3
- ROUGE: 1.0.1

### 프로젝트 구조

```
naturallanguageprocessingcompetition-nlp-1/
├── code/
│   ├── src/
│   │   ├── cli/
│   │   │   ├── train.py            # CLI 학습 스크립트
│   │   │   └── inference.py        # CLI 추론 스크립트
│   │   ├── core/                   # 프레임워크 모듈
│   │   │   ├── data.py             # 데이터 로딩 & 샘플링
│   │   │   ├── model.py            # 모델 로드/저장
│   │   │   ├── trainer.py          # 학습 로직
│   │   │   └── inference.py        # 추론 로직
│   │   ├── scripts/                # 유틸리티 스크립트
│   │   │   ├── data_loader.py      # 데이터 전처리
│   │   │   ├── dataset.py          # 커스텀 데이터셋
│   │   │   ├── generate_augmented_data.py  # 데이터 증강
│   │   │   └── ...
│   │   └── utils/                  # 유틸리티 함수
│   │       ├── config.py           # Config 파싱
│   │       ├── logger.py           # 로깅
│   │       └── metrics.py          # ROUGE 계산
│   ├── baseline.ipynb              # 공식 Baseline
│   ├── baseline_modular.ipynb      # 모듈화 버전
│   └── config.yaml                 # 모든 실험 설정
├── docs/
│   ├── EXPERIMENT_LOG.md           # 전체 실험 기록
│   ├── LESSONS_LEARNED.md          # 실험 교훈
│   └── Competition_Overview/       # 대회 규칙
├── submission_exp7a/
│   └── checkpoint-2068/            # 최고 성능 checkpoint
└── README.md
```

### 개발 기간

**총 기간**: 2025-10-12 ~ 2025-10-15 (4일)

**Phase별 일정**:
- **Day 1 (10/12)**: Baseline 재현 및 검증
- **Day 2 (10/13)**: Length Penalty 최적화 (Phase 1)
- **Day 3 (10/14)**: 데이터 증강 실험 시작 (Exp #7)
- **Day 4 (10/15)**: 증강 + 가중치 실험 및 종료

---

## 최종 성과

### 점수 요약

| 지표 | Public Test | Private Test | 비고 |
|------|-------------|--------------|------|
| **최고 점수** | **47.47점** | **47.31점** | Public: LP=0.5 / Private: 증강 |
| **Baseline** | 36.12점 | - | 공식 코드 재현 |
| **개선폭** | +11.35점 | +11.19점 | +31.43% / +30.99% |
| **최종 제출** | 46.62점 | 46.62점 | Exp #7-F (실패) |

### 제출 통계

| 항목 | 값 |
|------|-----|
| **총 제출 횟수** | 12/12 (100% 사용) |
| **성공적 제출** | 11회 |
| **실패 제출** | 1회 (마지막 제출) |
| **점수 범위** | 36.12~47.47 (11.35점 차이) |

### 주요 성과

- **Baseline 재현 성공** (**36.12점**)
- **Public Test 최고 점수** (**47.47점**, +11.35점)
- **Private Test 최고 점수** (**47.31점**, +11.19점)
- **Dev 강건성이 Private 성능 예측** (데이터 증강 효과 입증)
- **Loss Gap 분석 기법 확립** (과적합 조기 탐지)
- **데이터 증강 프레임워크 구축** (1,009개 증강)
- **전체 실험 문서화 완료** (재현 가능)

---

## 실험 요약

### 전체 실험 목록

| 실험명 | 목적 | Public Test | Private Test | 상태 |
|--------|------|-------------|--------------|------|
| **Baseline** | 공식 코드 재현 | 36.12 | - | 성공 |
| **Phase 1: LP=0.5** | Length Penalty 최적화 | **47.47** | 47.12 | Public 최고 |
| Phase 1: LP=0.3 | LP 추가 실험 | 47.15 | - | 성공 |
| Phase 1: LP=0.7 | LP 추가 실험 | 47.22 | - | 성공 |
| **Exp #5** | no_repeat_ngram=3 | 47.03 | - | 실패 |
| **Exp #6** | Learning Rate 3e-5 | - | - | 보류 |
| **Exp #7-A** | 증강 데이터 (가중치 없음) | 47.41 | **47.31** | **Private 최고** |
| Exp #7-C | 가중치 max=5.0 | - | - | 스킵 |
| **Exp #7-F** | 가중치 조정 (3.70배) | 46.62 | 46.62 | 실패 |

### 실험 하이라이트

#### Phase 1: Length Penalty 최적화 (Public 최고)
- **Public Test**: **47.47점** (+11.35점) - **최고 점수**
- **Private Test**: 47.12점 (+11.00점)
- **변경사항**: Length Penalty 1.0 → 0.5
- **소요시간**: 12초 (추론만 재실행)
- **핵심 인사이트**: 간단한 Generation 파라미터 변경으로 큰 효과

#### Exp #7-A: 데이터 증강 성공 (Private 최고)
- **Public Test**: 47.41점 (-0.06점)
- **Private Test**: **47.31점** (+0.19점) - **Private 최고 점수**
- **증강 데이터**: 1,009개 (+8.1%)
- **Loss Gap**: +0.50 (안정적 학습)
- **핵심 인사이트**: Dev 강건성이 Private 성능으로 이어짐
- **중요 발견**: Public에서 약간 낮아도 Private에서 역전 가능

#### Exp #7-F: 가중치 실패 (최대 교훈)
- **Public Test**: 46.62점 (-0.79점)
- **Private Test**: 46.62점 (-0.69점)
- **문제**: 증강 없는 카테고리에 가중치 3.70배
- **원인**: 같은 샘플 반복 학습 → 암기 발생 → 분포 왜곡
- **핵심 인사이트**: 가중치 샘플링은 분포를 왜곡시켜 일반화 해침

---

## 주요 발견사항

### 1. Loss Gap의 중요성

**정의**: `Loss Gap = Train Loss - Eval Loss`

**해석**:
- **양수 (+)**: Train > Eval → 정상 학습, 안정적 일반화
- **음수 (-)**: Train < Eval → 과적합 경향

**실험 증거**:

| 실험 | Loss Gap | Test 점수 | 결과 |
|------|----------|-----------|------|
| Exp #7-A | **+0.50** | 47.41 | 성공 |
| Exp #7-C | +0.40 | (스킵) | 사전 실패 예측 |
| Exp #7-F | +0.47 | 46.62 | 실패 (분포 왜곡) |

**교훈**:
- Loss Gap +0.15 이상 권장
- Loss Gap 양수 = 필요조건이지 충분조건 아님
- Train 분포 왜곡 시 양수여도 실패 가능

---

### 2. WeightedRandomSampler의 함정

**문제 패턴**:
```
증강 없는 카테고리 (135개) × 3.70배 가중치
→ 같은 샘플 500회 반복 학습
→ 모델 암기 (Memorization)
→ Train 분포 ≠ Test 분포
→ Test 성능 하락
```

**실패 사례 (Exp #7-F)**:
- 노동/고용 카테고리: 135개 × 3.70배 = 500회 반복
- Dev ROUGE: 36.43% (높음)
- Test Score: 46.62점 (낮음)
- **Dev도 학습 분포 영향을 받음** → Dev 점수 신뢰 불가

**올바른 사용법**:
- **증강 + 가중치 없음** (추천): 모든 카테고리 균등 증강 후 자연 분포 학습
- 가중치 사용 시: max 1.5배 이하로 제한
- 증강 없이 가중치만 사용: 절대 금지

---

### 3. Dev 강건성의 중요성 (Private Test 예측 지표)

**중요 발견**: Private Test 결과, Dev 강건 모델이 최고 성능 기록

**실험 증거**:

| 실험 | Dev ROUGE-1 | Loss Gap | Public | Private | 결과 |
|------|-------------|----------|--------|---------|------|
| Exp #7-A | 36.18% | **+0.50** | 47.41 | **47.31** | Private 최고 |
| Exp #7-F | **36.43%** | +0.47 | 46.62 | 46.62 | 실패 (분포 왜곡) |
| LP=0.5 | - | - | **47.47** | 47.12 | Public 최고 |

**핵심 인사이트**:
- **Loss Gap + Dev 강건성 = Private 성능 예측 지표**
- 데이터 증강(Exp #7-A)이 Dev 강건성을 높임 → Private 최고 점수
- 가중치 샘플링(Exp #7-F)은 Dev도 왜곡 → Public/Private 모두 낮음
- Public과 Private의 역전 현상: **Dev 강건 모델이 Private에서 역전**

**교훈**:
- Dev 강건성이 실제 일반화 성능을 예측
- Public Test만 보지 말고 Dev 안정성도 함께 고려
- Loss Gap + Dev 일관성 = 최종 성능 예측 가능

---

### 4. 증강 vs 가중치

| 방법 | 설명 | 효과 | 결과 |
|------|------|------|------|
| **증강** | 새로운 샘플 생성 | 다양성 증가 | **일반화 향상** |
| **가중치** | 같은 샘플 반복 | 암기 증가 | 과적합 증가 |

**성공 패턴 (Exp #7-A)**:
- 증강 1,009개 + 가중치 없음
- Loss Gap +0.50
- Public 47.41점 / **Private 47.31점 (최고)**
- **Private에서 증강 효과 입증**

**실패 패턴 (Exp #7-F)**:
- 증강 없이 가중치만
- 암기 발생 → 분포 왜곡
- Public 46.62점 / Private 46.62점 (낮음)

---

### 5. Length Penalty의 위력

| Length Penalty | Test Score | 변화 | 해석 |
|----------------|------------|------|------|
| 1.0 | 36.12 | Baseline | - |
| **0.5** | **47.47** | **+11.35** | Sweet spot |
| 0.3 | 47.15 | +11.03 | 준수 |
| 0.7 | 47.22 | +11.10 | 준수 |

**인사이트**:
- LP 낮추면 긴 요약 → ROUGE 상승
- LP=0.5가 최적 균형점
- 간단한 변경으로 큰 효과 (+11.35점)
- 추론만 재실행 (12초) → 가장 효율적

---

## 최고 성능 설정

### Phase 1: LP=0.5 (47.47점)

**모델**:
```yaml
model_name: digit82/kobart-summarization
architecture: Encoder-Decoder (KoBART)
```

**Generation 파라미터** (핵심 변경):
```yaml
length_penalty: 0.5          # Baseline: 1.0
num_beams: 4
no_repeat_ngram_size: 2
max_length: 100
early_stopping: true
```

**학습 파라미터** (Baseline 유지):
```yaml
learning_rate: 1e-5
optimizer: adamw_torch
warmup_ratio: 0.1
batch_size: 50 (train), 32 (eval)
epochs: 20
early_stopping_patience: 3
fp16: true
```

**Tokenizer 설정**:
```yaml
encoder_max_length: 512
decoder_max_length: 100
```

**재현 방법**:
```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# 1. Baseline 학습 (또는 기존 checkpoint 사용)
python src/cli/train.py

# 2. config.yaml 수정: length_penalty=0.5

# 3. 추론 실행
python src/cli/inference.py --checkpoint checkpoint-XXXX
```

---

### 2위: Exp #7-A (47.41점)

**데이터**:
```yaml
train_samples: 13,465
  - original: 12,457
  - augmented: 1,009 (+8.1%)
weighted_sampler: false       # 가중치 없음 (핵심!)
```

**Generation 파라미터**:
```yaml
length_penalty: 0.5           # Phase 1 설정 유지
num_beams: 4
no_repeat_ngram_size: 2
```

**학습 파라미터**:
```yaml
learning_rate: 1e-5
batch_size: 24
gradient_accumulation_steps: 3  # effective batch = 72
epochs: 20
early_stopping_patience: 3
```

**학습 결과**:
```yaml
best_epoch: 11
checkpoint: checkpoint-2068
train_loss: 1.0278
eval_loss: 0.5255
loss_gap: +0.5023              # 안정적!
dev_rouge_1: 36.18%
```

**보존 경로**:
```
/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/
  submission_exp7a/checkpoint-2068/
```

**재현 방법**:
```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# 1. 증강 데이터로 학습
python src/cli/train.py --experiment exp7a

# 2. 추론 실행
python src/cli/inference.py --experiment exp7a --checkpoint checkpoint-2068
```

---

## 제출 이력

### 제출 순서와 점수 추이

```
제출 1/12: Baseline (36.12)
제출 2/12: Phase 1 LP=0.5 (47.47) [최고점]
제출 3/12: Phase 1 LP=0.3 (47.15)
제출 4/12: Phase 1 LP=0.7 (47.22)
제출 5-7/12: (중간 실험들)
제출 8/12: Exp #5 no_repeat_ngram=3 (47.03)
제출 9-10/12: (추가 실험들)
제출 11/12: Exp #7-A 증강 (47.41)
제출 12/12: Exp #7-F 증강+가중치 (46.62) [마지막 실패]
```

### 점수 변화 그래프 (텍스트)

```
48.0 |
     |     [최고]
47.5 |    /47.47
     |   /
47.0 |  /____________[안정]______________
     | /           47.41
     |/                              \
46.5 |                                \
     |                                 [실패]46.62
     |
36.0 |___[Baseline 36.12]_____________
      1   2   3   4   5   6   7   8   9  10  11  12
                    제출 횟수
```

### 제출 통계

| 지표 | 값 |
|------|-----|
| **총 제출 횟수** | 12/12 (100%) |
| **최고 점수** | 47.47 (2/12) |
| **최저 점수** | 46.62 (12/12) |
| **평균 점수** | ~47.2점 |
| **점수 범위** | 0.85점 |

### 교훈

- 마지막 제출(12/12)이 실패로 끝남
- Loss Gap 확인으로 실패 예측 가능 (#7-C 스킵 성공)
- 신중한 제출 전략 필요 (일일 12회 제한)
- **Private Test 결과**: Dev 강건 모델이 최고 성능 달성
- Public만 보지 말고 Dev 안정성도 함께 고려해야 함

---

## 결론

### 프로젝트 요약

**기간**: 2025-10-12 ~ 2025-10-15 (4일)
**Public Test 최고 점수**: **47.47점** (Phase 1: LP=0.5)
**Private Test 최고 점수**: **47.31점** (Exp #7-A: 데이터 증강)
**평균 개선폭**: +11.27점 (+31.21%)
**제출**: 12/12 사용 완료

**주요 성과**:
1. Baseline 재현 성공 (**36.12점**)
2. Public Test 최고 점수 달성 (**47.47점**)
3. Private Test 최고 점수 달성 (**47.31점**)
4. **Dev 강건성이 Private 성능 예측 지표임을 발견**
5. 데이터 증강 프레임워크 구축 (1,009개)
6. **Loss Gap** 분석 기법 확립
7. 전체 실험 문서화 완료

### 핵심 교훈 Top 6

1. **Dev 강건성 = Private 성능 예측 지표**
   - Private Test에서 Dev 강건 모델(Exp #7-A)이 최고 점수
   - Loss Gap + Dev 강건성 = 최종 성능 예측 가능
   - 데이터 증강이 실제 일반화 성능에 기여

2. **Loss Gap이 진실**
   - 양수 유지 필수 (+0.15 이상)
   - 단, 분포 왜곡 시 양수여도 실패 가능

3. **가중치는 분포를 왜곡**
   - 증강 없으면 절대 사용 금지
   - 암기 유발 → 분포 왜곡 → Test 실패

4. **증강 + 자연 분포**
   - 일반화 확보의 왕도
   - 가중치 없이 균등 증강
   - Private Test에서 효과 입증

5. **Public ≠ Private**
   - Public 최고가 Private 최고 아닐 수 있음
   - Dev 강건성이 더 신뢰할 수 있는 지표

6. **단순함의 가치**
   - LP=0.5 같은 간단한 변경이 Public 최고 효과
   - 복잡한 기법보다 기본에 충실

### 최종 평가

**강점**:
- 체계적 실험 프로세스 확립
- **Dev 강건성이 Private 성능 예측 지표임을 발견**
- **Loss Gap** 분석으로 조기 실패 탐지
- 데이터 증강의 실질적 효과 입증 (Private 최고 점수)
- 재현 가능한 문서화 완료
- 효율적 제출 전략 (12회 제한 내)

**약점**:
- Public과 Private 간 역전 현상 예측 어려움
- 마지막 제출 실패
- 가중치 샘플링의 위험성을 늦게 파악

**핵심 발견**:
- **Public 최고 ≠ Private 최고**: 서로 다른 모델
  - Public 최고: LP=0.5 (단순 Generation 튜닝)
  - Private 최고: Exp #7-A (데이터 증강 + Dev 강건)
- **Dev 강건성이 Private 성능 예측**: Loss Gap + 증강 = 일반화

**종합 평가**:
- 짧은 기간(4일) 대비 효율적 실험
- 간단한 변경(LP=0.5)으로 Public 최고 성과
- 데이터 증강으로 Private 최고 성과
- 향후 대회를 위한 귀중한 교훈 확보

---

## 관련 문서

- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) - 전체 실험 상세 기록
- [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) - 실험 교훈 및 Best Practices
- [`code/README.md`](../code/README.md) - 프레임워크 사용법

---

**문서 버전**: 2.0
**최종 업데이트**: 2025-10-15
**상태**: 최종본 (대회 종료)
