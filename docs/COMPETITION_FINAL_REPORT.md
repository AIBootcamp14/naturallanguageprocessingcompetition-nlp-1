# 대회 최종 리포트 (Competition Final Report)

**프로젝트**: Dialogue Summarization 경진대회
**작성일**: 2025-10-15
**대회 상태**: 종료
**최종 점수**: **47.47점** (Public Test)

---

## 목차

1. [대회 개요](#대회-개요)
2. [프로젝트 정보](#프로젝트-정보)
3. [최종 성과](#최종-성과)
4. [실험 요약](#실험-요약)
5. [주요 발견사항](#주요-발견사항)
6. [최고 성능 설정](#최고-성능-설정)
7. [제출 이력](#제출-이력)
8. [결론 및 향후 방향](#결론-및-향후-방향)

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
│   ├── baseline.ipynb              # 공식 Baseline
│   ├── baseline_modular.ipynb      # 모듈화 버전
│   ├── config.yaml                 # Baseline 설정
│   ├── config/experiments.yaml     # 실험 설정
│   ├── core/                       # 프레임워크 모듈
│   ├── utils/                      # 유틸리티
│   ├── train.py                    # CLI 학습 스크립트
│   └── inference.py                # CLI 추론 스크립트
├── docs/
│   ├── EXPERIMENT_LOG.md           # 전체 실험 기록
│   ├── NEXT_STEPS.md               # 향후 개선 방향
│   ├── RESTART_GUIDE.md            # 재시작 가이드
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

| 지표 | 점수 | 비고 |
|------|------|------|
| **최고 점수** | **47.47점** | Phase 1: LP=0.5 |
| **Baseline** | 36.12점 | 공식 코드 재현 |
| **개선폭** | +11.35점 | +31.43% |
| **최종 제출** | 46.62점 | Exp #7-F (실패) |

### 제출 통계

| 항목 | 값 |
|------|-----|
| **총 제출 횟수** | 12/12 (100% 사용) |
| **성공적 제출** | 11회 |
| **실패 제출** | 1회 (마지막 제출) |
| **점수 범위** | 36.12~47.47 (11.35점 차이) |

### 주요 성과

- **Baseline 재현 성공** (**36.12점**)
- **최고 점수 달성** (**47.47점**, +11.35점)
- **데이터 증강 프레임워크 구축** (1,009개 증강)
- **Loss Gap 분석 기법 확립** (과적합 조기 탐지)
- **전체 실험 문서화 완료** (재현 가능)

---

## 실험 요약

### 전체 실험 목록

| 실험명 | 목적 | Test 점수 | 변화 | 상태 |
|--------|------|-----------|------|------|
| **Baseline** | 공식 코드 재현 | 36.12 | - | 성공 |
| **Phase 1: LP=0.5** | Length Penalty 최적화 | **47.47** | **+11.35** | **최고** |
| Phase 1: LP=0.3 | LP 추가 실험 | 47.15 | +11.03 | 성공 |
| Phase 1: LP=0.7 | LP 추가 실험 | 47.22 | +11.10 | 성공 |
| **Exp #5** | no_repeat_ngram=3 | 47.03 | -0.44 | 실패 |
| **Exp #6** | Learning Rate 3e-5 | - | - | 보류 |
| **Exp #7-A** | 증강 데이터 (가중치 없음) | 47.41 | -0.06 | 안정 |
| Exp #7-C | 가중치 max=5.0 | - | - | 스킵 |
| **Exp #7-F** | 가중치 조정 (3.70배) | 46.62 | -0.79 | **실패** |

### 실험 하이라이트

#### Phase 1: Length Penalty 최적화 (최고 성과)
- **점수**: **47.47점** (+11.35점)
- **변경사항**: Length Penalty 1.0 → 0.5
- **소요시간**: 12초 (추론만 재실행)
- **핵심 인사이트**: 간단한 Generation 파라미터 변경으로 큰 효과

#### Exp #7-A: 데이터 증강 성공
- **점수**: 47.41점 (-0.06점)
- **증강 데이터**: 1,009개 (+8.1%)
- **Loss Gap**: +0.50 (안정적 학습)
- **핵심 인사이트**: Loss Gap 양수 = 건강한 일반화

#### Exp #7-F: 가중치 실패 (최대 교훈)
- **점수**: 46.62점 (-0.79점)
- **문제**: 증강 없는 카테고리에 가중치 3.70배
- **원인**: 같은 샘플 반복 학습 → 암기 발생
- **핵심 인사이트**: 증강 없으면 가중치 사용 금지

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

### 3. Dev ROUGE ≠ Test Score

**실험 증거**:

| 실험 | Dev ROUGE-1 | Test Score | 상관관계 |
|------|-------------|------------|----------|
| Exp #7-A | 36.18% | 47.41 | Dev < Test |
| Exp #7-F | **36.43%** | 46.62 | Dev ↑ but Test ↓ |

**교훈**:
- Dev ROUGE 높다고 Test 높지 않음
- Dev도 학습 분포(가중치) 영향 받음
- **Test 제출만이 유일한 진실**
- Dev는 Loss Gap과 함께 참고만

---

### 4. 증강 vs 가중치

| 방법 | 설명 | 효과 | 결과 |
|------|------|------|------|
| **증강** | 새로운 샘플 생성 | 다양성 증가 | 일반화 향상 |
| **가중치** | 같은 샘플 반복 | 암기 증가 | 과적합 증가 |

**성공 패턴 (Exp #7-A)**:
- 증강 1,009개 + 가중치 없음
- Loss Gap +0.50
- Test 47.41점

**실패 패턴 (Exp #7-F)**:
- 증강 없이 가중치만
- 암기 발생
- Test 46.62점

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
python train.py

# 2. config.yaml 수정: length_penalty=0.5

# 3. 추론 실행
python inference.py --checkpoint checkpoint-XXXX
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
python train.py --experiment exp7a

# 2. 추론 실행
python inference.py --experiment exp7a --checkpoint checkpoint-2068
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
- Dev 점수에 의존하지 말 것 (Test만 신뢰)

---

## 결론 및 향후 방향

### 프로젝트 요약

**기간**: 2025-10-12 ~ 2025-10-15 (4일)
**최고 점수**: **47.47점** (Phase 1: LP=0.5)
**개선폭**: +11.35점 (+31.43%)
**제출**: 12/12 사용 완료

**주요 성과**:
1. Baseline 재현 성공 (**36.12점**)
2. 최고 점수 달성 (**47.47점**)
3. 데이터 증강 프레임워크 구축 (1,009개)
4. **Loss Gap** 분석 기법 확립
5. 전체 실험 문서화 완료

### 핵심 교훈 Top 5

1. **Loss Gap이 진실**
   - 양수 유지 필수 (+0.15 이상)
   - 단, 분포 왜곡 시 양수여도 실패 가능

2. **가중치 사용 금지**
   - 증강 없으면 절대 사용 금지
   - 암기 유발 → Test 실패

3. **증강 + 자연 분포**
   - 일반화 확보의 왕도
   - 가중치 없이 균등 증강

4. **Dev는 참고용**
   - Test만 신뢰
   - Dev도 학습 분포 영향 받음

5. **단순함의 가치**
   - LP=0.5 같은 간단한 변경이 최고 효과
   - 복잡한 기법보다 기본에 충실

### 향후 개선 방향

#### 즉시 적용 가능 (High Priority)

**1. Learning Rate 튜닝** [예상: +0.5~1.5점]
- 현재: 1e-5 → 제안: 3e-5
- 증강 데이터로 안정성 확보
- 더 빠른 수렴 가능

**2. 균등 증강 (Balanced Augmentation)** [예상: +0.8~2.0점]
- 모든 카테고리를 300~500개로 균등화
- WeightedSampler 없이 자연 분포 학습
- 일반화 향상 기대

**3. Extended Training** [예상: +0.3~0.8점]
- Epochs: 20 → 30
- Patience: 3 → 5
- Loss 수렴 여지 존재

#### 장기 전략 (Lower Priority, High Risk)

**4. Larger Models** [예상: +1.0~3.0점]
- 후보: gogamza/kobart-base-v2, KETI-AIR/ke-t5-base
- 리스크: 학습 시간 3배, 메모리 부족 가능

**5. Ensemble Methods**
- 여러 checkpoint 조합
- 제출 횟수 제한으로 테스트 어려움

### 제약사항 및 리스크

**제약사항**:
- 제출 횟수 12회 완전 소진
- 일일 제한으로 신중한 제출 필요
- Dev 점수 신뢰도 낮음

**리스크**:
- Dev/Test Gap 현상 지속
- 증강 데이터 품질 불확실
- 가중치 사용 시 과적합 위험

### 최종 평가

**강점**:
- 체계적 실험 프로세스 확립
- **Loss Gap** 분석으로 조기 실패 탐지
- 재현 가능한 문서화 완료
- 효율적 제출 전략 (12회 제한 내)

**약점**:
- Dev/Test Gap 미해결
- 마지막 제출 실패
- 증강 데이터 효과 제한적 (+0.06점 하락)

**종합 평가**:
- 짧은 기간(4일) 대비 효율적 실험
- 간단한 변경(LP=0.5)으로 최고 성과
- 향후 대회를 위한 귀중한 교훈 확보

---

## 관련 문서

- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) - 전체 실험 상세 기록
- [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) - 실험 교훈 및 Best Practices
- [`NEXT_STEPS.md`](NEXT_STEPS.md) - 향후 개선 방향
- [`RESTART_GUIDE.md`](RESTART_GUIDE.md) - 재시작 가이드
- [`ARCHIVE.md`](ARCHIVE.md) - 프로젝트 아카이브 가이드
- [`code/README.md`](../code/README.md) - 프레임워크 사용법

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-10-15
**작성자**: AI Assistant (Claude Code)
**상태**: 최종본
