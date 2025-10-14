# PRD: 일상 대화 요약 모델 성능 개선 (v2.0)

> **버전**: 2.0 (EDA 심층 분석 반영)
> **최종 업데이트**: 2025-10-14
> **기반**: `/Competition/NLP/docs/EDA.md` v2.0 (전문가 평가 반영)

---

## 1. Introduction/Overview

본 프로젝트는 **일상 대화 한국어 요약 경진대회**에서 현재 Baseline 성능(47점 ROUGE-F1)을 **53.7~60.6점으로 향상**시켜 **1등을 달성**하는 것을 목표로 합니다.

### 핵심 문제
- 이전 세션에서 Dev set 과적합 문제 발생 (Dev 94점 → Test 20점)
- 복잡한 개선 시도로 인한 디버깅 불가능 상태
- CausalLM 방식 실패 (Dev 76.30 → Test 27.21)
- **토큰 제약 초과 문제**: 1.09% (136개) 샘플에서 정보 손실
- **숫자/시간 환각 문제**: 87.88% 샘플에 숫자, 63.35% 샘플에 시간 표현
- **템플릿 과적합**: "~에 대해 이야기했습니다" 526회 반복

### 해결 방안
**"한 번에 하나씩, Test로 검증"** 원칙을 바탕으로 **EDA 기반의 구체적이고 검증된 전략**을 단계적으로 적용하여, 재현 가능하고 데이터 기반의 성능 개선을 달성

---

## 2. Goals

### 주요 목표 🔄 **업데이트**
1. **Baseline 47점 → 53.7~60.6점 달성** (ROUGE-F1 기준) - **목표 상향**
   - Phase 1: 47 → 49.4~51.6점
   - Phase 2: 49.4~51.6 → 51.4~55.8점
   - Phase 3: 51.4~55.8 → 53.7~60.6점
2. **Dev/Test 격차 5점 이내 유지** (과적합 방지)
3. **재현 가능한 파이프라인 구축** (모든 단계 문서화)
4. **제출 횟수 12회 내 효율적 활용** (Daily 제한)

### 부가 목표
5. W&B 기반 실험 추적 시스템 구축
6. KoBART Baseline 안정화 후 CausalLM 방식 재검증
7. Git/문서 자동 최신화 파이프라인 구축
8. 대회 1등 달성

---

## 3. User Stories

### Story 1: 연구자/개발자로서의 실험 관리
```
As a: 대회 참가자 (연구자/개발자)
I want to: 각 개선사항의 효과를 개별적으로 측정하고 추적
So that: 무엇이 성능 향상에 기여했는지 명확히 파악하고 재현할 수 있다
```

### Story 2: 안정적인 성능 개선
```
As a: 대회 참가자
I want to: Dev set 점수를 참고만 하고 Test set으로 최종 검증
So that: 이전처럼 Dev 과적합 문제를 피하고 실전 성능을 정확히 예측할 수 있다
```

### Story 3: 효율적인 제출 관리
```
As a: 대회 참가자
I want to: Dev set으로 빠른 실험 후 3-4개 개선사항마다 Test 제출
So that: 제출 횟수 12회 제한 내에서 최대한 많은 검증을 수행할 수 있다
```

### Story 4: 데이터 기반 의사결정 ⭐ **추가**
```
As a: 연구자
I want to: EDA 분석 결과를 기반으로 우선순위가 높은 전략부터 적용
So that: 제한된 시간과 제출 횟수에서 최대 효과를 얻을 수 있다
```

### Story 5: 자동화된 워크플로우
```
As a: 개발자
I want to: 학습/추론/CSV 생성을 자동화하되 제출 전 수동 검증
So that: 시간을 절약하면서도 에러를 최소화할 수 있다
```

---

## 4. Functional Requirements 🔄 **전면 개편**

### Phase 1: 즉시 실행 ⚡ (예상 소요 시간: 1.5일)

**목표**: Baseline 47점 → **49.4~51.6점**

#### FR1.1: Max Token Length 확장 ⭐ **EDA 반영**

**목적**: 512 토큰 초과 샘플 (136개, 1.09%) 정보 손실 방지

**구현**:
```yaml
# config.yaml 수정
encoder_max_length: 768  # 512 → 768
decoder_max_length: 128  # 100 → 128
```

**근거**:
- Train 136개 샘플에서 truncation 발생 → 대화 뒷부분 정보 완전 손실
- Dev 8개, Test 18개도 512 토큰 초과
- Baseline 예측 분석 결과: 512 토큰 초과 샘플의 ROUGE가 평균보다 15-20% 낮음

**검증**:
- Dev 평가 → 512 토큰 초과 샘플 성능 확인
- Test 제출 (Phase 1 통합)

**예상 효과**: +0.5~1.0 ROUGE

**리스크**: 메모리 1.5배 증가 → batch size 50→32 조정

**참고**: EDA.md Section 2.2, 전략 1

---

#### FR1.2: 제약 디코딩 적용 ⭐⭐ **전문가 최우선 권장, EDA 반영**

**목적**: 숫자/시간/PII 토큰 환각 방지 (87.88% 샘플 영향)

**구현**:
```python
# constrained_decoding.py (새 파일)
import re
import torch
from transformers import LogitsProcessor

class NumberTimeConstrainedLogits(LogitsProcessor):
    """숫자, 시간, PII 토큰만 허용하는 제약 디코딩"""

    def __init__(self, tokenizer, input_text):
        self.tokenizer = tokenizer

        # 입력에서 허용할 패턴 추출
        self.allowed_numbers = set(re.findall(r'\d+', input_text))
        self.allowed_times = set(re.findall(r'\d+시|\d+분|\d+초', input_text))
        self.allowed_pii = set(re.findall(r'#\w+#', input_text))

        # 허용 토큰 ID 변환
        self.allowed_token_ids = self._convert_to_ids()

    def _convert_to_ids(self):
        allowed = []
        for token in self.allowed_numbers | self.allowed_times | self.allowed_pii:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            allowed.extend(ids)
        return set(allowed)

    def __call__(self, input_ids, scores):
        # 허용 토큰 외에는 확률 낮춤 (부분 제약, 완전 차단 금지)
        mask = torch.ones_like(scores) * float('-inf')
        mask[:, list(self.allowed_token_ids)] = 0
        scores = scores + mask * 0.3  # 30% 패널티
        return scores

# baseline.ipynb에 통합
logits_processor = NumberTimeConstrainedLogits(tokenizer, input_text)
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    logits_processor=[logits_processor],  # 추가
    **generation_config
)
```

**근거**:
- 숫자 패턴 87.88%, 시간 패턴 63.35% 출현
- 정확히 복사되어야 하는 factual 정보
- 모델이 환각 발생 가능 (예: "3시" → "4시")
- **Copy Mechanism 대신 즉시 적용 가능** (모델 구조 변경 불필요)

**검증**:
- Dev 평가 → 숫자/시간 정확도 측정
- Test 제출 (Phase 1 통합)

**예상 효과**: +0.5~1.0 ROUGE

**리스크**: 완전 차단 시 필요한 생성 불가 → 부분 제약 (30% 패널티) 적용

**참고**: EDA.md Section 2.3, 전략 4

---

#### FR1.3: 디코딩 길이 정규화 + 커버리지 패널티 ⭐ **전문가 권장, EDA 반영**

**목적**: 적정 길이 유지, 정보 누락 방지

**구현**:
```python
# config.yaml 또는 generation_config 수정
generation_config = {
    "max_length": 128,
    "num_beams": 4,
    "no_repeat_ngram_size": 2,

    # GNMT 길이 정규화 (Google Neural Machine Translation 표준 방식)
    "length_penalty": 0.6,  # α=0.6 (0.6~1.0 실험)
    # score = log_prob / ((5 + length) / 6) ** α

    # 커버리지 관련
    "repetition_penalty": 1.2,  # 반복 억제
    "diversity_penalty": 0.5,   # 다양성 촉진 (beam group)

    "early_stopping": True
}

# baseline.ipynb에서 적용
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    **generation_config
)
```

**근거**:
- 압축 비율 편차 큼 (1.01~19.85배) → 너무 짧거나 긴 요약 방지 필요
- GNMT 표준 방식으로 검증됨
- 정보 누락 문제 개선 (커버리지 패널티)

**검증**:
- Dev 평가 → 요약 평균 길이 변화 확인
- Test 제출 (Phase 1 통합)

**예상 효과**: +0.3~0.8 ROUGE

**참고**: EDA.md Section 2.5, 전략 5

---

#### FR1.4: Learning Rate 최적화

**목적**: 빠른 수렴, 학습 속도 향상

**구현**:
```yaml
# config.yaml
learning_rate: 3e-5  # 1e-5 → 3e-5

# Cosine annealing with warmup
scheduler_type: "cosine"
warmup_ratio: 0.1
num_cycles: 0.5
```

**근거**:
- 현재 1e-5는 안전하지만 느린 학습
- 토큰 길이 분포가 넓고 압축 비율 편차가 큼 → adaptive learning 필요

**검증**:
- Dev 평가 → Loss 수렴 속도 확인
- Test 제출 (Phase 1 통합)

**예상 효과**: +0.3~0.8 ROUGE

**리스크**: 3e-5에서 발산 가능 → Warmup ratio 0.1 유지, gradient clipping 강화 (1.0)

**참고**: EDA.md 전략 2

---

#### FR1.5: Hard Sample Mining & Oversampling ⭐ **EDA 반영**

**목적**: 복합 문제 샘플 (683개, 5.48%) 집중 개선

**구현**:
```python
# prepare_hard_samples.py (새 파일)
import pandas as pd

def is_hard_sample(row):
    """복합 문제 샘플 판별"""
    return (row['dialogue_tokens'] > 512 or
            row['compression_ratio'] < 3.18 or
            row['dialogue_len'] > 700)

# 1. Hard sample 판별
train_df = pd.read_csv('data/train.csv')
hard_samples = train_df[train_df.apply(is_hard_sample, axis=1)]
easy_samples = train_df[~train_df.apply(is_hard_sample, axis=1)]

print(f"Hard samples: {len(hard_samples)} ({len(hard_samples)/len(train_df)*100:.2f}%)")
print(f"Easy samples: {len(easy_samples)}")

# 2. Hard sample 2배 oversampling (2배 이상 금지)
train_dataset = pd.concat([
    easy_samples,
    hard_samples,
    hard_samples  # 중복
]).sample(frac=1, random_state=42)  # 셔플

train_dataset.to_csv('data/train_hard_sampled.csv', index=False)
print(f"Total samples: {len(train_dataset)}")
```

**근거**:
- 복합 문제 샘플 683개가 성능 저하의 주범
- Baseline 예측에서 저성능 샘플의 특징: 긴 대화 + 낮은 압축 비율 + 512 토큰 초과

**검증**:
- Dev 평가 → Hard sample 성능 개선 확인
- Test 제출 (Phase 1 통합)

**예상 효과**: +0.8~1.5 ROUGE

**리스크**: 어려운 샘플 과적합 → 쉬운 샘플 성능 저하 가능 → validation 면밀히 모니터링

**참고**: EDA.md Section 2.8, 전략 3

---

#### Phase 1 실행 방법

```bash
# 1. 디스크 용량 확인
du -sh /Competition/NLP

# 2. Git 저장소로 이동
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1

# 3. config.yaml 수정 (FR1.1, FR1.3, FR1.4)
# encoder_max_length: 768
# decoder_max_length: 128
# learning_rate: 3e-5
# generation_config 업데이트

# 4. Hard sample 준비 (FR1.5)
python prepare_hard_samples.py

# 5. 제약 디코딩 구현 (FR1.2)
# baseline.ipynb에 constrained_decoding.py 통합

# 6. 학습 실행
jupyter notebook code/baseline.ipynb

# 7. Test 제출 및 검증
```

**Phase 1 예상 총 효과**: **+2.4~4.6 ROUGE** (47 → 49.4~51.6점)

---

### Phase 2: 단기 🚀 (예상 소요 시간: 1-2주)

**목표**: 49.4~51.6점 → **51.4~55.8점**

#### FR2.1: Longer Training ⭐ **EDA 반영**

**목적**: 충분한 학습 시간 확보

**구현**:
```yaml
# config.yaml
num_train_epochs: 30  # 20 → 30
early_stopping_patience: 5  # 3 → 5
```

**근거**:
- 현재 20 epochs, patience=3
- 복잡한 데이터셋이므로 더 긴 학습 가능

**검증**:
- Dev 평가 → Loss 수렴 확인
- Test 제출 (Phase 2 통합)

**예상 효과**: +0.3~0.7 ROUGE

**참고**: EDA.md 전략 8

---

#### FR2.2: IWCV 분석 및 적용 ⭐⭐ **전문가 권장, EDA 반영**

**목적**: Dev/Test 분포 괴리 완화, 12회 제출 제한 환경 최적화

**구현**:
```python
# iwcv_analysis.py (새 파일)
from sklearn.linear_model import LogisticRegression
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import numpy as np

def extract_features(df):
    """Dev/Test 분포 차이 분석을 위한 특성 추출"""
    features = df[[
        'dialogue_len',
        'summary_len',
        'dialogue_tokens',
        'summary_tokens',
        'compression_ratio',
        'num_speakers'  # 필요 시 추가
    ]].values
    return features

# 1. Dev/Test 분포 차이 분석
dev_df = pd.read_csv('data/dev.csv')
test_df = pd.read_csv('data/test.csv')  # Label 없음, feature만 사용

# Feature 추출
dev_features = extract_features(dev_df)
test_features = extract_features(test_df)

# Dev(0), Test(1) 레이블로 분류기 학습
features = np.vstack([dev_features, test_features])
labels = [0] * len(dev_df) + [1] * len(test_df)

clf = LogisticRegression(max_iter=1000)
clf.fit(features, labels)

# 2. Dev 샘플 가중치 계산 (Test 분포와 유사할수록 높은 가중치)
dev_weights = clf.predict_proba(dev_features)[:, 1]  # Test일 확률

# 3. Weighted sampling
sampler = WeightedRandomSampler(
    weights=dev_weights,
    num_samples=len(dev_df),
    replacement=True
)

# baseline.ipynb에 통합
# train_dataloader = DataLoader(
#     dev_dataset,
#     batch_size=50,
#     sampler=sampler  # weighted sampling
# )

print(f"Dev weights: min={dev_weights.min():.3f}, max={dev_weights.max():.3f}, mean={dev_weights.mean():.3f}")
```

**근거**:
- RESTART_GUIDE.md에서 Dev/Test gap 문제 심각성 확인
- 12회 제출 제한 → 효율적 검증 방법 필수
- **Importance-Weighted Cross-Validation** (JMLR 논문 검증)

**검증**:
- Phase 1 완료 후 IWCV 분석 수행
- Dev/Test gap 변화 모니터링
- Test 제출 (Phase 2 통합)

**예상 효과**: 직접적 ROUGE 향상 없음, **예측 안정성 향상**

**참고**: EDA.md 전략 13

---

#### FR2.3: LED 모델 탐색 및 실험 ⭐⭐ **전문가 최우선 권장 (조건부), EDA 반영**

**조건**: Phase 1의 Max Length 768 효과 불충분 시

**목적**: 512 토큰 초과 샘플 (136개, 1.09%) 완벽 처리

**구현**:
```python
# led_model.py (새 파일, 조건부)
from transformers import LEDForConditionalGeneration, LEDTokenizer

# 1. 한국어 LED 모델 탐색
# HuggingFace에서 "LED Korean" 검색
# 예: "klue/led-base-korean" (존재 여부 확인 필요)

# 2. 없으면 영어 LED 모델에 한국어 fine-tuning
model_name = "allenai/led-base-16384"  # 16,384 토큰 지원
model = LEDForConditionalGeneration.from_pretrained(model_name)
tokenizer = LEDTokenizer.from_pretrained(model_name)

# 3. Global attention 설정 (중요 토큰 핀)
import torch

def set_global_attention(input_ids, tokenizer):
    """중요 토큰에 global attention 설정"""
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1  # [CLS]

    # #Person#, 시간, 숫자 등 중요 토큰 위치 찾기
    for i in range(input_ids.size(1)):
        token = tokenizer.decode(input_ids[:, i])
        if '#' in token or any(char.isdigit() for char in token):
            global_attention_mask[:, i] = 1

    return global_attention_mask

# 학습 시 적용
global_attention_mask = set_global_attention(input_ids, tokenizer)
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    global_attention_mask=global_attention_mask,
    labels=labels
)
```

**근거**:
- 512 토큰 초과 1.09% 샘플 → 정보 손실 방지
- LED는 **16,384 토큰**까지 효율적 처리 (window + global attention)
- mBART/KoT5보다 장문 처리 최적

**검증**:
1. Phase 1 완료 후 Max Length 768 효과 검증
2. 불충분 시 한국어 LED 모델 탐색
3. 512 토큰 초과 샘플 성능 개선 확인
4. Test 제출 (Phase 2 통합)

**예상 효과**: +1.0~2.0 ROUGE

**리스크**: 한국어 모델 없으면 1-2주 소요

**참고**: EDA.md 전략 14

---

#### FR2.4: Dynamic Batch Size 구현

**목적**: GPU 효율 증가, 긴 샘플 처리 최적화

**구현**:
```python
# dynamic_batch.py (새 파일)

def dynamic_batch_size(num_tokens):
    """토큰 길이 기준 동적 배치 크기"""
    if num_tokens < 256:
        return 64
    elif num_tokens < 512:
        return 50
    else:
        return 32

# config.yaml에 gradient accumulation 추가
# gradient_accumulation_steps: 2
# effective_batch_size: 100  # 50 * 2
```

**근거**:
- 긴 샘플과 짧은 샘플이 섞여 GPU 활용도 비효율
- 현재 batch size 50은 짧은 샘플에 최적화

**검증**:
- GPU 메모리 사용량 모니터링
- Dev 평가 → Test 제출 (Phase 2 통합)

**예상 효과**: +0.2~0.5 ROUGE

**참고**: EDA.md 전략 7

---

#### FR2.5: Unlikelihood Training 실험 ⭐ **전문가 권장 (실험적), EDA 반영**

**목적**: 템플릿 과적합 해결 ("~에 대해 이야기했습니다" 526회 반복)

**구현**:
```python
# unlikelihood_trainer.py (새 파일, 실험적)
import torch
import torch.nn.functional as F

class UnlikelihoodTrainer:
    """N-gram 반복 억제 학습"""

    def __init__(self, model, alpha=0.05):
        self.model = model
        self.alpha = alpha  # unlikelihood weight (0.05~0.1, 절대 0.1 초과 금지)

    def compute_loss(self, logits, labels, prev_ngrams):
        """MLE loss + Unlikelihood loss"""
        # 1. 기본 cross-entropy loss
        mle_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        # 2. Unlikelihood loss (이미 생성된 n-gram 억제)
        ul_loss = 0
        for ngram in prev_ngrams:
            ngram_prob = torch.exp(logits[:, ngram])
            ul_loss += torch.log(1 - ngram_prob + 1e-8)

        ul_loss = -ul_loss.mean()

        # 3. 결합
        total_loss = mle_loss + self.alpha * ul_loss
        return total_loss

# 학습 시 적용
trainer = UnlikelihoodTrainer(model, alpha=0.05)

for batch in train_dataloader:
    outputs = model(**batch)
    logits = outputs.logits
    labels = batch['labels']

    # 이미 생성된 tri-gram 추적
    prev_ngrams = extract_trigrams(outputs.sequences)

    loss = trainer.compute_loss(logits, labels, prev_ngrams)
    loss.backward()
```

**근거**:
- "~에 대해 이야기했습니다" 526회 반복 (Tri-gram top 1)
- 템플릿 과적합 문제 심각 → 정보 누락 가능성
- **Unlikelihood Loss** (Welleck et al., 2019 arXiv)

**검증**:
- Alpha 0.05로 시작 (절대 0.1 초과 금지)
- Dev ROUGE 점진적 모니터링 (epoch마다)
- 2 epoch 연속 저하 시 즉시 롤백
- A/B 테스트: Unlikelihood 有/無 병행 학습
- Test 제출 (Phase 2 통합)

**예상 효과**: +0.3~0.6 ROUGE (불확실)

**리스크**: Alpha 과다 시 ROUGE 저하

**참고**: EDA.md Section 2.7, 전략 15

---

#### FR2.6: 첫 Test 제출

- Phase 1+2 누적 효과 확인
- Dev와 Test 점수 gap 분석 (IWCV 효과 검증)

**Phase 2 예상 총 효과**: **+2.0~4.2 ROUGE** (49.4~51.6 → 51.4~55.8점)

---

### Phase 3: 중장기 🎯 (예상 소요 시간: 1-2주)

**목표**: 51.4~55.8점 → **53.7~60.6점**

#### FR3.1: Ensemble 전략 구축 ⭐⭐ **최고 효과**

**목적**: 여러 checkpoint 조합으로 최대 성능 달성

**구현**:
```python
# ensemble.py (새 파일)
import torch
from transformers import BartForConditionalGeneration

# 1. 여러 epoch checkpoint 저장
top_checkpoints = [
    'checkpoints/epoch_15',
    'checkpoints/epoch_18',
    'checkpoints/epoch_22'
]

# 2. Top-3 checkpoint 앙상블
predictions = []
for model_path in top_checkpoints:
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to('cuda')
    model.eval()

    pred = model.generate(input_ids, **generation_config)
    predictions.append(pred)

# 3. ROUGE 기반 선택 (각 샘플마다 최고 ROUGE 예측 선택)
from rouge import Rouge
rouge = Rouge()

final_summaries = []
for i in range(len(test_dataset)):
    sample_preds = [tokenizer.decode(pred[i], skip_special_tokens=True) for pred in predictions]

    # Dev set 기준 최고 ROUGE 모델 선택 (또는 다수결)
    best_pred = max(sample_preds, key=lambda x: rouge.get_scores(x, reference[i])['rouge-l']['f'])
    final_summaries.append(best_pred)
```

**근거**:
- 단일 모델의 한계 극복
- 여러 checkpoint 조합으로 안정성 향상

**검증**:
- Dev 평가 → Ensemble vs Single model 비교
- Test 제출 (Phase 3 통합)

**예상 효과**: +1.5~3.0 ROUGE (가장 큰 효과)

**리스크**: 추론 시간 3배 → submission 파일 생성 느림 (최종 제출에만 사용)

**참고**: EDA.md 전략 10

---

#### FR3.2: Copy Mechanism 정밀 구현 (조건부) 🔄 **우선순위 하향**

**조건**: 제약 디코딩 효과 불충분 시에만 구현

**목적**: 숫자/시간/PII 토큰 정확 복사

**구현**:
```python
# Pointer-Generator Network 구현 (복잡)
# 1. Encoder attention 추가
# 2. Copy probability 계산
# 3. Generation vs Copy 선택 메커니즘

# 복잡도 높음 - 제약 디코딩 효과 불충분 시에만 고려
```

**근거**:
- 제약 디코딩으로 80% 문제 해결 가능
- 모델 구조 변경 필요 (Pointer-Generator) → 구현 비용 높음 (1주일)

**검증**:
- 제약 디코딩 후 숫자/시간 정확도 확인
- 불충분 시에만 구현 착수
- Test 제출 (Phase 3 통합)

**예상 효과**: 제약 디코딩 후 추가 개선분만 +0.3~0.8 ROUGE

**참고**: EDA.md 전략 6 (우선순위 하향)

---

#### FR3.3: Paraphrase Augmentation

**목적**: 데이터 다양성 증가, 표현 획일화 문제 해결

**구현**:
```python
# paraphrase_augment.py (새 파일)

# 1. Back-translation으로 paraphrase 생성
# 한국어 → 영어 → 한국어

# 2. Summary를 다양하게 재작성
augmented_data = []
for row in train_df.iterrows():
    original = row['summary']
    paraphrased = paraphrase_model(original)  # Solar Mini 또는 Llama-3.2-Korean-3B
    augmented_data.append({
        'dialogue': row['dialogue'],
        'summary': paraphrased
    })

# 3. 규정 준수 (DialogSum 미사용)
# 4. 2-3배 증강 (12,457 → 25,000~37,000)
```

**근거**:
- 요약 표현이 너무 획일화
- 다양한 표현 학습 필요

**검증**:
- 소량 테스트 (1,000개) → 효과 검증 → 전체 증강
- Dev 평가 → Test 제출 (Phase 3 통합)

**예상 효과**: +0.5~1.0 ROUGE

**참고**: EDA.md 전략 11

---

#### FR3.4: Model Upgrade (조건부) 🔄 **조건 변경**

**조건**: LED 효과 불충분 시에만 mBART/KoT5 시도

**목적**: 더 큰 모델로 성능 천장 높이기

**구현**:
```python
# Option 1: mBART-large (611M)
model_name = "facebook/mbart-large-cc25"

# Option 2: KoT5-large (770M)
model_name = "KETI-AIR/ke-t5-large"

# Option 3: LED + Ensemble 조합 (최우선)
```

**근거**:
- KoBART는 작은 모델 (123M parameters)
- LED Phase 2에서 우선 적용 → Model Upgrade 필요성 감소

**검증**:
- Phase 2까지 효과 확인 후 결정
- Dev 평가 → Test 제출 (Phase 3 통합)

**예상 효과**: +1.0~2.5 ROUGE

**참고**: EDA.md 전략 12

---

**Phase 3 예상 총 효과**: **+2.3~4.8 ROUGE** (51.4~55.8 → 53.7~60.6점)

---

### Phase 4: W&B 추적 시스템

#### FR4.1: W&B 기본 설정
- 프로젝트: `dialogue-summarization`
- Entity: 팀/개인 계정
- Run naming: `{date}_{experiment_name}_{lr}`

#### FR4.2: 추적 메트릭
**기본 메트릭**:
- Train/Val Loss
- ROUGE-1, ROUGE-2, ROUGE-L (F1/Precision/Recall)
- Learning Rate (epoch별)
- Gradient Norm

**상세 메트릭**:
- 샘플 예측 결과 (Table 형식, 각 epoch마다 5개)
- 하이퍼파라미터 비교 대시보드
- Confusion Matrix (길이별 성능)
- **숫자/시간 정확도** (제약 디코딩 효과 측정) ⭐ **추가**

#### FR4.3: 체크포인트 관리
- Best model (ROUGE 기준)
- Last checkpoint
- Epoch별 체크포인트 (save_total_limit=5)

---

### Phase 5: 자동화 파이프라인

#### FR5.1: 학습 자동화 스크립트
```python
# scripts/train.py
- Config 로드
- 데이터 전처리 (Hard sample mining 포함)
- 모델 학습 (W&B 추적)
- Best model 저장
- 결과 로깅
```

#### FR5.2: 추론 자동화 스크립트
```python
# scripts/inference.py
- Best model 로드
- Test 데이터 추론 (제약 디코딩 적용)
- CSV 생성 (,fname,summary 형식)
- 검증 (499 samples, index 포함)
```

#### FR5.3: 수동 검증 체크리스트
- [ ] CSV 포맷 확인 (`head -5 prediction/output.csv`)
- [ ] 샘플 수 확인 (`wc -l prediction/output.csv` = 500)
- [ ] Index 컬럼 존재 확인
- [ ] 특수 토큰 제거 확인
- [ ] 수동으로 5개 샘플 확인
- [ ] **숫자/시간 정확도 확인** ⭐ **추가**

---

### Phase 6: Git/문서 자동 최신화

#### FR6.1: Git 자동 커밋 (선택적)
- 실험 완료 시 자동 커밋 옵션
- 커밋 메시지: `Experiment #{N}: {description} - {score}`
- 수동 확인 후 Push

#### FR6.2: 실험 로그 자동 업데이트
- `experiment_logs.md`에 자동 추가
- 형식: 실험 번호, 변경사항, 설정, 결과, 다음 단계

#### FR6.3: README 업데이트
- Best score 기록
- 성능 개선 그래프 (선택적)

---

### Phase 7: CausalLM 재검증 (Baseline 50점 달성 후)

#### FR7.1: 검증된 전처리 적용
- Phase 2-3에서 검증된 전처리를 Korean_DCS_2024에 적용
- 노이즈 처리, 특수 토큰 최적화, 제약 디코딩

#### FR7.2: 하이퍼파라미터 신중한 튜닝
- Learning rate: 2e-5 기준으로 작은 범위 탐색
- Epoch: 5 → 3, 7 실험
- Batch size/Gradient accumulation 조정

#### FR7.3: A/B 테스트
- KoBART best model vs CausalLM best model
- Dev 평가 → 격차 분석 → 유망 시 Test 제출

---

### Phase 8: 최종 제출

#### FR8.1: 최종 모델 선택
- Dev/Test 격차 5점 이내
- Test 53.7점 이상 목표 ⭐ **업데이트**
- 재현 가능성 확인

#### FR8.2: 제출 파일 생성
- 최종 CSV 생성 및 다중 검증
- sample_submission.csv와 포맷 일치 확인

#### FR8.3: 코드 정리
- 재현 스크립트 작성
- 요구사항 문서 업데이트
- 트러블슈팅 가이드 작성

---

## 5. Non-Goals (Out of Scope) 🔄 **업데이트**

### 명시적으로 제외되는 항목

❌ **Phase 1 완료 전 복잡한 개선 시도**
- Baseline 47점 달성 전 LLM, 앙상블, 새 모델 시도 금지

❌ **한 번에 여러 개선사항 적용**
- 반드시 하나씩 테스트 (디버깅 가능성 유지)

❌ **Dev set 점수만 보고 만족**
- Test 검증 없이 진행 금지

❌ **DialogSum 데이터셋 사용**
- 직접/간접 사용 모두 금지 (대회 규정)

❌ **유료 API 사용**
- Solar 제외, 다른 유료 API 금지

❌ **평가 데이터를 학습에 활용**
- 분석만 가능, Label 생성 학습 금지

❌ **완전 자동화 제출**
- 제출 전 반드시 수동 검증

❌ **CheckList 프레임워크 (대회 중 보류)** ⭐ **새 추가**
- 전문가 평가: 대회 중 구현 시간 대비 효과 낮음

❌ **MTLD/MATTR 지표 (ROUGE 외 평가)** ⭐ **새 추가**
- 대회 평가 지표: ROUGE-F1만 사용

---

## 6. Design Considerations 🔄 **보강**

### 6.1 디렉토리 구조
```
/Competition/NLP/
├── naturallanguageprocessingcompetition-nlp-1/  # Git 저장소
│   ├── code/
│   │   ├── baseline.ipynb
│   │   ├── config.yaml
│   │   ├── constrained_decoding.py          # 새로 추가 ⭐
│   │   └── requirements.txt
│   ├── scripts/                               # 새로 추가
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── prepare_hard_samples.py           # 새로 추가 ⭐
│   │   ├── iwcv_analysis.py                  # 새로 추가 ⭐
│   │   ├── led_model.py                      # 새로 추가 (조건부) ⭐
│   │   ├── unlikelihood_trainer.py           # 새로 추가 (실험적) ⭐
│   │   ├── ensemble.py                       # 새로 추가 ⭐
│   │   └── paraphrase_augment.py
│   ├── checkpoints/
│   ├── prediction/
│   └── logs/
├── docs/
│   ├── EDA.md                                 # v2.0 (전문가 평가 반영)
│   ├── experiment_logs.md                     # 새로 추가
│   └── ...
└── tasks/
    ├── prd-dialogue-summarization-performance-improvement.md  # 본 문서 (v2.0)
    └── tasks-prd-dialogue-summarization.md                    # 생성 예정
```

### 6.2 실험 로그 템플릿
```markdown
## 실험 #N: [실험명]

**날짜**: 2025-10-__
**베이스**: Baseline / 실험 #(N-1)

### 변경사항
- [한 가지만 명시]

### 설정
```yaml
[변경된 파라미터만]
```

### 결과
- Baseline/이전: XX.XX
- 현재 Dev: XX.XX
- 현재 Test: XX.XX (제출한 경우)
- **Dev/Test 격차**: XX.XX
- **변화**: +X.XX ✅/❌
- **숫자/시간 정확도**: XX.XX% ⭐ 추가 (제약 디코딩 적용 시)

### 판단
- [유지/롤백/재시도]
- [이유]

### 다음 단계
- [다음에 시도할 것]
```

### 6.3 W&B 대시보드 구성
- **Overview**: 전체 실험 비교 (ROUGE 점수)
- **Training**: Loss, Learning rate, Gradient norm
- **Evaluation**: ROUGE-1/2/L, Dev/Test 격차
- **Samples**: 예측 결과 샘플 (5개)
- **Hyperparameters**: 파라미터 비교 테이블
- **숫자/시간 정확도**: 제약 디코딩 효과 측정 ⭐ **추가**

### 6.4 제약 디코딩 설계 ⭐ **새 추가**

**핵심 원칙**:
- 부분 제약 (30% 패널티) 적용, 완전 차단 지양
- 허용 토큰 외에도 생성 가능하도록 유연성 유지

**알고리즘**:
1. 입력 대화에서 숫자/시간/PII 패턴 추출
2. 허용 토큰 ID 리스트 생성
3. Generation 시 LogitsProcessor로 허용 토큰 외 확률 낮춤 (30% 패널티)
4. Beam search와 결합하여 최적 요약 생성

### 6.5 IWCV 알고리즘 개요 ⭐ **새 추가**

**핵심 원칙**:
- Dev/Test 분포 차이를 학습으로 완화
- 12회 제출 제한 환경에서 안정적 성능 예측

**알고리즘**:
1. Dev/Test 특성 추출 (길이, 압축비, 토큰 수 등)
2. Logistic Regression으로 Dev(0) vs Test(1) 분류
3. Dev 샘플의 Test 유사도 계산 (확률)
4. WeightedRandomSampler로 학습 시 가중치 적용

### 6.6 LED 모델 전환 전략 ⭐ **새 추가**

**전환 조건**:
- Phase 1 완료 후 Max Length 768 효과 검증
- 512 토큰 초과 샘플 성능 개선 불충분 시

**전환 단계**:
1. 한국어 LED 모델 탐색 (HuggingFace)
2. 없으면 영어 LED (allenai/led-base-16384) fine-tuning
3. Global attention 설정 (#Person#, 시간, 숫자 등)
4. Baseline과 A/B 테스트

---

## 7. Technical Considerations 🔄 **업데이트**

### 7.1 환경
- GPU: RTX 3090 24GB
- Python: 3.10
- PyTorch: 2.5.1
- Transformers: 4.46.3 (Baseline)

### 7.2 주요 의존성
```
transformers==4.46.3
rouge==1.0.1
wandb==0.16.1
pandas==2.1.4
torch==2.5.1
tqdm==4.66.1
scikit-learn==1.3.2  # IWCV 분석 ⭐ 추가
```

### 7.3 디스크 용량 관리
- ⚠️ **150GB 제한 절대 준수**
- 모든 run 전 `du -sh / 2>/dev/null` 확인
- 체크포인트: save_total_limit=5
- 예측 결과: 이전 버전 삭제

### 7.4 제출 횟수 관리
- Daily 12회 제한
- 전략: Dev로 3-4개 실험 → Test 1회
- 추적: 스프레드시트/문서로 제출 이력 관리

### 7.5 Dev/Test 격차 모니터링
- 목표: 격차 5점 이내
- 경고: 격차 10점 이상 시 과적합 의심
- 조치: 정규화, Dropout, Early stopping 강화, **IWCV 적용** ⭐ **추가**

### 7.6 재현성 보장
- Random seed: 42 (고정)
- 모든 설정 YAML/config 파일로 관리
- Git commit hash 기록
- 환경 정보 저장 (`pip freeze > requirements_frozen.txt`)

### 7.7 LED 모델 요구사항 ⭐ **새 추가**
- 메모리: 16,384 토큰 지원 → 메모리 약 2배 증가
- Batch size: 32 → 16 감소 필요
- Global attention: 중요 토큰 위치 설정 필수

### 7.8 Unlikelihood Training 구현 고려사항 ⭐ **새 추가**
- Alpha: 0.05로 시작, 절대 0.1 초과 금지
- 모니터링: Epoch마다 Dev ROUGE 확인
- 롤백: 2 epoch 연속 저하 시 즉시 중단
- A/B 테스트: Unlikelihood 有/無 병행 학습

### 7.9 제약 디코딩 성능 최적화 ⭐ **새 추가**
- 허용 토큰 리스트 사전 계산 (배치 단위)
- LogitsProcessor 오버헤드 최소화
- Beam search와 결합 시 성능 영향 모니터링

---

## 8. Success Metrics 🔄 **조정**

### 8.1 주요 성공 지표

**Metric 1: Test ROUGE-F1 점수** ⭐ **업데이트**
- 목표: **53.7점 이상** (최종 60.6점 목표)
- 측정: 대회 플랫폼 제출
- 빈도: 3-4 실험마다 1회

**Metric 2: Dev/Test 격차**
- 목표: **5점 이내**
- 측정: Dev 점수 - Test 점수 (절대값)
- 빈도: 매 Test 제출 시

**Metric 3: 제출 효율성**
- 목표: **12회 제출 내 53.7점 달성**
- 측정: 제출 횟수 추적
- 평가: 제출당 평균 개선폭 (+0.5점 이상)

**Metric 4: 숫자/시간 정확도** ⭐ **새 추가**
- 목표: **95% 이상**
- 측정: 생성 요약의 숫자/시간이 원본 대화와 일치하는 비율
- 빈도: 제약 디코딩 적용 후 매 평가 시

### 8.2 부가 성공 지표

**Metric 5: 재현성**
- 목표: 동일 설정으로 ±0.5점 이내 재현
- 측정: 3회 재학습 후 표준편차
- 빈도: 최종 모델 선정 시

**Metric 6: 실험 속도**
- 목표: 1회 실험(학습+평가) 30분 이내
- 측정: W&B run time
- 개선: 불필요한 로깅 제거, 배치 최적화

**Metric 7: 문서화 완성도**
- 목표: 모든 실험 로그 기록
- 측정: `experiment_logs.md` 항목 수
- 평가: 누락 없음

**Metric 8: Phase별 중간 목표** ⭐ **새 추가**
- Phase 1: 49.4~51.6점
- Phase 2: 51.4~55.8점
- Phase 3: 53.7~60.6점

### 8.3 최종 목표
- **대회 순위**: 1등 (또는 상위 3위)
- **학습 성과**: 재현 가능한 성능 개선 파이프라인 구축
- **지식 축적**: 트러블슈팅 가이드 및 베스트 프랙티스 문서화

---

## 9. Open Questions 🔄 **업데이트**

### Q1: LED 한국어 모델 존재 여부? ⭐ **새 추가**
- 한국어 사전학습 LED 모델이 HuggingFace에 있는지?
- 없으면 영어 LED fine-tuning (1-2주 소요) 또는 Longformer-BART 대안
- → 답변: Phase 1 완료 후 탐색

### Q2: Unlikelihood Alpha 최적값? ⭐ **새 추가**
- 0.05 vs 0.08 vs 0.1 중 선택?
- ROUGE 저하 리스크 관리 방법?
- → 답변: 0.05로 시작, A/B 테스트로 결정

### Q3: 제약 디코딩 완벽성 vs 유연성 균형? ⭐ **새 추가**
- 30% 패널티가 최적인지?
- 완전 차단 vs 부분 제약 실험 필요?
- → 답변: Phase 1에서 실험

### Q4: 하이퍼파라미터 탐색 범위
- Learning rate: 3e-5 외에 2e-5, 5e-5도 시도?
- Epoch: 30 외에 40까지 늘려볼지?
- → 답변: Phase 2.1 결과 보고 결정

### Q5: 데이터 증강 규모
- 2배 vs 3배 증강 중 선택?
- 증강 품질 검증 방법?
- → 답변: 소량 테스트(1,000개) 후 결정

### Q6: CausalLM 재시도 조건
- KoBART 53.7점 달성 시 무조건 시도?
- 시간 부족 시 스킵?
- → 답변: 대회 마감 7일 전까지 53.7점 달성 시 시도

### Q7: 앙상블 전략
- KoBART + CausalLM 앙상블 고려?
- 규칙 기반 후처리 + 모델 조합?
- → 답변: 53.7점 달성 후 검토

### Q8: W&B 공개 설정
- Public 프로젝트로 설정? (다른 참가자 볼 수 있음)
- Private 유지?
- → 답변: Private 유지 (대회 종료 후 공개 고려)

### Q9: Git 브랜치 전략
- main: 안정 버전
- experiment: 실험용 브랜치 분리?
- → 답변: 단순하게 main만 사용 (각 실험은 commit으로 관리)

### Q10: 비상 시나리오
- 53.7점 달성 실패 시 대안?
  - A: 51점 안정화 + 코드 품질 향상
  - B: 앙상블로 52-53점 목표
  - C: 완전히 새로운 접근 (mBART 등)
- → 답변: A 우선 (재현 가능성 최우선)

---

## 10. Implementation Timeline (예상) 🔄 **재조정**

### Week 1: Foundation + Phase 1 (1.5일)
- **Day 1-2**: Phase 1 실행 (Max Length, 제약 디코딩, 길이 정규화, Learning Rate, Hard Sample Mining)
  - 예상 효과: 47 → 49.4~51.6점
  - Test 제출 (1회)
- **Day 3-4**: Phase 1 결과 분석 및 Phase 2 준비
  - IWCV 분석 수행
  - LED 모델 탐색 (필요 시)

### Week 2: Phase 2 (1-2주)
- **Day 5-8**: Longer Training, IWCV 적용, Dynamic Batch Size
- **Day 9-12**: LED 모델 실험 (조건부, 한국어 모델 존재 시)
- **Day 13-14**: Unlikelihood Training 실험 (실험적, A/B 테스트)
  - 예상 효과: 49.4~51.6 → 51.4~55.8점
  - Test 제출 (2-3회)

### Week 3: Phase 3 + Finalization (1-2주)
- **Day 15-17**: Ensemble 전략 구축 (최우선)
- **Day 18-19**: Paraphrase Augmentation
- **Day 20-21**: Copy Mechanism (조건부), Model Upgrade (조건부)
  - 예상 효과: 51.4~55.8 → 53.7~60.6점
  - Test 제출 (2-3회)

### Week 4: Advanced & Finalization
- **Day 22-23**: CausalLM 재검증 (조건부, 53.7점 달성 시)
- **Day 24-25**: 최종 모델 선정 및 검증
- **Day 26-28**: 코드 정리, 문서 완성, 최종 제출

---

## Appendix: 체크리스트

### 실험 전 체크리스트
- [ ] 디스크 용량 150GB 미만 확인
- [ ] Git 최신 상태 확인
- [ ] 이전 실험 로그 작성 완료
- [ ] Config 파일 백업
- [ ] 제출 횟수 확인 (12회 이내)
- [ ] **EDA 기반 우선순위 확인** ⭐ **추가**

### 실험 중 체크리스트
- [ ] W&B run 시작 확인
- [ ] GPU 사용률 모니터링
- [ ] Loss 수렴 확인
- [ ] 샘플 예측 결과 확인
- [ ] **숫자/시간 정확도 모니터링** (제약 디코딩 적용 시) ⭐ **추가**

### 실험 후 체크리스트
- [ ] Dev ROUGE 점수 기록
- [ ] Dev/Test 격차 분석 (Test 제출 시)
- [ ] 실험 로그 작성
- [ ] Best model 체크포인트 확인
- [ ] Git 커밋 (변경사항 설명)
- [ ] 다음 실험 계획 수립
- [ ] **Phase별 목표 달성 여부 확인** ⭐ **추가**

### Test 제출 전 체크리스트
- [ ] CSV 포맷 검증 (,fname,summary)
- [ ] 샘플 수 확인 (500 lines = header + 499)
- [ ] Index 컬럼 존재
- [ ] 특수 토큰 제거 확인
- [ ] 5개 샘플 수동 검토
- [ ] **숫자/시간 정확도 확인** (원본 대화와 일치 여부) ⭐ **추가**
- [ ] Dev 점수와 비교 분석
- [ ] 제출 횟수 체크

### Phase별 완료 체크리스트 ⭐ **새 추가**

**Phase 1 완료 체크**:
- [ ] Max Length 768 적용 완료
- [ ] 제약 디코딩 구현 완료
- [ ] 길이 정규화 적용 완료
- [ ] Learning Rate 3e-5 적용 완료
- [ ] Hard Sample Mining 적용 완료
- [ ] Test 제출 완료 (목표: 49.4~51.6점)
- [ ] 효과 검증 완료 (LED 필요성 판단)

**Phase 2 완료 체크**:
- [ ] Longer Training 적용 완료
- [ ] IWCV 분석 및 적용 완료
- [ ] LED 모델 실험 완료 (조건부)
- [ ] Dynamic Batch Size 적용 완료
- [ ] Unlikelihood Training 실험 완료 (실험적)
- [ ] Test 제출 완료 (목표: 51.4~55.8점)

**Phase 3 완료 체크**:
- [ ] Ensemble 전략 구축 완료
- [ ] Paraphrase Augmentation 적용 완료
- [ ] Copy Mechanism 구현 완료 (조건부)
- [ ] Model Upgrade 실험 완료 (조건부)
- [ ] Test 제출 완료 (목표: 53.7~60.6점)

---

**작성일**: 2025-10-12
**최종 업데이트**: 2025-10-14
**버전**: 2.0 (EDA 심층 분석 반영)
**작성자**: Claude Code + User
**기반 문서**: `/Competition/NLP/docs/EDA.md` v2.0
**다음 단계**: `generate-tasks.md`로 상세 Task List 생성

---

## 변경 이력 (v2.0)

### 주요 변경사항
1. ⭐ **5개 새 전략 추가** (전문가 권장):
   - FR1.2: 제약 디코딩 (Phase 1)
   - FR1.3: 디코딩 길이 정규화 (Phase 1)
   - FR2.2: IWCV 분석 (Phase 2)
   - FR2.3: LED 모델 (Phase 2, 조건부)
   - FR2.5: Unlikelihood Training (Phase 2, 실험적)

2. 🔄 **목표 점수 상향**: 50점 → **53.7~60.6점**

3. 🔄 **Phase별 구조 개편**:
   - Phase 1: 5개 전략 (기존 3개 → 5개)
   - Phase 2: 6개 전략 (IWCV, LED, Unlikelihood 추가)
   - Phase 3: Copy Mechanism 우선순위 하향 (조건부)

4. ⭐ **Design Considerations 3개 섹션 추가**:
   - 6.4: 제약 디코딩 설계
   - 6.5: IWCV 알고리즘 개요
   - 6.6: LED 모델 전환 전략

5. 🔄 **Non-Goals 2개 추가**:
   - CheckList 프레임워크 (보류)
   - MTLD/MATTR 지표 (제외)

6. ⭐ **Success Metrics 업데이트**:
   - Metric 1: 50점 → **53.7점** 목표
   - Metric 4: 숫자/시간 정확도 (새 추가)
   - Metric 8: Phase별 중간 목표 (새 추가)

7. 🔄 **Open Questions 3개 추가**:
   - Q1: LED 한국어 모델 존재 여부
   - Q2: Unlikelihood Alpha 최적값
   - Q3: 제약 디코딩 완벽성 vs 유연성

8. 🔄 **Implementation Timeline 재조정**:
   - Week 1: 1.5일 (Phase 1 확장)
   - Week 2: Phase 2 (1-2주)
   - Week 3-4: Phase 3 + Finalization

### 유지된 내용
- Section 1, 3: Introduction, User Stories (핵심 유지)
- "한 번에 하나씩, Test로 검증" 원칙
- Baseline 재현 필수 (Phase 1 시작)
- 실험 로그 시스템 (Section 6.2)
- W&B 추적 시스템 (Phase 4)
- 자동화 파이프라인 (Phase 5)
- Git/문서 최신화 (Phase 6)
- CausalLM 재검증 (Phase 7, 조건부)
- 제출 체크리스트 (Appendix)