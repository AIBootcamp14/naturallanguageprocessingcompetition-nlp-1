# 참조 구현 분석 및 개선 방안

> 작성일: 2025-10-14
> 목적: 다양한 구현 사례 분석을 통한 현재 모듈화 시스템 개선 방안 도출

---

## 1. 개요

여러 구현 사례를 분석하여 현재 모듈화 시스템에 적용 가능한 개선 사항을 파악하였다. 각 구현의 핵심 설정값, 전략, 기법들을 추출하고 성능 향상에 기여할 수 있는 요소들을 식별하였다.

---

## 2. 분석 대상

### 2.1 KBH 구현 분석

**파일**: `notebooks/team/KBH/eda_notebook.ipynb`

#### 주요 특징

- **종합적인 EDA (Exploratory Data Analysis)**
  - 13가지 분석 항목을 체계적으로 수행
  - 데이터 구조, 텍스트 길이, 주제 분포, 화자 분석, 발화 분석 등

- **토큰 길이 분석 (KoBART Tokenizer 기준)**
  - 512 토큰 초과 샘플: 1.09% (136개)
  - 100 토큰 초과 Summary: 0.39% (48개)
  - 평균 문자/토큰 비율: 2.03 (한국어)

- **특수 패턴 빈도 분석**
  - 시간, 금액, 전화번호, 주소 등의 패턴 식별
  - 패턴별 출현 빈도 및 샘플 비율 통계

- **성능 문제 샘플 식별**
  - 긴 대화 (상위 10%)
  - 복잡한 대화 (화자 3명 이상)
  - 512 토큰 초과 샘플
  - 낮은 압축 비율 샘플 (정보 밀도 높음)

#### 적용 가능한 개선 사항

1. **토큰 길이 기반 샘플 필터링**
   - 512 토큰 초과 샘플에 대한 특별 처리 전략 필요
   - Truncation 전략 개선 또는 Sliding Window 기법 적용 고려

2. **특수 패턴 기반 전처리 강화**
   - 시간, 금액 등 고빈도 패턴에 대한 정규화 규칙 추가
   - 패턴별 특수 토큰 추가 검토

3. **데이터 품질 메트릭 도입**
   - Type-Token Ratio (TTR) 기반 어휘 다양성 평가
   - 압축 비율 분석을 통한 요약 품질 검증

4. **성능 문제 샘플 집중 학습**
   - 긴 대화, 복잡한 구조 샘플에 대한 가중치 증가
   - 또는 별도의 Fine-tuning 단계 추가

---

### 2.2 JSH 구현 분석

**파일**: `notebooks/team/JSH/find_topic_categories.py`, `find_topic_categories_result.txt`

#### 주요 특징

- **Semantic Topic Clustering**
  - 총 9,235개의 고유 주제를 12개 도메인으로 분류
  - 키워드 기반 도메인 매칭 전략

- **도메인 분류 결과**
  ```
  기타: 6,215 (46.2%)
  쇼핑/구매: 1,998 (14.8%)
  여행/교통: 1,079 (8.0%)
  음식/식사: 848 (6.3%)
  여가/취미: 577 (4.3%)
  학습/교육: 509 (3.8%)
  가족/친구: 466 (3.5%)
  고객서비스: 401 (3.0%)
  건강/의료: 400 (3.0%)
  업무/직장: 376 (2.8%)
  면접/취업: 373 (2.8%)
  금융/은행: 215 (1.6%)
  ```

- **데이터 증강 우선순위**
  - HIGH 우선순위: 금융/은행 (1.6%)
  - MEDIUM 우선순위: 여가/취미, 학습/교육, 가족/친구, 고객서비스, 건강/의료, 업무/직장, 면접/취업

#### 적용 가능한 개선 사항

1. **도메인별 증강 전략**
   - 저빈도 도메인 (금융/은행 등)에 대한 집중적인 데이터 증강
   - 도메인별 타겟 증강 비율 설정
     - 금융/은행: 최대 2,018개 생성 (총 비율 50% 달성)
     - 중간 빈도 도메인: 각 672개 생성

2. **도메인 기반 Stratified Sampling**
   - K-Fold 분할 시 도메인 비율 유지
   - 도메인별 균등 분포 보장

3. **Topic-aware Data Augmentation**
   - 각 도메인의 특성을 반영한 증강 전략
   - 도메인별 키워드 사전 활용

4. **미분류 주제 재검토**
   - 기타 46.2% 샘플에 대한 재분류 시도
   - 새로운 도메인 카테고리 추가 고려

---

### 2.3 KSM 구현 분석

**파일**: `notebooks/team/KSM/baseline.py`, `config.yaml`, `generation.py`, `preprocess.py`

#### 주요 설정값 (config.yaml)

```yaml
tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100
  special_tokens:
    - '#Person1#'
    - '#Person2#'
    - '#Person3#'
    - '#PhoneNumber#'
    - '#Address#'
    - '#PassportNumber#'

training:
  num_train_epochs: 20
  learning_rate: 1.0e-05
  per_device_train_batch_size: 50  # ⭐ 큰 배치 크기
  per_device_eval_batch_size: 32
  warmup_ratio: 0.1
  weight_decay: 0.01
  lr_scheduler_type: 'cosine'
  optim: 'adamw_torch'
  fp16: true
  early_stopping_patience: 3
  early_stopping_threshold: 0.001

inference:
  num_beams: 7  # ⭐ 높은 빔 개수
  length_penalty: 1.0
  no_repeat_ngram_size: 3
  repetition_penalty: 1.4  # ⭐ 강한 반복 억제
  min_new_tokens: 12
  max_new_tokens: 60
  early_stopping: true
  batch_size: 32
```

#### 핵심 전략

1. **생성 파라미터 최적화**
   - `num_beams: 7` (높은 빔 서치)
   - `repetition_penalty: 1.4` (강력한 반복 억제)
   - `max_new_tokens: 60` (적절한 길이 제한)

2. **전처리 모듈화**
   - `preprocess.py`: 별도 전처리 로직 분리
   - 화자 정규화 및 텍스트 정제 체계화

3. **생성 설정 유틸리티**
   - `generation.py`: 생성 파라미터 빌더 함수 제공
   - 재사용 가능한 설정 관리

#### 적용 가능한 개선 사항

1. **배치 크기 최적화**
   - 현재 모듈화: `per_device_train_batch_size: 50` (동일)
   - GPU 메모리 허용 시 배치 크기 증가 고려

2. **빔 서치 파라미터 조정**
   - 현재 모듈화: `num_beams: 4-5`
   - KSM 설정: `num_beams: 7`
   - **권장**: 5-7 범위에서 Optuna 탐색 범위 확장

3. **반복 억제 강화**
   - 현재 모듈화: `repetition_penalty: 1.5`
   - KSM 설정: `repetition_penalty: 1.4`
   - **권장**: 1.4-1.6 범위 유지 (현재 설정 적절)

4. **길이 제어 전략**
   - KSM: `min_new_tokens: 12, max_new_tokens: 60`
   - 현재 모듈화: `min_new_tokens: 30, max_new_tokens: 100`
   - **검토**: 최소 토큰 12로 낮추면 더 짧은 요약 허용 가능

---

## 3. 현재 모듈화 구현과의 비교

### 3.1 강점

1. **체계적인 설정 관리**
   - 계층적 YAML 구조 (base → models → experiments)
   - 설정 오버라이드 및 병합 지원

2. **포괄적인 실험 옵션**
   - PRD 14 기반 통합 명령행 인터페이스
   - 다양한 모드 지원 (single, kfold, multi_model, optuna, full)

3. **Optuna 최적화 적용**
   - 자동 하이퍼파라미터 탐색
   - 최적 설정값 저장 및 재사용

4. **WandB 로깅 통합**
   - 실험 추적 및 시각화
   - 메트릭 자동 기록

### 3.2 개선 필요 영역

#### A. 데이터 분석 및 품질 관리

**현재 상태**:
- 기본적인 데이터 로딩 및 전처리만 구현
- 데이터 품질 검증은 선택적 기능

**개선 방안**:
1. EDA 모듈 통합
   - KBH 구현의 13가지 분석 항목 적용
   - 자동 데이터 프로파일링 기능 추가

2. 토큰 길이 기반 필터링
   - 512 토큰 초과 샘플 자동 감지
   - Sliding Window 또는 청킹 전략 적용

3. 특수 패턴 전처리
   - 시간, 금액 등 패턴 정규화 규칙 추가
   - 도메인별 전처리 파이프라인 구축

#### B. Topic/Domain 기반 전략

**현재 상태**:
- Topic 정보 활용이 제한적
- 균등 샘플링 위주

**개선 방안**:
1. 도메인 클러스터링 모듈 추가
   - JSH의 12개 도메인 분류 체계 적용
   - `src/data/topic_classifier.py` 생성

2. 도메인별 증강 전략
   - 저빈도 도메인 집중 증강
   - 도메인별 타겟 비율 설정 지원

3. Stratified K-Fold
   - 도메인 비율을 유지하는 분할 전략
   - `src/validation/stratified_kfold.py` 추가

#### C. 생성 파라미터 최적화

**현재 상태**:
```yaml
# configs/models/kobart.yaml
inference:
  num_beams: 4
  repetition_penalty: 1.5
  min_new_tokens: 30
  max_new_tokens: 100
```

**KSM 참조 설정**:
```yaml
inference:
  num_beams: 7
  repetition_penalty: 1.4
  min_new_tokens: 12
  max_new_tokens: 60
```

**개선 방안**:
1. **빔 서치 확장**
   - Optuna 탐색 범위: 4-7로 확장
   - 성능 vs 속도 트레이드오프 분석

2. **길이 제어 유연화**
   - 최소 토큰 12-30 범위 지원
   - 요약 길이 요구사항에 따라 조정 가능하도록

3. **반복 억제 Fine-tuning**
   - 1.2-1.6 범위에서 도메인별 최적값 탐색

---

## 4. 구체적인 개선 액션 아이템

### 4.1 즉시 적용 가능 (High Priority)

#### 1. 생성 파라미터 조정
**파일**: `configs/models/kobart.yaml`

```yaml
inference:
  num_beams: 5  # 4 → 5 (속도와 품질 균형)
  min_new_tokens: 20  # 30 → 20 (짧은 요약 허용)
  repetition_penalty: 1.4  # 1.5 → 1.4 (KSM 설정)
```

**근거**: KSM 구현에서 높은 성능 확인

---

#### 2. Optuna 탐색 범위 확장
**파일**: `src/optimization/optuna_tuner.py`

```python
# num_beams 범위 확장
trial.suggest_int('num_beams', 4, 7)  # 기존: 3-5

# min_new_tokens 범위 확장
trial.suggest_int('min_new_tokens', 12, 40)  # 기존: 20-40

# repetition_penalty 범위 조정
trial.suggest_float('repetition_penalty', 1.2, 1.6)  # 기존: 1.0-2.0
```

---

#### 3. Special Tokens 확장
**파일**: `configs/base/encoder_decoder.yaml`

현재:
```yaml
special_tokens:
  - '#Person1#'
  - '#Person2#'
  - '#Person3#'
  - '#PhoneNumber#'
  - '#Address#'
  - '#PassportNumber#'
```

추가:
```yaml
special_tokens:
  # ... 기존 토큰 ...
  - '#Person4#'
  - '#Person5#'
  - '#Person6#'
  - '#Person7#'
  - '#DateOfBirth#'
  - '#SSN#'
  - '#CardNumber#'
  - '#CarNumber#'
  - '#Email#'
```

---

### 4.2 단기 구현 (Medium Priority)

#### 1. 토큰 길이 필터링 모듈
**새 파일**: `src/data/token_filter.py`

```python
"""토큰 길이 기반 샘플 필터링 및 처리"""

def filter_long_sequences(
    dataset,
    tokenizer,
    max_length: int = 512,
    strategy: str = 'truncate'  # 'truncate', 'sliding_window', 'split'
):
    """
    긴 시퀀스 감지 및 처리

    Args:
        dataset: 데이터셋
        tokenizer: 토크나이저
        max_length: 최대 토큰 길이
        strategy: 처리 전략
            - truncate: 단순 절단
            - sliding_window: 윈도우 슬라이딩
            - split: 여러 샘플로 분할
    """
    pass
```

---

#### 2. 도메인 클러스터링 모듈
**새 파일**: `src/data/topic_classifier.py`

```python
"""주제 기반 도메인 분류"""

# JSH 구현의 12개 도메인 분류 체계
DOMAIN_KEYWORDS = {
    '음식/식사': ['음식', '주문', '식당', '레스토랑', ...],
    '쇼핑/구매': ['쇼핑', '구매', '가격', ...],
    # ... (총 12개 도메인)
}

def classify_topic_to_domain(topic: str) -> str:
    """주제를 도메인으로 분류"""
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in topic for kw in keywords):
            return domain
    return '기타'

def add_domain_column(df):
    """데이터프레임에 domain 컬럼 추가"""
    df['domain'] = df['topic'].apply(classify_topic_to_domain)
    return df
```

---

#### 3. 도메인별 증강 전략
**파일**: `src/augmentation/text_augmenter.py` (수정)

```python
def augment_by_domain_priority(
    dataset,
    target_ratio: float = 0.5,
    domain_priorities: dict = None
):
    """
    도메인 우선순위 기반 증강

    Args:
        dataset: 원본 데이터셋
        target_ratio: 전체 증강 비율
        domain_priorities: 도메인별 우선순위
            예: {'금융/은행': 'HIGH', '여가/취미': 'MEDIUM', ...}
    """
    # 도메인별 현재 샘플 수 계산
    domain_counts = dataset['domain'].value_counts()

    # 목표 샘플 수 계산 (전체의 50% 달성)
    total_samples = len(dataset)
    target_total = total_samples / (1 - target_ratio)

    # 도메인별 증강 수 계산
    augmentation_plan = {}
    for domain in domain_priorities:
        current_count = domain_counts.get(domain, 0)
        if domain_priorities[domain] == 'HIGH':
            # LOW → 전체의 일정 비율까지 증강
            target_count = target_total * 0.05  # 5%
            aug_count = max(0, target_count - current_count)
        elif domain_priorities[domain] == 'MEDIUM':
            # 균등 분포 목표
            target_count = target_total / len(DOMAIN_KEYWORDS)
            aug_count = max(0, target_count - current_count)
        else:
            aug_count = 0

        augmentation_plan[domain] = int(aug_count)

    # 도메인별 증강 실행
    # ...
```

---

#### 4. Stratified K-Fold
**새 파일**: `src/validation/stratified_kfold.py`

```python
"""도메인 비율 유지 K-Fold"""

from sklearn.model_selection import StratifiedKFold

def create_stratified_folds(dataset, n_splits=5, domain_column='domain'):
    """
    도메인 비율을 유지하는 K-Fold 분할

    Args:
        dataset: 데이터셋 (domain 컬럼 포함)
        n_splits: Fold 수
        domain_column: 도메인 컬럼명

    Returns:
        fold_indices: 각 Fold의 인덱스
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 도메인을 레이블로 사용
    labels = dataset[domain_column]

    fold_indices = []
    for train_idx, val_idx in skf.split(dataset, labels):
        fold_indices.append({
            'train': train_idx,
            'val': val_idx
        })

    return fold_indices
```

---

### 4.3 중장기 구현 (Low Priority)

#### 1. EDA 자동화 모듈
**새 파일**: `src/analysis/eda_module.py`

KBH 구현의 13가지 분석 항목을 자동화:
- 데이터 구조 확인
- 텍스트 길이 분석
- 주제 분포
- 화자 분석
- 발화 분석
- 토큰 길이 분석
- 특수 패턴 분석
- PII 마스킹 분석
- 문장 수 분석
- 어휘 다양성 분석
- N-gram 분석
- 성능 문제 샘플 분석
- Baseline 예측 비교

---

#### 2. 성능 문제 샘플 집중 학습
**새 파일**: `src/training/sample_weighting.py`

```python
"""샘플 가중치 계산"""

def calculate_sample_weights(dataset, tokenizer):
    """
    성능 문제 샘플에 높은 가중치 부여

    기준:
    - 긴 대화 (상위 10%)
    - 복잡한 대화 (화자 3명 이상)
    - 512 토큰 초과
    - 낮은 압축 비율 (하위 10%)
    """
    weights = []

    for sample in dataset:
        weight = 1.0

        # 토큰 길이 계산
        tokens = tokenizer.encode(sample['dialogue'])
        if len(tokens) > 512:
            weight *= 1.5

        # 화자 수
        num_speakers = count_speakers(sample['dialogue'])
        if num_speakers >= 3:
            weight *= 1.3

        # ... 기타 기준

        weights.append(weight)

    return weights
```

---

## 5. 우선순위 및 실행 계획

### Phase 1: 즉시 적용 (1-2일)
1. ✅ 생성 파라미터 조정 (configs/models/kobart.yaml)
2. ✅ Optuna 탐색 범위 확장
3. ✅ Special Tokens 확장

**예상 효과**: ROUGE +0.01~0.02

---

### Phase 2: 단기 구현 (1주)
1. 도메인 클러스터링 모듈 (`src/data/topic_classifier.py`)
2. 도메인별 증강 전략 (`src/augmentation/text_augmenter.py` 수정)
3. Stratified K-Fold (`src/validation/stratified_kfold.py`)
4. 토큰 길이 필터링 (`src/data/token_filter.py`)

**예상 효과**: ROUGE +0.02~0.04

---

### Phase 3: 중장기 구현 (2-3주)
1. EDA 자동화 모듈
2. 성능 문제 샘플 집중 학습
3. 도메인별 모델 Fine-tuning

**예상 효과**: ROUGE +0.03~0.05

---

## 6. 결론

다양한 구현 사례 분석을 통해 현재 모듈화 시스템의 강점과 개선 영역을 명확히 파악하였다. 특히:

- **KBH 구현**에서 체계적인 EDA 방법론
- **JSH 구현**에서 도메인 기반 증강 전략
- **KSM 구현**에서 최적화된 생성 파라미터

를 학습하였으며, 이를 단계적으로 적용하여 모델 성능을 향상시킬 수 있을 것으로 기대된다.

즉시 적용 가능한 개선 사항부터 순차적으로 실험하고, 각 단계에서 성능 변화를 면밀히 모니터링하며 진행하는 것을 권장한다.

---

## 참고 문서

- `notebooks/team/KBH/eda_notebook.ipynb`
- `notebooks/team/JSH/find_topic_categories.py`
- `notebooks/team/KSM/baseline.py`, `config.yaml`
- `configs/models/kobart.yaml`
- `scripts/train.py`
