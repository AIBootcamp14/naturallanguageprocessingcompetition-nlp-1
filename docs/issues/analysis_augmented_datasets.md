# 증강 데이터셋 분석 및 활용 방안

> 작성일: 2025-10-14
> 목적: 증강 데이터셋 분석을 통한 성능 향상 전략 수립

---

## 1. 개요

`data/processed/` 폴더에 저장된 두 개의 증강 데이터셋을 분석하여 현재 모듈화 시스템에 통합 가능한 활용 방안을 도출한다.

---

## 2. 증강 데이터셋 분석

### 2.1 데이터셋 개요

#### 파일 1: `train_with_augmentation_20251011_112913.csv`
- **총 라인 수**: 257,953줄 (헤더 포함)
- **샘플 수**: 약 257,952개
- **컬럼 구조**:
  ```
  fname, dialogue, summary, topic, is_augmented, augmentation_success
  ```

#### 파일 2: `high_quality_augmented.csv`
- **총 라인 수**: 43,735줄 (헤더 포함)
- **샘플 수**: 약 43,734개
- **컬럼 구조**:
  ```
  fname, dialogue, summary
  ```

### 2.2 증강 전략 분석

#### 원본 데이터 구성
- Train: 12,457개
- Dev: 499개
- Test: 499개 (추론 대상)

#### 증강 비율 추정

**파일 1**: `train_with_augmentation_20251011_112913.csv`
- 원본 + 증강 = 257,952개
- 원본 12,457개 → 증강 약 245,495개
- **증강 비율**: 약 1,970% (원본 대비 약 20배)

**파일 2**: `high_quality_augmented.csv`
- 순수 증강 데이터 = 43,734개
- **증강 비율**: 약 351% (원본 대비 약 3.5배)

### 2.3 샘플 특징 분석

#### 원본 샘플 (is_augmented=False)
```csv
train_0,"#Person1#: 안녕하세요, Mr. Smith...",
"Mr. Smith는 Dr. Hawkins에게 건강검진을 받으러 와서...",
"건강검진",False,False
```

#### 증강 샘플 (`high_quality_augmented.csv`)
```csv
train_9687_aug,"#Person1#: 보상 요청을 받고 놀랐습니다...",
"#Person2#는 상품의 3분의 1이 기준에 미달하여..."
```

**특징**:
- `fname`에 `_aug` 접미사 추가
- 화자 패턴 및 대화 구조 유지
- 요약문 길이 및 스타일 일관성 유지
- Topic 정보는 없음 (파일 2)

---

## 3. 증강 품질 평가

### 3.1 긍정적 측면

#### 1. 대규모 증강
- **파일 1**: 약 20배 증강 (245k 샘플)
- **파일 2**: 약 3.5배 증강 (44k 샘플)
- 데이터 부족 문제 완화

#### 2. 구조 유지
- 화자 패턴 보존 (`#Person1#`, `#Person2#` 등)
- 대화 형식 일관성
- 요약 스타일 유지

#### 3. 도메인 다양성
- 샘플 예시에서 다양한 주제 확인
  - 보상 요청, 가계부 관리, 휴대폰 구매 등

### 3.2 우려 사항

#### 1. 과도한 증강 (파일 1)
- 20배 증강은 과적합 위험
- 원본 데이터의 특성이 희석될 가능성
- 증강 샘플의 품질 편차 가능성

#### 2. Topic 정보 부재 (파일 2)
- `high_quality_augmented.csv`에는 topic 컬럼 없음
- 도메인 기반 전략 적용 어려움
- Stratified Sampling 불가

#### 3. 증강 성공 여부 불명확
- `augmentation_success` 컬럼이 있지만 활용 방안 불명확
- 품질 필터링 기준 불명확

---

## 4. 활용 방안

### 4.1 단계별 증강 데이터 적용 전략

#### Strategy A: 보수적 접근 (권장)
**목표**: 품질 우선, 안정성 확보

1. **Phase 1**: 원본 데이터로 Baseline 구축
   - 현재 모듈화 시스템 그대로 사용
   - 성능 기준선 확립

2. **Phase 2**: 소규모 증강 적용 (50%)
   - 원본 12,457개 + 증강 6,228개 = 18,685개
   - `high_quality_augmented.csv`의 일부 사용
   - 성능 변화 모니터링

3. **Phase 3**: 중규모 증강 (100%)
   - 원본 12,457개 + 증강 12,457개 = 24,914개
   - `high_quality_augmented.csv` 전체 또는 일부
   - 과적합 징후 체크

4. **Phase 4**: 대규모 증강 (선택적)
   - 원본 12,457개 + 증강 50,000~100,000개
   - Phase 3에서 성능 향상 확인 시에만 진행
   - `train_with_augmentation_20251011_112913.csv` 일부 활용

---

#### Strategy B: 공격적 접근
**목표**: 최대 성능, 실험적

1. **Phase 1**: 전체 증강 데이터 적용
   - `train_with_augmentation_20251011_112913.csv` 전체 사용 (257k)
   - 에폭 수 감소 (20 → 5-10)
   - 과적합 방지를 위한 정규화 강화

2. **Phase 2**: 앙상블 및 Fine-tuning
   - 증강 데이터로 사전학습
   - 원본 데이터로 Fine-tuning
   - 2단계 학습 전략

---

### 4.2 구체적인 적용 방법

#### 방법 1: Config 기반 데이터셋 선택

**새 파일**: `configs/data/augmented_train.yaml`

```yaml
# ==================== 증강 데이터셋 설정 ==================== #

data:
  # 데이터 소스
  sources:
    - path: "data/raw/train.csv"
      type: "original"
      weight: 1.0
    - path: "data/processed/high_quality_augmented.csv"
      type: "augmented"
      weight: 0.5  # 증강 데이터 가중치 (50%)
      limit: 6228  # 최대 샘플 수 제한

  # 증강 설정
  augmentation:
    enabled: true
    strategy: "mixed"  # 원본 + 증강 혼합
    ratio: 0.5  # 증강 비율 (0.0~1.0)
    quality_filter: true  # 품질 필터링 활성화

  # 품질 필터링 기준 (선택적)
  quality_criteria:
    min_dialogue_length: 50  # 최소 대화 길이
    max_dialogue_length: 2000  # 최대 대화 길이
    min_summary_length: 10  # 최소 요약 길이
    max_summary_length: 200  # 최대 요약 길이
    min_compression_ratio: 2.0  # 최소 압축 비율
    max_compression_ratio: 15.0  # 최대 압축 비율
```

---

#### 방법 2: 데이터 로더 수정

**파일**: `src/data/dataset.py` (수정)

```python
"""증강 데이터 통합 로더"""

import pandas as pd
from pathlib import Path

def load_mixed_dataset(
    original_path: str,
    augmented_path: str = None,
    augmentation_ratio: float = 0.5,
    quality_filter: bool = True,
    seed: int = 42
):
    """
    원본 + 증강 데이터 혼합 로드

    Args:
        original_path: 원본 데이터 경로
        augmented_path: 증강 데이터 경로 (None이면 증강 없음)
        augmentation_ratio: 증강 비율 (0.0~1.0)
        quality_filter: 품질 필터링 여부
        seed: 랜덤 시드

    Returns:
        mixed_dataset: 혼합 데이터셋
    """
    # 1. 원본 데이터 로드
    df_original = pd.read_csv(original_path)
    print(f"원본 데이터: {len(df_original)}개")

    # 2. 증강 데이터 없으면 원본만 반환
    if augmented_path is None or augmentation_ratio == 0.0:
        return df_original

    # 3. 증강 데이터 로드
    df_augmented = pd.read_csv(augmented_path)
    print(f"증강 데이터 (전체): {len(df_augmented)}개")

    # 4. 품질 필터링 (선택적)
    if quality_filter:
        df_augmented = apply_quality_filter(df_augmented)
        print(f"증강 데이터 (필터링 후): {len(df_augmented)}개")

    # 5. 증강 비율에 따라 샘플링
    num_augmented = int(len(df_original) * augmentation_ratio)
    df_augmented_sampled = df_augmented.sample(
        n=min(num_augmented, len(df_augmented)),
        random_state=seed
    )
    print(f"증강 데이터 (샘플링): {len(df_augmented_sampled)}개")

    # 6. 원본 + 증강 결합
    df_mixed = pd.concat([df_original, df_augmented_sampled], ignore_index=True)
    df_mixed = df_mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # 셔플

    print(f"최종 데이터: {len(df_mixed)}개 (원본 {len(df_original)} + 증강 {len(df_augmented_sampled)})")

    return df_mixed


def apply_quality_filter(df, criteria=None):
    """
    증강 데이터 품질 필터링

    Args:
        df: 증강 데이터프레임
        criteria: 필터링 기준 (dict)

    Returns:
        filtered_df: 필터링된 데이터프레임
    """
    if criteria is None:
        criteria = {
            'min_dialogue_length': 50,
            'max_dialogue_length': 2000,
            'min_summary_length': 10,
            'max_summary_length': 200,
            'min_compression_ratio': 2.0,
            'max_compression_ratio': 15.0
        }

    df_filtered = df.copy()

    # 텍스트 길이 계산
    df_filtered['dialogue_len'] = df_filtered['dialogue'].str.len()
    df_filtered['summary_len'] = df_filtered['summary'].str.len()
    df_filtered['compression_ratio'] = df_filtered['dialogue_len'] / df_filtered['summary_len']

    # 필터링
    mask = (
        (df_filtered['dialogue_len'] >= criteria['min_dialogue_length']) &
        (df_filtered['dialogue_len'] <= criteria['max_dialogue_length']) &
        (df_filtered['summary_len'] >= criteria['min_summary_length']) &
        (df_filtered['summary_len'] <= criteria['max_summary_length']) &
        (df_filtered['compression_ratio'] >= criteria['min_compression_ratio']) &
        (df_filtered['compression_ratio'] <= criteria['max_compression_ratio'])
    )

    df_filtered = df_filtered[mask].drop(
        columns=['dialogue_len', 'summary_len', 'compression_ratio']
    )

    return df_filtered
```

---

#### 방법 3: 명령행 옵션 추가

**파일**: `scripts/train.py` (수정)

```python
# ==================== 데이터 경로 ====================
parser.add_argument(
    '--train_data',
    type=str,
    default='data/raw/train.csv',
    help='학습 데이터 경로'
)

# ✨ 새로 추가
parser.add_argument(
    '--augmented_data',
    type=str,
    default=None,
    help='증강 데이터 경로 (None: 증강 없음)'
)

parser.add_argument(
    '--augmentation_ratio',
    type=float,
    default=0.5,
    help='증강 데이터 비율 (0.0~5.0, 기본값: 0.5 = 50%)'
)

parser.add_argument(
    '--quality_filter_augmented',
    action='store_true',
    help='증강 데이터 품질 필터링 활성화'
)
```

**사용 예시**:
```bash
# 증강 없음 (기본)
python scripts/train.py --mode single --models kobart

# 50% 증강 적용
python scripts/train.py --mode single --models kobart \
  --augmented_data data/processed/high_quality_augmented.csv \
  --augmentation_ratio 0.5

# 100% 증강 + 품질 필터링
python scripts/train.py --mode single --models kobart \
  --augmented_data data/processed/high_quality_augmented.csv \
  --augmentation_ratio 1.0 \
  --quality_filter_augmented

# 대규모 증강 (200%)
python scripts/train.py --mode single --models kobart \
  --augmented_data data/processed/train_with_augmentation_20251011_112913.csv \
  --augmentation_ratio 2.0 \
  --quality_filter_augmented
```

---

### 4.3 Topic 정보 보완 전략

`high_quality_augmented.csv`에는 topic 정보가 없으므로 자동 분류 필요.

**새 스크립트**: `scripts/add_topics_to_augmented.py`

```python
"""증강 데이터에 Topic 정보 자동 추가"""

import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.topic_classifier import classify_topic_from_dialogue

def add_topics_to_augmented_data(
    input_path: str,
    output_path: str = None
):
    """
    증강 데이터에 자동으로 Topic 컬럼 추가

    Args:
        input_path: 입력 CSV 경로
        output_path: 출력 CSV 경로 (None이면 원본 덮어쓰기)
    """
    # 데이터 로드
    df = pd.read_csv(input_path)
    print(f"증강 데이터 로드: {len(df)}개")

    # Topic 자동 분류
    print("Topic 자동 분류 중...")
    df['topic'] = df['dialogue'].apply(classify_topic_from_dialogue)

    # 도메인 분류
    from src.data.topic_classifier import classify_topic_to_domain
    df['domain'] = df['topic'].apply(classify_topic_to_domain)

    # 저장
    if output_path is None:
        output_path = input_path.replace('.csv', '_with_topics.csv')

    df.to_csv(output_path, index=False)
    print(f"✅ 저장 완료: {output_path}")

    # 통계 출력
    print("\nTopic 분포:")
    print(df['topic'].value_counts().head(20))

    print("\nDomain 분포:")
    print(df['domain'].value_counts())


if __name__ == "__main__":
    # 실행 예시
    add_topics_to_augmented_data(
        input_path="data/processed/high_quality_augmented.csv",
        output_path="data/processed/high_quality_augmented_with_topics.csv"
    )
```

---

## 5. 실험 계획

### 5.1 Baseline vs Augmentation 비교

| 실험 ID | 데이터 구성 | 샘플 수 | 예상 효과 |
|---------|------------|---------|-----------|
| **EXP-A0** | 원본만 | 12,457 | Baseline |
| **EXP-A1** | 원본 + 25% 증강 | 15,571 | +0.01~0.02 ROUGE |
| **EXP-A2** | 원본 + 50% 증강 | 18,685 | +0.02~0.03 ROUGE |
| **EXP-A3** | 원본 + 100% 증강 | 24,914 | +0.03~0.05 ROUGE |
| **EXP-A4** | 원본 + 200% 증강 | 37,371 | +0.04~0.06 ROUGE (과적합 위험) |

---

### 5.2 품질 필터링 효과

| 실험 ID | 필터링 | 샘플 수 | 예상 효과 |
|---------|--------|---------|-----------|
| **EXP-B1** | 필터링 없음 | 24,914 | 품질 편차 큼 |
| **EXP-B2** | 기본 필터링 | ~22,000 | 품질 향상 |
| **EXP-B3** | 엄격 필터링 | ~18,000 | 고품질 유지 |

**필터링 기준**:
- 기본: 대화 50-2000자, 요약 10-200자, 압축 비율 2-15
- 엄격: 대화 100-1500자, 요약 20-150자, 압축 비율 3-10

---

### 5.3 2단계 학습 전략

#### 전략: Pre-training + Fine-tuning

**Stage 1: Pre-training (증강 데이터)**
```bash
python scripts/train.py \
  --mode single \
  --models kobart \
  --augmented_data data/processed/train_with_augmentation_20251011_112913.csv \
  --augmentation_ratio 5.0 \
  --quality_filter_augmented \
  --epochs 5 \
  --output_dir experiments/pretrain_augmented
```

**Stage 2: Fine-tuning (원본 데이터)**
```bash
python scripts/train.py \
  --mode single \
  --models kobart \
  --train_data data/raw/train.csv \
  --epochs 10 \
  --resume_from experiments/pretrain_augmented/checkpoint-best \
  --output_dir experiments/finetune_original
```

**예상 효과**:
- Pre-training: 다양한 패턴 학습
- Fine-tuning: 원본 도메인 적응
- **예상 향상**: +0.05~0.08 ROUGE

---

## 6. 리스크 및 대응 방안

### 6.1 과적합 위험

**증상**:
- Train 성능 ↑, Dev 성능 ↓
- 증강 데이터에 과도하게 최적화

**대응**:
1. Early Stopping 강화
   - `early_stopping_patience: 2` (기존 3에서 감소)
2. 정규화 강화
   - `weight_decay: 0.02` (기존 0.01에서 증가)
3. Dropout 증가
   - `dropout: 0.2` (기존 0.1에서 증가)
4. 증강 비율 감소
   - 100% → 50%로 조정

---

### 6.2 품질 편차

**증상**:
- 증강 샘플 일부가 저품질
- 학습 불안정

**대응**:
1. 품질 필터링 적용
2. Confidence 기반 샘플 가중치
3. 증강 데이터 검증 루프 추가

---

### 6.3 Topic 불일치

**증상**:
- 증강 데이터에 topic 정보 없음
- Stratified Sampling 불가

**대응**:
1. Topic 자동 분류 스크립트 실행
2. 도메인 분류 추가
3. 검증 후 사용

---

## 7. 실행 가이드

### 7.1 즉시 실행 가능한 실험

#### 실험 1: 50% 증강 (권장 시작점)

```bash
# 1. Topic 정보 추가 (최초 1회)
python scripts/add_topics_to_augmented.py

# 2. 학습 실행
python scripts/train.py \
  --mode single \
  --models kobart \
  --augmented_data data/processed/high_quality_augmented_with_topics.csv \
  --augmentation_ratio 0.5 \
  --quality_filter_augmented \
  --epochs 20 \
  --experiment_name kobart_aug50

# 3. Baseline과 비교
python scripts/evaluate.py \
  --prediction experiments/baseline/predictions.csv \
  --prediction_aug experiments/kobart_aug50/predictions.csv
```

---

#### 실험 2: 100% 증강

```bash
python scripts/train.py \
  --mode single \
  --models kobart \
  --augmented_data data/processed/high_quality_augmented_with_topics.csv \
  --augmentation_ratio 1.0 \
  --quality_filter_augmented \
  --epochs 20 \
  --experiment_name kobart_aug100
```

---

#### 실험 3: 2단계 학습

```bash
# Stage 1: Pre-training
python scripts/train.py \
  --mode single \
  --models kobart \
  --augmented_data data/processed/train_with_augmentation_20251011_112913.csv \
  --augmentation_ratio 5.0 \
  --quality_filter_augmented \
  --epochs 5 \
  --experiment_name kobart_pretrain

# Stage 2: Fine-tuning
python scripts/train.py \
  --mode single \
  --models kobart \
  --train_data data/raw/train.csv \
  --epochs 10 \
  --resume_from experiments/<date>/kobart_pretrain/checkpoint-best \
  --experiment_name kobart_finetune
```

---

### 7.2 모니터링 체크리스트

실험 진행 시 다음 지표를 모니터링:

- [ ] Train Loss 감소 추세
- [ ] Dev ROUGE 점수 (rouge-1, rouge-2, rouge-l)
- [ ] Dev Loss (과적합 여부)
- [ ] Epoch별 성능 변화
- [ ] 생성 예시 품질 (정성 평가)

**Early Warning Signs**:
- Train Loss ↓, Dev Loss ↑ → **과적합**
- Train Loss ↓, Dev ROUGE ↓ → **증강 데이터 품질 문제**
- Train Loss 정체 → **학습률 조정 필요**

---

## 8. 결론 및 권장 사항

### 8.1 핵심 결론

1. **고품질 증강 데이터 확보**
   - `high_quality_augmented.csv` (44k 샘플)
   - `train_with_augmentation_20251011_112913.csv` (258k 샘플)

2. **단계적 적용 권장**
   - 50% → 100% → 200% 순차 실험
   - 각 단계에서 성능 검증

3. **품질 관리 필수**
   - 필터링 적용
   - Topic 정보 보완
   - 샘플 가중치 조정

---

### 8.2 우선순위 권장 사항

#### High Priority (즉시 실행)
1. ✅ `add_topics_to_augmented.py` 스크립트 작성 및 실행
2. ✅ `load_mixed_dataset()` 함수 구현
3. ✅ 50% 증강 실험 (EXP-A2)

#### Medium Priority (1주 내)
1. 품질 필터링 기능 구현
2. 100% 증강 실험 (EXP-A3)
3. 2단계 학습 전략 실험

#### Low Priority (2주 이상)
1. 대규모 증강 실험 (200%+)
2. 도메인별 증강 전략
3. 증강 품질 자동 평가 시스템

---

### 8.3 예상 성능 향상

보수적 추정:
- 50% 증강: **+0.02~0.03 ROUGE**
- 100% 증강: **+0.03~0.05 ROUGE**
- 2단계 학습: **+0.05~0.08 ROUGE**

**총 예상 향상**: **+0.03~0.08 ROUGE**

---

## 참고 자료

- `data/processed/train_with_augmentation_20251011_112913.csv`
- `data/processed/high_quality_augmented.csv`
- `src/data/dataset.py`
- `scripts/train.py`
- `docs/issues/analysis_reference_implementations.md` (참조 구현 분석)
