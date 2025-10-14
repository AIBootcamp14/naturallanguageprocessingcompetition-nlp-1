# K-Fold 학습 모델 추론 가이드

> **작성일**: 2025-10-14
> **대상 실험**: 20251014_090813_kobart_balanced (K-Fold 5-Fold)
> **목적**: K-Fold로 학습된 5개 모델을 활용한 추론 방법 안내

---

## 📋 목차

1. [문제 상황](#1-문제-상황)
2. [현재 inference.py 제약사항](#2-현재-inferencepy-제약사항)
3. [추론 방법](#3-추론-방법)
4. [옵션별 상세 가이드](#4-옵션별-상세-가이드)
5. [FAQ](#5-faq)

---

## 1. 문제 상황

### 1.1 K-Fold 학습 완료 후 상태

K-Fold 학습을 완료하면 다음과 같은 구조로 모델이 저장됩니다:

```
experiments/20251014/20251014_090813_kobart_balanced/
├── fold_1/default/final_model/  ✅ 모델 저장됨
├── fold_2/default/final_model/  ✅ 모델 저장됨
├── fold_3/default/final_model/  ✅ 모델 저장됨
├── fold_4/default/final_model/  ✅ 모델 저장됨 (최고 성능)
├── fold_5/default/final_model/  ✅ 모델 저장됨
├── kfold_results.json          ✅ 학습 결과 요약
└── train.log                    ✅ 학습 로그

❌ predictions.csv (추론 결과 없음)
```

**현재 상태**:
- ✅ 학습 완료 (5개 Fold 모두)
- ✅ 검증 완료 (validation 데이터로 ROUGE 측정)
- ❌ 추론 미진행 (test 데이터 예측 없음)

### 1.2 발생 오류

다음과 같은 명령어 실행 시 오류 발생:

```bash
python scripts/inference.py \
  --mode ensemble \
  --model_paths \
    experiments/.../fold_1/default/final_model \
    experiments/.../fold_2/default/final_model \
  --ensemble_strategy weighted
```

**오류 메시지**:
```
inference.py: error: unrecognized arguments: --mode --model_paths
inference.py: error: argument --ensemble_strategy: invalid choice: 'weighted'
```

---

## 2. 현재 inference.py 제약사항

### 2.1 지원되는 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `--model` | 단수 (필수) | **단일 모델 경로만 지원** |
| `--test_data` | 문자열 | 테스트 데이터 경로 |
| `--output` | 문자열 | 출력 파일 경로 |
| `--batch_size` | 정수 | 배치 크기 (기본값: 32) |
| `--num_beams` | 정수 | Beam search 크기 (기본값: 4) |
| `--max_new_tokens` | 정수 | 최대 생성 토큰 수 |
| `--min_new_tokens` | 정수 | 최소 생성 토큰 수 |
| `--repetition_penalty` | 실수 | 반복 억제 강도 |
| `--no_repeat_ngram_size` | 정수 | N-gram 반복 방지 크기 |
| `--ensemble_strategy` | 선택 | `weighted_avg`, `quality_based`, `voting` |
| `--use_pretrained_correction` | 플래그 | HF 모델 보정 사용 |
| `--correction_models` | 리스트 | 보정 모델 경로들 |

### 2.2 지원되지 않는 파라미터

| 파라미터 | 상태 | 이유 |
|---------|------|------|
| ❌ `--mode` | 미지원 | inference.py는 항상 단일 모드 |
| ❌ `--model_paths` (복수) | 미지원 | 단일 `--model`만 지원 |
| ❌ `--weights` | 미지원 | K-Fold 앙상블 미구현 |
| ❌ `--ensemble_strategy weighted` | 오류 | `weighted_avg` 사용 필요 |

### 2.3 `--ensemble_strategy` 허용 값

- ✅ `weighted_avg`: 가중 평균 앙상블
- ✅ `quality_based`: 품질 기반 선택
- ✅ `voting`: 투표 방식
- ❌ `weighted`: **지원 안됨** (오타 주의)

---

## 3. 추론 방법

K-Fold 학습 후 추론을 위한 3가지 방법:

### 방법 비교표

| 방법 | 속도 | 성능 | 복잡도 | 추천도 |
|------|------|------|--------|--------|
| **옵션 1: 최고 성능 Fold만 사용** | ⚡⚡⚡ 빠름 | 🎯 높음 (1.2352) | ⭐ 간단 | 🌟🌟🌟 **강력 추천** |
| **옵션 2: 개별 추론 후 수동 앙상블** | ⚡ 느림 | 🎯🎯 매우 높음 | ⭐⭐⭐ 복잡 | 🌟🌟 추천 |
| **옵션 3: 평균 앙상블 (동일 가중치)** | ⚡⚡ 중간 | 🎯🎯 높음 | ⭐⭐ 보통 | 🌟 선택적 |

---

## 4. 옵션별 상세 가이드

### 옵션 1: 최고 성능 Fold만 사용 ⚡ (권장)

#### 4.1.1 개요

- **대상 모델**: Fold 4 (ROUGE-Sum: 1.2352, 전체 최고)
- **장점**: 가장 빠름, 단일 명령어로 완료, 우수한 성능
- **단점**: 앙상블 효과 없음 (하지만 성능 충분)

#### 4.1.2 명령어

```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_090813_kobart_balanced/fold_4/default/final_model \
  --test_data data/raw/test.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3
```

#### 4.1.3 파라미터 설명

| 파라미터 | 값 | 이유 |
|---------|-----|------|
| `--model` | fold_4/default/final_model | 최고 성능 모델 (ROUGE-Sum: 1.2352) |
| `--num_beams` | 5 | 학습 시 사용한 값과 동일 |
| `--max_new_tokens` | 100 | 학습 시 사용한 값과 동일 |
| `--min_new_tokens` | 30 | 학습 시 사용한 값과 동일 |
| `--repetition_penalty` | 1.5 | 학습 시 사용한 값과 동일 |
| `--no_repeat_ngram_size` | 3 | 학습 시 사용한 값과 동일 |
| `--use_pretrained_correction` | (플래그) | 학습 시 활성화했던 보정 기능 |
| `--correction_strategy` | quality_based | 학습 시 사용한 전략 |

#### 4.1.4 예상 출력

```
실험 폴더: experiments/20251014/20251014_XXXXXX_inference_kobart_bs32_beam5_maxnew100_minnew30_rep1.5_ngram3_hf/
└── submissions/
    └── 20251014_XXXXXX_inference_kobart_bs32_beam5_maxnew100_minnew30_rep1.5_ngram3_hf.csv

전역 제출 폴더: submissions/20251014/
└── 20251014_XXXXXX_inference_kobart_bs32_beam5_maxnew100_minnew30_rep1.5_ngram3_hf.csv
```

#### 4.1.5 실행 후 확인

```bash
# 1. 파일 생성 확인
ls -lh experiments/20251014/*/submissions/*.csv
ls -lh submissions/20251014/*.csv

# 2. 샘플 확인
head -5 submissions/20251014/20251014_*_inference_kobart*.csv

# 3. 행 개수 확인 (테스트 데이터와 동일해야 함)
wc -l submissions/20251014/20251014_*_inference_kobart*.csv
wc -l data/test.csv
```

---

### 옵션 2: 개별 추론 후 수동 앙상블 (최고 성능)

#### 4.2.1 개요

- **방법**: 5개 Fold를 각각 추론 → Python으로 앙상블
- **장점**: 앙상블 효과로 최고 성능, 가중치 커스터마이징 가능
- **단점**: 5번의 추론 필요 (시간 5배), 추가 코드 작성 필요

#### 4.2.2 Step 1: 각 Fold 개별 추론

**Fold 1 추론**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_090813_kobart_balanced/fold_1/default/final_model \
  --test_data data/raw/test.csv \
  --output experiments/20251014/20251014_090813_kobart_balanced/predictions_fold1.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3
```

**Fold 2 추론**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_090813_kobart_balanced/fold_2/default/final_model \
  --test_data data/raw/test.csv \
  --output experiments/20251014/20251014_090813_kobart_balanced/predictions_fold2.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3
```

**Fold 3 추론**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_090813_kobart_balanced/fold_3/default/final_model \
  --test_data data/raw/test.csv \
  --output experiments/20251014/20251014_090813_kobart_balanced/predictions_fold3.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3
```

**Fold 4 추론**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_090813_kobart_balanced/fold_4/default/final_model \
  --test_data data/raw/test.csv \
  --output experiments/20251014/20251014_090813_kobart_balanced/predictions_fold4.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3
```

**Fold 5 추론**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_090813_kobart_balanced/fold_5/default/final_model \
  --test_data data/raw/test.csv \
  --output experiments/20251014/20251014_090813_kobart_balanced/predictions_fold5.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3
```

#### 4.2.3 Step 2: 앙상블 스크립트 작성

**파일**: `scripts/ensemble_kfold.py`

```python
"""
K-Fold 추론 결과 앙상블 스크립트

사용법:
    python scripts/ensemble_kfold.py \
        --input_dir experiments/20251014/20251014_090813_kobart_balanced \
        --output submissions/20251014/ensemble_kobart_balanced.csv \
        --strategy weighted \
        --weights 0.19 0.18 0.20 0.23 0.20
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np


def weighted_ensemble(predictions_list, weights):
    """
    가중 평균 앙상블 (문자열 기반 투표)

    각 샘플에 대해:
    1. 각 Fold의 예측을 가중치만큼 투표
    2. 가장 많은 투표를 받은 예측 선택
    """
    ensemble_results = []

    for idx in range(len(predictions_list[0])):
        votes = []
        for fold_idx, (preds, weight) in enumerate(zip(predictions_list, weights)):
            # 가중치만큼 투표 (가중치를 정수로 변환하여 반복)
            vote_count = int(weight * 100)  # 0.23 -> 23표
            votes.extend([preds[idx]] * vote_count)

        # 최다 득표 예측 선택
        most_common = Counter(votes).most_common(1)[0][0]
        ensemble_results.append(most_common)

    return ensemble_results


def quality_based_ensemble(predictions_list, quality_scores):
    """
    품질 기반 앙상블

    각 샘플에 대해 가장 높은 품질 점수를 가진 Fold의 예측 선택
    """
    best_fold_idx = quality_scores.index(max(quality_scores))
    return predictions_list[best_fold_idx]


def voting_ensemble(predictions_list):
    """
    단순 투표 앙상블 (동일 가중치)

    가장 많이 등장하는 예측 선택
    """
    ensemble_results = []

    for idx in range(len(predictions_list[0])):
        votes = [preds[idx] for preds in predictions_list]
        most_common = Counter(votes).most_common(1)[0][0]
        ensemble_results.append(most_common)

    return ensemble_results


def main():
    parser = argparse.ArgumentParser(description="K-Fold 추론 결과 앙상블")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Fold별 추론 결과가 저장된 디렉토리"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="앙상블 결과 출력 경로"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="weighted",
        choices=["weighted", "voting", "quality_based"],
        help="앙상블 전략"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=5,
        default=[0.19, 0.18, 0.20, 0.23, 0.20],
        help="각 Fold의 가중치 (5개, 합=1.0)"
    )
    parser.add_argument(
        "--quality_scores",
        type=float,
        nargs=5,
        default=[1.2233, 1.2078, 1.2264, 1.2352, 1.2075],
        help="각 Fold의 ROUGE-Sum 점수 (quality_based 전략용)"
    )

    args = parser.parse_args()

    # 입력 디렉토리 확인
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")

    # Fold별 예측 파일 로드
    print("=" * 60)
    print("K-Fold 앙상블 시작")
    print("=" * 60)

    predictions_list = []
    fold_files = []

    for fold_idx in range(1, 6):
        fold_file = input_dir / f"predictions_fold{fold_idx}.csv"

        if not fold_file.exists():
            raise FileNotFoundError(f"Fold {fold_idx} 예측 파일이 없습니다: {fold_file}")

        df = pd.read_csv(fold_file)
        predictions_list.append(df['summary'].tolist())
        fold_files.append(fold_file)

        print(f"✅ Fold {fold_idx} 로드 완료: {len(df)} 샘플")

    # 가중치 정규화
    if args.strategy == "weighted":
        weights_sum = sum(args.weights)
        normalized_weights = [w / weights_sum for w in args.weights]
        print(f"\n가중치 (정규화): {normalized_weights}")

    # 앙상블 수행
    print(f"\n앙상블 전략: {args.strategy}")

    if args.strategy == "weighted":
        ensemble_summaries = weighted_ensemble(predictions_list, normalized_weights)
    elif args.strategy == "quality_based":
        print(f"품질 점수: {args.quality_scores}")
        ensemble_summaries = quality_based_ensemble(predictions_list, args.quality_scores)
    elif args.strategy == "voting":
        ensemble_summaries = voting_ensemble(predictions_list)

    # 결과 저장
    base_df = pd.read_csv(fold_files[0])
    result_df = base_df[['fname']].copy()
    result_df['summary'] = ensemble_summaries

    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(args.output, index=False, encoding='utf-8')

    print(f"\n✅ 앙상블 완료: {args.output}")
    print(f"   샘플 수: {len(result_df)}")

    # 샘플 출력
    print("\n샘플 결과 (처음 3개):")
    for idx, row in result_df.head(3).iterrows():
        print(f"  [{row['fname']}]: {row['summary'][:60]}...")

    print("\n" + "=" * 60)
    print("🎉 앙상블 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

#### 4.2.4 Step 3: 앙상블 실행

**가중 평균 앙상블** (Fold 4 가중치 증가):
```bash
python scripts/ensemble_kfold.py \
  --input_dir experiments/20251014/20251014_090813_kobart_balanced \
  --output submissions/20251014/ensemble_weighted_kobart_balanced.csv \
  --strategy weighted \
  --weights 0.19 0.18 0.20 0.23 0.20
```

**품질 기반 앙상블** (최고 품질 Fold 선택):
```bash
python scripts/ensemble_kfold.py \
  --input_dir experiments/20251014/20251014_090813_kobart_balanced \
  --output submissions/20251014/ensemble_quality_kobart_balanced.csv \
  --strategy quality_based \
  --quality_scores 1.2233 1.2078 1.2264 1.2352 1.2075
```

**투표 앙상블** (동일 가중치):
```bash
python scripts/ensemble_kfold.py \
  --input_dir experiments/20251014/20251014_090813_kobart_balanced \
  --output submissions/20251014/ensemble_voting_kobart_balanced.csv \
  --strategy voting
```

---

### 옵션 3: 평균 앙상블 (동일 가중치)

#### 4.3.1 개요

옵션 2와 동일하지만, 모든 Fold에 동일한 가중치(0.2) 부여

#### 4.3.2 명령어

```bash
# Step 1: 5번 개별 추론 (옵션 2와 동일)
# ... (생략)

# Step 2: 동일 가중치 앙상블
python scripts/ensemble_kfold.py \
  --input_dir experiments/20251014/20251014_090813_kobart_balanced \
  --output submissions/20251014/ensemble_equal_kobart_balanced.csv \
  --strategy weighted \
  --weights 0.20 0.20 0.20 0.20 0.20
```

---

## 5. FAQ

### Q1. K-Fold 학습 시 추론도 자동으로 되나요?

**A**: 아니요. K-Fold 학습(`train.py`)은 검증(validation) 데이터로 성능만 측정하고, 테스트 데이터 추론은 별도로 `inference.py`를 실행해야 합니다.

### Q2. 왜 `--mode ensemble`이 안되나요?

**A**: 현재 `inference.py`는 단일 모델 추론만 지원하도록 구현되어 있습니다. K-Fold 앙상블을 위해서는:
- 각 Fold를 개별 추론 후 수동 앙상블 (옵션 2)
- 또는 최고 성능 Fold만 사용 (옵션 1)

### Q3. `--ensemble_strategy weighted`를 쓰면 왜 오류가 나나요?

**A**: 올바른 값은 `weighted_avg`입니다. 허용되는 값:
- ✅ `weighted_avg`
- ✅ `quality_based`
- ✅ `voting`
- ❌ `weighted` (오타)

### Q4. 어떤 옵션이 가장 좋나요?

**A**: 상황에 따라 다릅니다:

| 우선순위 | 추천 옵션 |
|---------|----------|
| **속도 우선** | 옵션 1 (Fold 4만) ⚡ |
| **성능 우선** | 옵션 2 (가중 앙상블) 🎯 |
| **균형** | 옵션 1 (Fold 4도 충분히 좋음) ⚖️ |

대부분의 경우 **옵션 1 (Fold 4 단독)**으로 충분합니다.

### Q5. Fold 4가 왜 최고 성능인가요?

**A**: 실험 결과에 따르면:

| Fold | ROUGE-Sum | Best Epoch |
|------|-----------|------------|
| Fold 1 | 1.2233 | 12 |
| Fold 2 | 1.2078 | 4 |
| Fold 3 | 1.2264 | 11 |
| **Fold 4** | **1.2352** 🏆 | **10** |
| Fold 5 | 1.2075 | 12 |

Fold 4가 모든 ROUGE 지표에서 최고 성능을 보였습니다.

### Q6. 학습 시 사용한 파라미터를 추론에도 똑같이 써야 하나요?

**A**: 네, **생성 파라미터는 학습 시와 동일하게** 사용하는 것이 좋습니다:

```bash
# 학습 시 설정
--num_beams 5
--max_new_tokens 100
--min_new_tokens 30
--repetition_penalty 1.5
--no_repeat_ngram_size 3

# 추론 시에도 동일하게 사용
--num_beams 5
--max_new_tokens 100
--min_new_tokens 30
--repetition_penalty 1.5
--no_repeat_ngram_size 3
```

### Q7. 앙상블 스크립트는 어디에 저장하나요?

**A**: `scripts/ensemble_kfold.py` 파일을 생성하고, 위의 옵션 2 코드를 복사하세요.

### Q8. 추론 결과는 어디에 저장되나요?

**A**: 2곳에 저장됩니다:

1. **실험 폴더**: `experiments/날짜/실행폴더/submissions/파일명.csv`
2. **전역 제출 폴더**: `submissions/날짜/파일명.csv`

예시:
```
experiments/20251014/20251014_103045_inference_kobart/submissions/20251014_103045_inference_kobart.csv
submissions/20251014/20251014_103045_inference_kobart.csv
```

### Q9. 제출 파일 형식은 어떻게 확인하나요?

**A**:
```bash
# 1. 헤더 확인 (fname, summary 필요)
head -1 submissions/20251014/*.csv

# 2. 샘플 확인
head -5 submissions/20251014/*.csv

# 3. 행 개수 확인 (테스트 데이터와 동일해야 함)
wc -l submissions/20251014/*.csv
wc -l data/raw/test.csv

# 4. 결측치 확인
python -c "import pandas as pd; df = pd.read_csv('submissions/20251014/파일명.csv'); print(df.isnull().sum())"
```

### Q10. 추론 시간은 얼마나 걸리나요?

**A**: GPU와 테스트 데이터 크기에 따라 다릅니다:

| 모델 수 | 예상 시간 (RTX 3090 기준) |
|--------|--------------------------|
| 1개 Fold | 10~20분 |
| 5개 Fold (개별) | 50~100분 |

---

## 📚 참고 자료

### 관련 문서
- [실험 분석 보고서](experiments/20251014_090813_kobart_balanced_실험분석.md)
- [전략 2 문서](모듈화/04_02_*.md)

### 실험 파일
- **학습 로그**: `experiments/20251014/20251014_090813_kobart_balanced/train.log`
- **K-Fold 결과**: `experiments/20251014/20251014_090813_kobart_balanced/kfold_results.json`
- **Fold 모델들**: `experiments/20251014/20251014_090813_kobart_balanced/fold_*/default/final_model/`

---

## 🎯 빠른 시작 (Quick Start)

가장 간단하게 추론을 시작하려면:

```bash
# 최고 성능 Fold 4로 추론
python scripts/inference.py \
  --model experiments/20251014/20251014_090813_kobart_balanced/fold_4/default/final_model \
  --test_data data/raw/test.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3

# 결과 확인
ls -lh submissions/20251014/*.csv
head -5 submissions/20251014/*.csv
```

---

**문서 버전**: 1.0
**최종 수정일**: 2025-10-14
**작성자**: Claude Code
