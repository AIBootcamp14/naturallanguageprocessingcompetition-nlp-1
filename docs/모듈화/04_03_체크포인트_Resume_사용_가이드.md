# 체크포인트 Resume 사용 가이드

> **목적**: 학습/추론/검증 중단 시 체크포인트에서 이어서 실행하는 방법 안내
> **작성일**: 2025-10-14
> **버전**: 1.0

---

## 📋 목차

1. [체크포인트 시스템 개요](#1-체크포인트-시스템-개요)
2. [명령행 옵션 상세 설명](#2-명령행-옵션-상세-설명)
3. [단계별 Resume 명령어](#3-단계별-resume-명령어)
4. [실전 시나리오 예제](#4-실전-시나리오-예제)
5. [주의사항 및 FAQ](#5-주의사항-및-faq)

---

## 1. 체크포인트 시스템 개요

### 1.1 지원되는 체크포인트 유형

| 체크포인트 유형 | 저장 시점 | 저장 내용 | Resume 시 동작 |
|----------------|----------|----------|---------------|
| **Optuna 최적화** | Trial 완료마다 | Trial 결과, 최적 파라미터 | 완료된 Trial 건너뛰고 이어서 실행 |
| **K-Fold 학습** | Fold 완료마다 | Fold 모델, 평가 메트릭 | 완료된 Fold 건너뛰고 이어서 실행 |
| **데이터 증강** | 100개마다 | 증강된 데이터, 진행률 | 완료된 증강 데이터 로드 후 이어서 실행 |
| **HuggingFace 보정** | 배치마다 | 보정된 요약, 진행률 | 완료된 보정 데이터 로드 후 이어서 실행 |
| **Solar API 호출** | 배치마다 | API 응답, 통계 | 완료된 호출 데이터 로드 후 이어서 실행 |
| **검증** | 배치마다 | 예측 결과, 메트릭 | 완료된 평가 데이터 로드 후 이어서 실행 |

### 1.2 체크포인트 저장 위치

```
experiments/{날짜}/{시간}_{실험명}/
├── checkpoints/                              # 체크포인트 폴더
│   ├── optuna_kobart_ultimate_checkpoint.pkl # Optuna 체크포인트
│   ├── kfold_checkpoint.json                 # K-Fold 체크포인트
│   ├── augmentation_checkpoint.pkl           # 데이터 증강 체크포인트
│   ├── correction_checkpoint.pkl             # HF 보정 체크포인트
│   ├── solar_api_checkpoint.pkl              # Solar API 체크포인트
│   └── validation_checkpoint.pkl             # 검증 체크포인트
├── fold_0/                                   # Fold 모델들
├── fold_1/
└── train.log                                 # 학습 로그
```

---

## 2. 명령행 옵션 상세 설명

### 2.1 기본 Resume 옵션

#### `--resume`
체크포인트에서 이어서 실행

```bash
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --resume  # ✅ 체크포인트에서 이어서 실행
```

**동작 방식**:
- 출력 디렉토리에서 체크포인트 파일 자동 탐지
- 체크포인트가 있으면: 완료된 작업 건너뛰고 이어서 실행
- 체크포인트가 없으면: 처음부터 시작 (경고 없음)

#### `--resume_from`
특정 체크포인트 디렉토리에서 Resume

```bash
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --resume \
  --resume_from experiments/20251014/20251014_143000_kobart_ultimate/checkpoints
```

**동작 방식**:
- 지정된 체크포인트 디렉토리에서 Resume 시작
- `checkpoints` 폴더를 지정하면 자동으로 상위 폴더를 실험 디렉토리로 인식
- 실험 폴더 경로를 직접 지정해도 자동으로 `checkpoints/` 하위 폴더 탐색

**사용 시나리오**:
- 다른 실험의 체크포인트를 이어받을 때
- 명시적으로 체크포인트 경로를 지정하고 싶을 때

#### `--ignore_checkpoint`
체크포인트 무시하고 처음부터 시작

```bash
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --ignore_checkpoint  # ✅ 체크포인트 무시
```

**사용 시나리오**:
- 동일한 실험명으로 새로 시작하고 싶을 때
- 체크포인트가 손상되었을 때

---

## 3. 단계별 Resume 명령어

### 3.1 Optuna 최적화 Resume

#### 시나리오: Trial 11/20 완료 후 중단

**초기 실행 (중단됨)**:
```bash
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --optuna_timeout 10800 \
  --epochs 7 \
  --batch_size 16 \
  --gradient_accumulation_steps 10 \
  --learning_rate 9.14e-5 \
  --warmup_ratio 0.00136 \
  --weight_decay 0.0995 \
  --scheduler_type cosine \
  --max_grad_norm 1.0 \
  --label_smoothing 0.1 \
  --use_augmentation \
  --augmentation_ratio 0.5 \
  --augmentation_methods back_translation paraphrase \
  --k_folds 5 \
  --fold_seed 42 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --num_beams 4 \
  --repetition_penalty 1.5 \
  --length_penalty 0.938 \
  --no_repeat_ngram_size 3 \
  --use_solar_api \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --save_visualizations \
  --experiment_name kobart_ultimate \
  --seed 42

# Trial 11/20 완료 후 Ctrl+C 또는 오류 발생
```

**Resume 실행 (남은 Trial 9개만 실행)**:
```bash
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --optuna_timeout 10800 \
  --epochs 7 \
  --batch_size 16 \
  --gradient_accumulation_steps 10 \
  --learning_rate 9.14e-5 \
  --warmup_ratio 0.00136 \
  --weight_decay 0.0995 \
  --scheduler_type cosine \
  --max_grad_norm 1.0 \
  --label_smoothing 0.1 \
  --use_augmentation \
  --augmentation_ratio 0.5 \
  --augmentation_methods back_translation paraphrase \
  --k_folds 5 \
  --fold_seed 42 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --num_beams 4 \
  --repetition_penalty 1.5 \
  --length_penalty 0.938 \
  --no_repeat_ngram_size 3 \
  --use_solar_api \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --save_visualizations \
  --experiment_name kobart_ultimate \
  --seed 42 \
  --resume  # ✅ 추가: Resume 옵션
```

**예상 로그**:
```
🔄 체크포인트에서 Resume: 11/20 Trial 이미 완료
  - 현재 최적값: 0.4616
  - 마지막 저장: 2025-10-14T14:30:00
  - 남은 Trial: 9개

Trial 12/20:
  - learning_rate: 7.25e-5
  - epochs: 6
  ...
```

---

### 3.2 K-Fold 학습 Resume

#### 시나리오: Fold 2/5 완료 후 중단

**초기 실행 (중단됨)**:
```bash
python scripts/train.py \
  --mode kfold \
  --models kobart \
  --epochs 7 \
  --batch_size 16 \
  --gradient_accumulation_steps 10 \
  --learning_rate 9.14e-5 \
  --warmup_ratio 0.00136 \
  --weight_decay 0.0995 \
  --scheduler_type cosine \
  --max_grad_norm 1.0 \
  --label_smoothing 0.1 \
  --use_augmentation \
  --augmentation_ratio 0.5 \
  --augmentation_methods back_translation paraphrase \
  --k_folds 5 \
  --fold_seed 42 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --num_beams 4 \
  --repetition_penalty 1.5 \
  --length_penalty 0.938 \
  --no_repeat_ngram_size 3 \
  --use_solar_api \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --experiment_name kobart_balanced \
  --seed 42

# Fold 2/5 완료 후 GPU 오류 발생
```

**Resume 실행 (남은 Fold 3개만 실행)**:
```bash
python scripts/train.py \
  --mode kfold \
  --models kobart \
  --epochs 7 \
  --batch_size 16 \
  --gradient_accumulation_steps 10 \
  --learning_rate 9.14e-5 \
  --warmup_ratio 0.00136 \
  --weight_decay 0.0995 \
  --scheduler_type cosine \
  --max_grad_norm 1.0 \
  --label_smoothing 0.1 \
  --use_augmentation \
  --augmentation_ratio 0.5 \
  --augmentation_methods back_translation paraphrase \
  --k_folds 5 \
  --fold_seed 42 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --num_beams 4 \
  --repetition_penalty 1.5 \
  --length_penalty 0.938 \
  --no_repeat_ngram_size 3 \
  --use_solar_api \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --experiment_name kobart_balanced \
  --seed 42 \
  --resume  # ✅ 추가: Resume 옵션
```

**예상 로그**:
```
🔄 체크포인트에서 Resume: 2/5 Fold 이미 완료
  완료된 Fold: [0, 1]

⏭️  Fold 1/5 - 이미 완료됨 (건너뜀)
⏭️  Fold 2/5 - 이미 완료됨 (건너뜀)

========================================
📌 Fold 3/5 학습 시작
========================================
  학습: 2632개
  검증: 658개
...
```

---

### 3.3 데이터 증강 Resume

#### 시나리오: 증강 50% 진행 후 중단

**초기 실행 (중단됨)**:
```bash
python scripts/train.py \
  --mode single \
  --models kobart \
  --epochs 5 \
  --batch_size 16 \
  --gradient_accumulation_steps 10 \
  --learning_rate 9.14e-5 \
  --use_augmentation \
  --augmentation_ratio 0.5 \
  --augmentation_methods back_translation paraphrase \
  --experiment_name kobart_test \
  --seed 42

# 증강 진행 중 (1500/3000) Ctrl+C
```

**Resume 실행 (50%부터 이어서 실행)**:
```bash
python scripts/train.py \
  --mode single \
  --models kobart \
  --epochs 5 \
  --batch_size 16 \
  --gradient_accumulation_steps 10 \
  --learning_rate 9.14e-5 \
  --use_augmentation \
  --augmentation_ratio 0.5 \
  --augmentation_methods back_translation paraphrase \
  --experiment_name kobart_test \
  --seed 42 \
  --resume  # ✅ 추가: Resume 옵션
```

**예상 로그**:
```
✅ 증강 데이터 체크포인트 발견. 로드 중...

데이터 증강 시작
  - 원본 데이터: 3290개
  - 증강 방법: ['back_translation', 'paraphrase']
  - 방법당 샘플 수: 1
  - 목표 데이터 크기: 9870개

[로드된 데이터 사용]
데이터 증강 완료: 9870개
```

---

### 3.4 HuggingFace 보정 Resume

#### 시나리오: 보정 진행 중 중단

**추론 명령어 (중단됨)**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_143000_kobart_balanced/fold_0/final_model \
  --test_data data/raw/test.csv \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --num_beams 4 \
  --length_penalty 0.938 \
  --repetition_penalty 1.5 \
  --batch_size 16 \
  --output submissions/kobart_hf_corrected.csv

# 보정 진행 중 (500/1000) 중단
```

**Resume 실행**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_143000_kobart_balanced/fold_0/final_model \
  --test_data data/raw/test.csv \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --num_beams 4 \
  --length_penalty 0.938 \
  --repetition_penalty 1.5 \
  --batch_size 16 \
  --output submissions/kobart_hf_corrected.csv \
  --resume  # ✅ 추가: Resume 옵션
```

**예상 로그**:
```
🔄 HF 보정 체크포인트 발견. 로드 중...
  - 완료: 500/1000
  - 진행률: 50.0%

보정 이어서 진행 중...
[501/1000] 보정 중...
[600/1000] 보정 중...
...
```

---

### 3.5 Solar API Resume

#### 시나리오: Solar API 배치 호출 중 중단

**추론 명령어 (중단됨)**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_143000_kobart_balanced/fold_0/final_model \
  --test_data data/raw/test.csv \
  --use_solar_api \
  --solar_model solar-1-mini-chat \
  --batch_size 10 \
  --output submissions/kobart_solar.csv

# API 호출 중 (300/1000) 타임아웃 발생
```

**Resume 실행**:
```bash
python scripts/inference.py \
  --model experiments/20251014/20251014_143000_kobart_balanced/fold_0/final_model \
  --test_data data/raw/test.csv \
  --use_solar_api \
  --solar_model solar-1-mini-chat \
  --batch_size 10 \
  --output submissions/kobart_solar.csv \
  --resume  # ✅ 추가: Resume 옵션
```

**예상 로그**:
```
🔄 Solar API 체크포인트 발견. 로드 중...
  - 완료: 300/1000
  - 진행률: 30.0%
  - 총 토큰: 45,000
  - 평균 지연시간: 1.2초

API 호출 이어서 진행 중...
[301/1000] API 호출 중...
[310/1000] 배치 완료 (지연시간: 1.1초)
...
```

---

### 3.6 검증 Resume

#### 시나리오: 대규모 검증 세트 평가 중 중단

**검증 명령어 (중단됨)**:
```bash
python scripts/validate.py \
  --model experiments/20251014/20251014_143000_kobart_balanced/fold_0/final_model \
  --validation_data data/raw/dev.csv \
  --batch_size 16 \
  --output_dir experiments/20251014/20251014_143000_kobart_balanced/validation

# 평가 진행 중 (200/500) 중단
```

**Resume 실행**:
```bash
python scripts/validate.py \
  --model experiments/20251014/20251014_143000_kobart_balanced/fold_0/final_model \
  --validation_data data/raw/dev.csv \
  --batch_size 16 \
  --output_dir experiments/20251014/20251014_143000_kobart_balanced/validation \
  --resume  # ✅ 추가: Resume 옵션
```

**예상 로그**:
```
🔄 검증 체크포인트 발견. 로드 중...
  - 완료: 200/500
  - 진행률: 40.0%
  - 현재 ROUGE-L: 0.42

검증 이어서 진행 중...
[201/500] 평가 중...
[210/500] 배치 완료
...
```

---

## 4. 실전 시나리오 예제

### 4.1 시나리오 1: 긴급 중단 후 Resume

**상황**: Optuna 최적화 중 GPU 서버 재부팅 필요

```bash
# 1단계: 진행 중인 학습 확인
tail -f experiments/20251014/*/train.log
# 출력: Trial 8/20 완료

# 2단계: Ctrl+C로 중단

# 3단계: GPU 서버 재부팅

# 4단계: 동일한 명령어에 --resume 추가하여 재실행
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  ... (기존 옵션 동일) ... \
  --experiment_name kobart_ultimate \
  --seed 42 \
  --resume  # ✅ Resume 옵션 추가

# 결과: Trial 9부터 이어서 실행 (Trial 1-8은 건너뜀)
```

---

### 4.2 시나리오 2: 다른 실험의 체크포인트 이어받기

**상황**: 이전 실험의 Optuna 결과를 이어받아 추가 Trial 실행

```bash
# 1단계: 이전 실험 체크포인트 위치 확인
ls experiments/20251014/20251014_120000_kobart_ultimate/checkpoints/
# 출력: optuna_kobart_ultimate_checkpoint.pkl

# 2단계: --resume_from으로 명시적으로 지정
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 30 \  # ✅ 20 → 30으로 증가
  ... (기존 옵션 동일) ... \
  --experiment_name kobart_ultimate_extended \
  --seed 42 \
  --resume \
  --resume_from experiments/20251014/20251014_120000_kobart_ultimate/checkpoints

# 결과: 이전 20 trials + 추가 10 trials = 총 30 trials 실행
```

---

### 4.3 시나리오 3: 손상된 체크포인트 무시하고 재시작

**상황**: 체크포인트 파일이 손상되어 로드 실패

```bash
# 1단계: Resume 시도 (실패)
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  ... (기존 옵션) ... \
  --resume

# 출력: ❌ 체크포인트 로드 실패: 파일 손상

# 2단계: --ignore_checkpoint로 처음부터 재시작
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  ... (기존 옵션) ... \
  --ignore_checkpoint  # ✅ 체크포인트 무시

# 결과: 처음부터 새로 시작 (Trial 1부터)
```

---

### 4.4 시나리오 4: 병렬 실험 관리

**상황**: 여러 전략을 동시에 실험하면서 중단/재개

```bash
# 실험 A: 최고 성능 전략 (Optuna)
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --experiment_name kobart_ultimate_A \
  --resume &

# 실험 B: 균형 전략 (K-Fold 5)
python scripts/train.py \
  --mode kfold \
  --models kobart \
  --k_folds 5 \
  --experiment_name kobart_balanced_B \
  --resume &

# 실험 C: 빠른 전략 (K-Fold 3)
python scripts/train.py \
  --mode kfold \
  --models kobart \
  --k_folds 3 \
  --experiment_name kobart_fast_C \
  --resume &

# 모든 실험이 체크포인트에서 자동으로 이어서 실행됨
```

---

## 5. 주의사항 및 FAQ

### 5.1 주의사항

#### ⚠️ 하이퍼파라미터 변경 금지
```bash
# ❌ 잘못된 예: Resume 시 하이퍼파라미터 변경
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --epochs 10 \  # ❌ 원래는 7이었는데 변경
  --resume

# ✅ 올바른 예: 동일한 하이퍼파라미터 유지
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --epochs 7 \  # ✅ 원래 값 유지
  --resume
```

**이유**: 하이퍼파라미터가 변경되면 체크포인트가 무효화될 수 있음

#### ⚠️ 실험명 일치 필수
```bash
# ❌ 잘못된 예: 실험명 변경
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --experiment_name kobart_ultimate_v2 \  # ❌ 원래는 kobart_ultimate
  --resume

# ✅ 올바른 예: 동일한 실험명 유지
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --experiment_name kobart_ultimate \  # ✅ 원래 값 유지
  --resume
```

**이유**: 실험명이 변경되면 출력 디렉토리가 달라져 체크포인트를 찾을 수 없음

#### ⚠️ 데이터 변경 주의
Resume 시 원본 데이터가 변경되면 일관성이 깨질 수 있으므로 주의

---

### 5.2 FAQ

#### Q1: 체크포인트가 자동으로 삭제되나요?
**A**: 아니요, 자동으로 삭제되지 않습니다. 필요 시 수동으로 삭제하거나 `cleanup_old_checkpoints()` 메서드를 사용하세요.

```python
# Python 스크립트에서 정리
from src.checkpoints import OptunaCheckpointManager

checkpoint_mgr = OptunaCheckpointManager("experiments/.../checkpoints", "optuna_study")
checkpoint_mgr.cleanup_old_checkpoints(keep_last_n=3)  # 최근 3개만 유지
```

#### Q2: 체크포인트 파일이 너무 큽니다. 어떻게 하나요?
**A**: 압축 기능을 사용하세요.

```python
# 체크포인트 압축
checkpoint_mgr.compress_checkpoint(compression_level=6)  # 레벨 1-9

# 필요 시 압축 해제
checkpoint_mgr.decompress_checkpoint()
```

#### Q3: Resume 시 진행률을 확인하고 싶습니다
**A**: 체크포인트에서 진행률을 확인할 수 있습니다.

```python
# Python 스크립트에서
progress = checkpoint_mgr.get_progress()
print(f"완료: {progress['completed']}/{progress['total']}")
print(f"진행률: {progress['ratio']*100:.1f}%")

# 또는 진행률 표시줄
print(checkpoint_mgr.format_progress_bar(width=50))
# 출력: [=========================                         ] 50.0%
```

#### Q4: 체크포인트가 손상되었는지 확인하는 방법은?
**A**: 체크포인트 로드를 시도해보세요.

```python
checkpoint = checkpoint_mgr.load_checkpoint()
if checkpoint is None:
    print("❌ 체크포인트 손상 또는 없음")
else:
    print("✅ 체크포인트 정상")
    print(f"저장 시간: {checkpoint['timestamp']}")
```

#### Q5: 여러 단계가 동시에 Resume되나요?
**A**: 네, 각 단계별로 독립적으로 Resume됩니다.

```bash
# Optuna + K-Fold + 데이터 증강 모두 Resume
python scripts/train.py \
  --mode optuna \
  --models kobart \
  --optuna_trials 20 \
  --use_augmentation \
  --k_folds 5 \
  --resume

# 동작:
# 1. 데이터 증강 체크포인트 확인 → Resume
# 2. Optuna 체크포인트 확인 → Resume
# 3. K-Fold는 Optuna 완료 후 별도 실행
```

#### Q6: Resume 시 시드가 달라지면 어떻게 되나요?
**A**: 이미 완료된 작업은 건너뛰고, 새로운 작업만 새 시드로 실행됩니다.

```bash
# 원래 실행 (seed=42, Trial 10/20 완료)
python scripts/train.py --mode optuna --optuna_trials 20 --seed 42

# Resume (seed=100)
python scripts/train.py --mode optuna --optuna_trials 20 --seed 100 --resume

# 결과:
# - Trial 1-10: 그대로 유지 (seed=42로 실행된 결과)
# - Trial 11-20: seed=100으로 실행
```

#### Q7: 체크포인트 없이 Resume하면 어떻게 되나요?
**A**: 경고 없이 처음부터 시작합니다.

```bash
python scripts/train.py --mode optuna --optuna_trials 20 --resume
# 체크포인트가 없으면 자동으로 처음부터 시작
```

#### Q8: 특정 단계의 체크포인트만 삭제하고 싶습니다
**A**: 해당 체크포인트 파일만 수동으로 삭제하세요.

```bash
# Optuna 체크포인트만 삭제
rm experiments/20251014/*/checkpoints/optuna_*_checkpoint.pkl

# K-Fold 체크포인트만 삭제
rm experiments/20251014/*/checkpoints/kfold_checkpoint.json
```

---

## 6. 고급 기능

### 6.1 체크포인트 압축 사용

```bash
# 학습 완료 후 압축 (디스크 공간 절약)
python -c "
from src.checkpoints import OptunaCheckpointManager
mgr = OptunaCheckpointManager('experiments/20251014/.../checkpoints', 'optuna_study')
mgr.compress_checkpoint(compression_level=9)  # 최대 압축
print(f'압축 전: {mgr.get_checkpoint_size()//1024}KB')
"
```

### 6.2 오래된 체크포인트 자동 정리

```bash
# 최근 3개 체크포인트만 유지, 7일 이상된 파일 삭제
python -c "
from src.checkpoints import OptunaCheckpointManager
mgr = OptunaCheckpointManager('experiments/20251014/.../checkpoints', 'optuna_study')
mgr.cleanup_old_checkpoints(keep_last_n=3, max_age_days=7)
print('✅ 오래된 체크포인트 정리 완료')
"
```

### 6.3 진행률 시각화

```bash
# 진행률 표시줄 출력
python -c "
from src.checkpoints import KFoldCheckpointManager
mgr = KFoldCheckpointManager('experiments/20251014/.../checkpoints', n_folds=5)
progress = mgr.get_progress()
print(mgr.format_progress_bar(width=50))
print(f'완료된 Fold: {progress[\"completed_folds\"]}/{progress[\"total_folds\"]}')
"
```

---

## 7. 요약

### 7.1 핵심 포인트

1. ✅ **모든 단계에서 체크포인트 지원**: Optuna, K-Fold, 데이터 증강, HF 보정, Solar API, 검증
2. ✅ **간단한 Resume**: 기존 명령어에 `--resume` 추가만
3. ✅ **자동 진행률 관리**: 완료된 작업 자동 건너뛰기
4. ✅ **안전한 저장**: 원자적 저장으로 파일 손상 방지
5. ✅ **유연한 관리**: 압축, 정리, 진행률 확인 기능

### 7.2 빠른 참조

| 명령 | 옵션 | 설명 |
|------|------|------|
| Resume | `--resume` | 체크포인트에서 이어서 실행 (자동 탐지) |
| 특정 경로 Resume | `--resume_from {경로}` | 지정된 체크포인트 디렉토리에서 Resume |
| 처음부터 시작 | `--ignore_checkpoint` | 체크포인트 무시하고 새로 시작 |

---

**작성**: 2025-10-14
**버전**: 1.0
**관련 문서**:
- `docs/modify/04_체크포인트_중간저장_기능_추가.md`
- `docs/모듈화/04_02_KoBART_단일모델_최강_성능_전략.md`
