# Optuna 하이퍼파라미터 최적화 시스템 상세 가이드

## 📋 목차
1. [개요](#개요)
2. [OptunaOptimizer 클래스](#optunaoptimizer-클래스)
3. [탐색 공간 정의](#탐색-공간-정의)
4. [최적화 전략](#최적화-전략)
5. [사용 방법](#사용-방법)
6. [실행 명령어](#실행-명령어)

---

## 📝 개요

### 목적
- Bayesian Optimization을 통한 하이퍼파라미터 자동 최적화
- NLP 특화 탐색 공간 정의 (LoRA, Generation 파라미터 등)
- 조기 종료를 통한 효율적 탐색
- ROUGE 점수 기반 최적화

### 핵심 기능
- ✅ TPE (Tree-structured Parzen Estimator) Sampler
- ✅ Median Pruner를 통한 조기 종료
- ✅ 15개 하이퍼파라미터 동시 탐색
- ✅ 최적 파라미터 자동 저장
- ✅ 시각화 지원 (Plotly)

---

## 🔧 OptunaOptimizer 클래스

### 파일 위치
```
src/optimization/optuna_optimizer.py
```

### 클래스 구조

```python
class OptunaOptimizer:
    def __init__(config, train_dataset, val_dataset, n_trials, ...)
    def create_search_space(trial)
    def objective(trial)
    def optimize()
    def get_best_params()
    def get_best_value()
    def save_results(output_path)
    def plot_optimization_history(output_path)
```

### 초기화

```python
from src.optimization import OptunaOptimizer
from src.data import load_and_preprocess_data

# 데이터 로드
train_df, val_df = load_and_preprocess_data(train_path, split_ratio=0.9)

# Config 로드
from src.config import ConfigLoader
config_loader = ConfigLoader()
config = config_loader.load("baseline_kobart")

# 데이터셋 생성
from src.data import DialogueSummarizationDataset
train_dataset = DialogueSummarizationDataset(
    train_df['dialogue'].tolist(),
    train_df['summary'].tolist(),
    tokenizer,
    config
)

val_dataset = DialogueSummarizationDataset(
    val_df['dialogue'].tolist(),
    val_df['summary'].tolist(),
    tokenizer,
    config
)

# Optimizer 초기화
optimizer = OptunaOptimizer(
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_trials=50,                    # 50회 시도
    timeout=None,                   # 무제한
    study_name="kobart_optuna",     # Study 이름
    direction="maximize"            # ROUGE 최대화
)
```

---

## 🔍 탐색 공간 정의

### 1. LoRA 파라미터

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| lora_r | [8, 16, 32, 64] | LoRA rank |
| lora_alpha | [16, 32, 64, 128] | LoRA scaling factor |
| lora_dropout | 0.0 ~ 0.2 | LoRA dropout rate |

**코드:**
```python
params['lora_r'] = trial.suggest_categorical('lora_r', [8, 16, 32, 64])
params['lora_alpha'] = trial.suggest_categorical('lora_alpha', [16, 32, 64, 128])
params['lora_dropout'] = trial.suggest_float('lora_dropout', 0.0, 0.2)
```

---

### 2. 학습 파라미터

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| learning_rate | 1e-6 ~ 1e-4 (log scale) | 학습률 |
| batch_size | [8, 16, 32, 64] | 배치 크기 |
| num_epochs | 3 ~ 10 | 에포크 수 |
| warmup_ratio | 0.0 ~ 0.2 | Warmup 비율 |
| weight_decay | 0.0 ~ 0.1 | Weight decay |

**코드:**
```python
params['learning_rate'] = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
params['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
params['num_epochs'] = trial.suggest_int('num_epochs', 3, 10)
params['warmup_ratio'] = trial.suggest_float('warmup_ratio', 0.0, 0.2)
params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
```

---

### 3. Scheduler

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| scheduler_type | [linear, cosine, cosine_with_restarts, polynomial] | Scheduler 종류 |

**코드:**
```python
params['scheduler_type'] = trial.suggest_categorical(
    'scheduler_type',
    ['linear', 'cosine', 'cosine_with_restarts', 'polynomial']
)
```

---

### 4. Generation 파라미터

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| temperature | 0.1 ~ 1.0 | 생성 온도 |
| top_p | 0.5 ~ 1.0 | Nucleus sampling |
| num_beams | [2, 4, 6, 8] | Beam search 빔 개수 |
| length_penalty | 0.5 ~ 2.0 | 길이 패널티 |

**코드:**
```python
params['temperature'] = trial.suggest_float('temperature', 0.1, 1.0)
params['top_p'] = trial.suggest_float('top_p', 0.5, 1.0)
params['num_beams'] = trial.suggest_categorical('num_beams', [2, 4, 6, 8])
params['length_penalty'] = trial.suggest_float('length_penalty', 0.5, 2.0)
```

---

### 5. Dropout 파라미터

| 파라미터 | 탐색 범위 | 설명 |
|---------|----------|------|
| hidden_dropout | 0.0 ~ 0.3 | Hidden layer dropout |
| attention_dropout | 0.0 ~ 0.3 | Attention dropout |

**코드:**
```python
if config.model.get('hidden_dropout_prob') is not None:
    params['hidden_dropout'] = trial.suggest_float('hidden_dropout', 0.0, 0.3)
    params['attention_dropout'] = trial.suggest_float('attention_dropout', 0.0, 0.3)
```

---

## ⚡ 최적화 전략

### 1. Bayesian Optimization (TPE)

**특징:**
- Tree-structured Parzen Estimator
- 이전 trial 결과를 활용하여 다음 탐색 위치 결정
- Random search보다 효율적

**설정:**
```python
from optuna.samplers import TPESampler

sampler = TPESampler(seed=42)
```

---

### 2. Median Pruner (조기 종료)

**특징:**
- 중간 결과가 median보다 낮으면 trial 종료
- 리소스 절약 (불필요한 trial 조기 중단)

**설정:**
```python
from optuna.pruners import MedianPruner

pruner = MedianPruner(
    n_startup_trials=5,   # 처음 5개는 pruning 안함
    n_warmup_steps=3,     # 3 에포크 후부터 체크
    interval_steps=1      # 매 에포크마다 체크
)
```

**동작 방식:**
```
Trial 0: [에포크1: 0.30] [에포크2: 0.32] [에포크3: 0.35] → 계속
Trial 1: [에포크1: 0.28] [에포크2: 0.29] [에포크3: 0.30] → 계속
Trial 2: [에포크1: 0.25] [에포크2: 0.26] [에포크3: 0.27] → Pruned! (median=0.32보다 낮음)
```

---

### 3. 목적 함수 (Objective Function)

**목표:** ROUGE-L F1 점수 최대화

**흐름:**
1. Trial에서 하이퍼파라미터 샘플링
2. Config 업데이트
3. 모델 로드 및 학습
4. 검증 데이터 평가
5. ROUGE-L F1 반환

**코드:**
```python
def objective(self, trial: optuna.Trial) -> float:
    # 1. 하이퍼파라미터 샘플링
    params = self.create_search_space(trial)

    # 2. Config 업데이트
    config.training.learning_rate = params['learning_rate']
    config.training.batch_size = params['batch_size']
    # ... 기타 파라미터 업데이트

    # 3. 모델 학습
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load()

    trainer = ModelTrainer(...)
    trainer.train()

    # 4. 평가
    metrics = trainer.evaluate()
    rouge_l_f1 = metrics['rouge_l_f1']

    # 5. Pruning 체크
    trial.report(rouge_l_f1, step=config.training.num_epochs)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return rouge_l_f1
```

---

## 💻 사용 방법

### 1. 기본 최적화

```python
from src.optimization import OptunaOptimizer
from src.config import ConfigLoader
from src.data import load_and_preprocess_data, DialogueSummarizationDataset

# Config 로드
config_loader = ConfigLoader()
config = config_loader.load("baseline_kobart")

# 데이터 로드
train_df, val_df = load_and_preprocess_data("data/raw/train.csv", split_ratio=0.9)

# 토크나이저 로드
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.name)

# 데이터셋 생성
train_dataset = DialogueSummarizationDataset(
    train_df['dialogue'].tolist(),
    train_df['summary'].tolist(),
    tokenizer,
    config
)

val_dataset = DialogueSummarizationDataset(
    val_df['dialogue'].tolist(),
    val_df['summary'].tolist(),
    tokenizer,
    config
)

# Optimizer 초기화
optimizer = OptunaOptimizer(
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_trials=50
)

# 최적화 실행
study = optimizer.optimize()

# 최적 파라미터 확인
best_params = optimizer.get_best_params()
best_value = optimizer.get_best_value()

print(f"최적 ROUGE-L F1: {best_value:.4f}")
print(f"최적 파라미터: {best_params}")
```

---

### 2. 결과 저장

```python
# 결과 저장
optimizer.save_results("outputs/optuna_results")

# 저장되는 파일:
# - outputs/optuna_results/best_params.json
# - outputs/optuna_results/all_trials.csv
# - outputs/optuna_results/study_stats.json
```

**best_params.json 예시:**
```json
{
  "best_params": {
    "learning_rate": 3.5e-05,
    "batch_size": 32,
    "num_epochs": 5,
    "lora_r": 16,
    "lora_alpha": 32,
    "temperature": 0.8,
    "num_beams": 6
  },
  "best_value": 0.4521,
  "n_trials": 50
}
```

**study_stats.json 예시:**
```json
{
  "study_name": "kobart_optuna",
  "n_trials": 50,
  "n_completed": 42,
  "n_pruned": 6,
  "n_failed": 2,
  "best_value": 0.4521,
  "best_trial_number": 37
}
```

---

### 3. 시각화

```python
# 시각화 생성 (requires plotly)
optimizer.plot_optimization_history("outputs/optuna_plots")

# 생성되는 파일:
# - optimization_history.html (최적화 히스토리)
# - param_importances.html (파라미터 중요도)
# - parallel_coordinate.html (병렬 좌표 플롯)
```

---

### 4. 최적 파라미터로 재학습

```python
# 최적 파라미터 로드
import json
with open("outputs/optuna_results/best_params.json", 'r') as f:
    data = json.load(f)
    best_params = data['best_params']

# Config 업데이트
config.training.learning_rate = best_params['learning_rate']
config.training.batch_size = best_params['batch_size']
config.training.num_epochs = best_params['num_epochs']
config.generation.temperature = best_params['temperature']
config.generation.num_beams = best_params['num_beams']

if 'lora_r' in best_params:
    config.lora.r = best_params['lora_r']
    config.lora.alpha = best_params['lora_alpha']

# 최종 학습
from src.training import ModelTrainer
trainer = ModelTrainer(...)
trainer.train()
```

---

## 🔧 실행 명령어

### Optuna 최적화 스크립트 (예시)

**파일:** `scripts/optimize.py`

```python
import argparse
from pathlib import Path

from src.config import ConfigLoader
from src.data import load_and_preprocess_data, DialogueSummarizationDataset
from src.optimization import OptunaOptimizer
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline_kobart", help="실험 config 이름")
    parser.add_argument("--n_trials", type=int, default=50, help="Trial 횟수")
    parser.add_argument("--timeout", type=int, default=None, help="최대 실행 시간 (초)")
    parser.add_argument("--output_dir", default="outputs/optuna_results", help="결과 저장 경로")
    args = parser.parse_args()

    # Config 로드
    config_loader = ConfigLoader()
    config = config_loader.load(args.experiment)

    # 데이터 로드
    train_df, val_df = load_and_preprocess_data("data/raw/train.csv", split_ratio=0.9)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # 데이터셋 생성
    train_dataset = DialogueSummarizationDataset(
        train_df['dialogue'].tolist(),
        train_df['summary'].tolist(),
        tokenizer,
        config
    )

    val_dataset = DialogueSummarizationDataset(
        val_df['dialogue'].tolist(),
        val_df['summary'].tolist(),
        tokenizer,
        config
    )

    # Optimizer 초기화
    optimizer = OptunaOptimizer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=f"optuna_{args.experiment}"
    )

    # 최적화 실행
    study = optimizer.optimize()

    # 결과 저장
    optimizer.save_results(args.output_dir)

    # 시각화
    try:
        optimizer.plot_optimization_history(args.output_dir)
    except ImportError:
        print("plotly가 설치되지 않아 시각화를 건너뜁니다")

    print(f"\n{'='*60}")
    print(f"최적화 완료!")
    print(f"{'='*60}")
    print(f"최적 ROUGE-L F1: {optimizer.get_best_value():.4f}")
    print(f"결과 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
```

**실행:**
```bash
# 기본 실행 (50 trials)
python scripts/optimize.py --experiment baseline_kobart

# Trial 횟수 조정
python scripts/optimize.py --experiment baseline_kobart --n_trials 100

# 시간 제한 (12시간 = 43200초)
python scripts/optimize.py --experiment baseline_kobart --timeout 43200

# 결과 디렉토리 지정
python scripts/optimize.py --experiment baseline_kobart --output_dir outputs/kobart_optuna
```

---

## 🧪 테스트

### 테스트 파일 위치
```
src/tests/test_optuna.py
```

### 테스트 실행

```bash
python src/tests/test_optuna.py
```

### 테스트 항목 (총 7개)

1. ✅ OptunaOptimizer 초기화
2. ✅ 탐색 공간 생성
3. ✅ 탐색 공간 범위 검증
4. ✅ Sampler 및 Pruner 설정
5. ✅ Best params 메서드
6. ✅ 결과 저장
7. ✅ create_optuna_optimizer 함수

**결과:** 7/7 테스트 통과 (100%)

**참고:** 실제 optimize() 테스트는 데이터셋과 모델이 필요

---

## 📊 예상 실행 시간

### 하드웨어별 예상 시간 (KoBART 기준)

| 하드웨어 | Trial당 시간 | 50 trials | 100 trials |
|---------|-------------|-----------|------------|
| A6000 (48GB) | 20-30분 | 16-25시간 | 33-50시간 |
| A100 (80GB) | 15-20분 | 12-16시간 | 25-33시간 |
| V100 (32GB) | 30-40분 | 25-33시간 | 50-66시간 |

**팁:**
- 초반 10-20 trials로 경향 파악 후 결정
- Pruning이 효과적이면 시간 단축 (약 20-30%)
- 디버그 모드로 먼저 테스트 권장

---

## ⚙️ 고급 설정

### 1. Study 저장소 (RDB)

**SQLite 사용:**
```python
optimizer = OptunaOptimizer(
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_trials=50,
    storage="sqlite:///optuna_study.db",  # SQLite DB에 저장
    study_name="kobart_optuna"
)
```

**PostgreSQL 사용:**
```python
storage = "postgresql://user:password@localhost:5432/optuna"
optimizer = OptunaOptimizer(..., storage=storage)
```

**장점:**
- 중단 후 재개 가능
- 여러 프로세스에서 동시 최적화
- 결과 영구 보존

---

### 2. 다중 목적 최적화

**ROUGE-1, ROUGE-2, ROUGE-L 동시 최적화:**

```python
def objective(trial):
    # ... 학습 및 평가

    # 다중 목적 반환
    return metrics['rouge_1_f1'], metrics['rouge_2_f1'], metrics['rouge_l_f1']

# Multi-objective study
study = optuna.create_study(
    directions=["maximize", "maximize", "maximize"]
)
```

---

### 3. 조건부 탐색 공간

**모델에 따라 다른 탐색 공간:**

```python
def create_search_space(self, trial):
    params = {}

    # 공통 파라미터
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)

    # KoBART 전용
    if 'bart' in self.config.model.name.lower():
        params['encoder_layers'] = trial.suggest_int('encoder_layers', 6, 12)
        params['decoder_layers'] = trial.suggest_int('decoder_layers', 6, 12)

    # LLM 전용
    elif 'llama' in self.config.model.name.lower() or 'qwen' in self.config.model.name.lower():
        params['lora_r'] = trial.suggest_categorical('lora_r', [8, 16, 32, 64])
        params['lora_alpha'] = trial.suggest_categorical('lora_alpha', [16, 32, 64, 128])

    return params
```

---

## ⚠️ 주의사항

### 1. GPU 메모리 관리

**문제:** Trial마다 모델 로드로 메모리 누적

**해결:**
```python
def objective(trial):
    try:
        # ... 학습
        return rouge_l_f1
    finally:
        # 명시적 메모리 해제
        import torch
        torch.cuda.empty_cache()

        # 모델 삭제
        del model
        del trainer
```

---

### 2. WandB 비활성화

**이유:** 수십 개 trial이 WandB에 로그되면 관리 어려움

**설정:**
```python
# objective 함수 내부
config.logging.use_wandb = False
```

---

### 3. 조기 종료 기준

**너무 공격적인 Pruning:**
```python
pruner = MedianPruner(
    n_startup_trials=2,   # 너무 적음
    n_warmup_steps=1      # 너무 빠름
)
# → 좋은 trial도 조기 종료될 수 있음
```

**권장 설정:**
```python
pruner = MedianPruner(
    n_startup_trials=5,   # 충분한 초기 trial
    n_warmup_steps=3      # 충분한 warmup
)
```

---

## 🔗 관련 파일

**소스 코드:**
- `src/optimization/optuna_optimizer.py` - Optuna optimizer
- `src/optimization/__init__.py` - 패키지 초기화

**테스트:**
- `src/tests/test_optuna.py` - Optuna 테스트

**문서:**
- `docs/PRD/13_Optuna_하이퍼파라미터_최적화.md` - PRD 문서
- `docs/모듈화/00_전체_시스템_개요.md` - 시스템 개요

**Config:**
- `configs/base/default.yaml` - 기본 하이퍼파라미터
- `configs/experiments/*.yaml` - 실험별 Config

---

## 📚 참고 자료

**Optuna 공식 문서:**
- https://optuna.readthedocs.io/
- Sampler: https://optuna.readthedocs.io/en/stable/reference/samplers.html
- Pruner: https://optuna.readthedocs.io/en/stable/reference/pruners.html

**TPE 논문:**
- Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"

**실전 팁:**
- https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
