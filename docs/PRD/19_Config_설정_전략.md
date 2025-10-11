# ⚙️ Config 설정 전략 (초기 모듈화 가이드)

## 🎯 목표
프로젝트 모듈화 진행 시 config 파일을 체계적으로 구성하여 실험 재현성과 유지보수성 확보

## 📊 Config 파일 분석 결과

### 1. Baseline Config (대회 제공)
**파일**: `notebooks/base/configs/config.yaml`

**특징**:
- 검증된 최소 설정 (KoBART 기준)
- 명확한 구조 (general, tokenizer, training, inference, wandb, paths)
- 필수 파라미터만 포함

**핵심 설정값**:
```yaml
training:
  learning_rate: 1.0e-05  # 1e-5 (검증됨)
  per_device_train_batch_size: 50  # 큰 배치
  num_train_epochs: 20

inference:
  no_repeat_ngram_size: 2  # 2가 최적
  num_beams: 4
```

**장점**:
- 단순하고 이해하기 쉬움
- 빠른 실험 가능
- KoBART 최적화

**단점**:
- 고급 기능 없음 (증강, 앙상블 등)
- LLM 파인튜닝 미지원
- 단일 모델만 지원

### 2. KoBART Fine-tuning Config
**파일**: `notebooks/team/CHH/configs/config_finetune_kobart.yaml`

**특징**:
- KoBART 전용 상세 설정
- 20 epoch 장기 학습
- Early stopping 포함
- 완전한 한글 주석

**핵심 추가 사항**:
```yaml
training:
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
  save_total_limit: 2
  load_best_model_at_end: true
```

**장점**:
- 베이스라인보다 안정적
- Early stopping으로 과적합 방지
- 재현성 높음

**단점**:
- 여전히 KoBART 전용
- 단일 모델 한정

### 3. Full Pipeline Config
**파일**: `notebooks/team/CHH/configs/config_full_pipeline.yaml`

**특징**:
- 가장 포괄적인 설정
- 모든 PRD 전략 통합
- LLM + Encoder-Decoder 지원
- 데이터 증강, 앙상블, 교차검증, Optuna 포함

**핵심 구조**:
```yaml
experiment:
  name: "full_pipeline"
  use_wandb: true

models:  # 다중 모델 지원
  - name: "kobart"
    type: "encoder_decoder"
  - name: "llama"
    type: "causal_lm"

data_augmentation:
  methods: ["backtranslation", "paraphrase"]

cross_validation:
  use_kfold: true
  n_splits: 5

ensemble:
  strategy: "weighted_average"

optuna:
  n_trials: 20
```

**장점**:
- 모든 전략 포함
- 고도로 설정 가능
- 프로덕션 레벨

**단점**:
- 복잡도 높음
- 초기 학습 곡선 가파름
- 실험 시간 길어짐

### 4. LLM Fine-tuning Config (검증 완료)
**파일**: `docs/참고/finetune_config.yaml`

**특징**:
- QLoRA 4-bit 양자화 설정
- 모델별 최적화 파라미터
- Chat template 토큰 자동 추가
- WandB 구조화 로깅

**검증된 성능**:
```yaml
# KoBART: ROUGE Sum 94.51
# Llama-3.2-Korean-3B: 진행 중 (목표 95+)

models:
  - model_name: "Bllossom/llama-3.2-Korean-Bllossom-3B"
    learning_rate: 2.0e-5     # 검증된 LLM 학습률
    batch_size: 8             # QLoRA 4-bit 최적값
    lora_dropout: 0.05        # 과적합 방지
    use_bf16: true            # Llama 권장 dtype

tokenizer:
  encoder_max_len: 1024       # Prompt truncation 0.11% (512는 6.07%)
  decoder_max_len: 200        # 여유 확보

lora:
  r: 16
  lora_alpha: 32
  target_modules:             # Attention + MLP 모두 포함
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  num_train_epochs: 3         # LLM은 3 epoch 충분
  gradient_accumulation_steps: 8
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  max_grad_norm: 1.2
  weight_decay: 0.1
```

**장점**:
- 실전 검증된 파라미터
- 치명적 이슈 해결 (Prompt truncation, Chat template 등)
- 메모리 효율적

## 🏗️ 초기 모듈화를 위한 Config 구조 전략

### 전략 1: 계층적 Config 시스템 (권장)

```
configs/
├── base/
│   ├── default.yaml              # 공통 기본 설정
│   ├── encoder_decoder.yaml      # Encoder-Decoder 공통
│   └── causal_lm.yaml            # Causal LM 공통
├── models/
│   ├── kobart.yaml               # KoBART 전용
│   ├── llama_3.2_3b.yaml         # Llama-3.2-3B 전용
│   ├── qwen3_4b.yaml             # Qwen3-4B 전용
│   └── qwen2.5_7b.yaml           # Qwen2.5-7B 전용
├── strategies/
│   ├── data_augmentation.yaml    # 데이터 증강
│   ├── ensemble.yaml             # 앙상블
│   ├── cross_validation.yaml     # 교차검증
│   └── optuna.yaml               # 하이퍼파라미터 최적화
├── experiments/
│   ├── baseline_kobart.yaml      # 실험 1: 베이스라인
│   ├── llama_finetune.yaml       # 실험 2: Llama 파인튜닝
│   ├── multi_model.yaml          # 실험 3: 다중 모델
│   └── full_pipeline.yaml        # 실험 4: 전체 파이프라인
└── inference/
    └── production.yaml            # 프로덕션 추론 설정
```

### 전략 2: Config 병합 메커니즘

```python
# config_loader.py
from omegaconf import OmegaConf

def load_config(experiment_config_path):
    """
    계층적 Config 병합

    실행 순서:
    1. base/default.yaml 로드
    2. base/{model_type}.yaml 로드 및 병합
    3. models/{model_name}.yaml 로드 및 병합
    4. strategies/*.yaml 로드 및 병합 (활성화된 전략만)
    5. experiments/{experiment}.yaml 로드 및 병합
    """
    # 1. 기본 설정
    base_config = OmegaConf.load("configs/base/default.yaml")

    # 2. 실험 설정
    exp_config = OmegaConf.load(experiment_config_path)

    # 3. 모델 타입별 설정
    model_type = exp_config.model.type
    type_config = OmegaConf.load(f"configs/base/{model_type}.yaml")

    # 4. 모델별 설정
    model_name = exp_config.model.name
    model_config = OmegaConf.load(f"configs/models/{model_name}.yaml")

    # 5. 전략 설정 (활성화된 것만)
    strategy_configs = []
    if exp_config.get('data_augmentation', {}).get('enabled', False):
        strategy_configs.append(OmegaConf.load("configs/strategies/data_augmentation.yaml"))
    if exp_config.get('ensemble', {}).get('enabled', False):
        strategy_configs.append(OmegaConf.load("configs/strategies/ensemble.yaml"))

    # 6. 병합 (나중 것이 우선)
    merged = OmegaConf.merge(
        base_config,
        type_config,
        model_config,
        *strategy_configs,
        exp_config
    )

    return merged
```

### 전략 3: 단계별 Config 도입 로드맵

#### Phase 1: 베이스라인 검증 (1-3일)
**목표**: 대회 베이스라인 재현

**사용 Config**:
- `configs/experiments/baseline_kobart.yaml` (단일 파일)

**내용**:
```yaml
general:
  model_name: digit82/kobart-summarization
  seed: 42

tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100

training:
  learning_rate: 1.0e-05
  per_device_train_batch_size: 50
  num_train_epochs: 20

inference:
  no_repeat_ngram_size: 2
  num_beams: 4
```

**검증 목표**: ROUGE Sum ≥ 94.51

#### Phase 2: 단일 모델 최적화 (4-7일)
**목표**: KoBART 성능 극대화 + LLM 파인튜닝 시작

**사용 Config**:
- `configs/base/default.yaml` (공통)
- `configs/base/encoder_decoder.yaml` (타입별)
- `configs/base/causal_lm.yaml` (타입별)
- `configs/models/kobart.yaml` (모델별)
- `configs/models/llama_3.2_3b.yaml` (모델별)
- `configs/experiments/kobart_optimized.yaml` (실험)
- `configs/experiments/llama_finetune.yaml` (실험)

**추가 기능**:
- Early stopping
- Learning rate scheduler
- Gradient accumulation
- WandB 통합

**검증 목표**: KoBART ≥ 95, Llama ≥ 95

#### Phase 3: 전략 통합 (8-12일)
**목표**: 데이터 증강 + 교차검증 적용

**추가 Config**:
- `configs/strategies/data_augmentation.yaml`
- `configs/strategies/cross_validation.yaml`

**검증 목표**: ROUGE Sum ≥ 96-97

#### Phase 4: 앙상블 및 최적화 (13-15일)
**목표**: 다중 모델 앙상블 + Optuna

**추가 Config**:
- `configs/strategies/ensemble.yaml`
- `configs/strategies/optuna.yaml`
- `configs/experiments/multi_model.yaml`
- `configs/experiments/full_pipeline.yaml`

**검증 목표**: ROUGE Sum ≥ 98-100

## 📋 Config 파일 구조 표준

### 1. Base Config (default.yaml)
```yaml
experiment:
  name: "default"
  seed: 42
  use_wandb: false
  wandb_project: "dialogue-summarization"
  wandb_entity: null

paths:
  train_data: "data/train.csv"
  dev_data: "data/dev.csv"
  test_data: "data/test.csv"
  output_dir: "outputs"
  model_save_dir: "models"

logging:
  log_level: "INFO"
  log_dir: "logs"
  save_steps: 100
  logging_steps: 10

strategies:
  data_augmentation: false
  cross_validation: false
  ensemble: false
  optuna: false
```

### 2. Model Type Config (encoder_decoder.yaml)
```yaml
model:
  type: "encoder_decoder"
  architecture: "bart"

tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100
  special_tokens:
    - '#Person1#'
    - '#Person2#'
    - '#Person3#'
    - '#Person4#'
    - '#Person5#'
    - '#Person6#'
    - '#Person7#'
    - '#PhoneNumber#'
    - '#Address#'
    - '#DateOfBirth#'
    - '#PassportNumber#'
    - '#SSN#'
    - '#CardNumber#'
    - '#CarNumber#'
    - '#Email#'

training:
  learning_rate: 1.0e-05
  per_device_train_batch_size: 50
  per_device_eval_batch_size: 32
  num_train_epochs: 20
  warmup_ratio: 0.1
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  optim: "adamw_torch"
  gradient_accumulation_steps: 1
  fp16: true
  gradient_checkpointing: false
  max_grad_norm: 1.0

  early_stopping_patience: 3
  early_stopping_threshold: 0.001

  save_strategy: "epoch"
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: "rouge_sum"
  greater_is_better: true

inference:
  batch_size: 32
  num_beams: 4
  no_repeat_ngram_size: 2
  early_stopping: true
  generate_max_length: 100
  remove_tokens:
    - '<usr>'
    - '<s>'
    - '</s>'
    - '<pad>'

evaluation:
  strategy: "epoch"
  metric: "rouge"
```

### 3. Model Specific Config (kobart.yaml)
```yaml
model:
  name: "kobart"
  checkpoint: "digit82/kobart-summarization"
  type: "encoder_decoder"
  architecture: "bart"

training:
  per_device_train_batch_size: 50
  learning_rate: 1.0e-05

inference:
  no_repeat_ngram_size: 2
```

### 4. Model Specific Config (llama_3.2_3b.yaml)
```yaml
model:
  name: "llama_3.2_3b"
  checkpoint: "Bllossom/llama-3.2-Korean-Bllossom-3B"
  type: "causal_lm"
  architecture: "llama"

  quantization:
    load_in_4bit: true
    bnb_4bit_use_double_quant: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "bfloat16"

  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"

tokenizer:
  encoder_max_len: 1024
  decoder_max_len: 200
  chat_template_type: "llama"
  chat_template_tokens:
    - "<|start_header_id|>"
    - "<|end_header_id|>"
    - "<|eot_id|>"

training:
  num_train_epochs: 3
  learning_rate: 2.0e-05
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 8

  optim: "paged_adamw_32bit"
  weight_decay: 0.1
  max_grad_norm: 1.2
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"

  bf16: true
  fp16: false
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  early_stopping_patience: 2
  metric_for_best_model: "eval_loss"
  greater_is_better: false

generation:
  max_new_tokens: 150
  min_new_tokens: 10
  do_sample: true
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  no_repeat_ngram_size: 3
  repetition_penalty: 1.1
```

### 5. Strategy Config (data_augmentation.yaml)
```yaml
data_augmentation:
  enabled: true

  methods:
    backtranslation:
      enabled: true
      model: "Helsinki-NLP/opus-mt-ko-en"
      ratio: 0.3

    paraphrase:
      enabled: true
      model: "beomi/KcELECTRA-base-v2022"
      ratio: 0.3

    dialogue_sampling:
      enabled: true
      min_turns: 3
      max_turns: 10
      ratio: 0.2

    synonym_replacement:
      enabled: false
      ratio: 0.1

  augmentation_factor: 2.0
  preserve_original: true

  cache_augmented: true
  cache_dir: "data/augmented_cache"
```

### 6. Experiment Config (baseline_kobart.yaml)
```yaml
experiment:
  name: "baseline_kobart"
  description: "대회 베이스라인 재현 실험"
  seed: 42
  use_wandb: true
  wandb_project: "dialogue-summarization"
  wandb_tags: ["baseline", "kobart"]

model:
  name: "kobart"
  checkpoint: "digit82/kobart-summarization"

strategies:
  data_augmentation: false
  cross_validation: false
  ensemble: false
  optuna: false

paths:
  output_dir: "outputs/baseline_kobart"
  model_save_dir: "models/baseline_kobart"
```

## 🔧 Config 로더 구현 예제

### 1. 기본 로더
```python
# src/config/loader.py
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """계층적 Config 로더"""

    def __init__(self, config_root: str = "configs"):
        self.config_root = Path(config_root)

    def load_base(self) -> Dict[str, Any]:
        """기본 설정 로드"""
        return OmegaConf.load(self.config_root / "base" / "default.yaml")

    def load_model_type(self, model_type: str) -> Dict[str, Any]:
        """모델 타입별 설정 로드"""
        path = self.config_root / "base" / f"{model_type}.yaml"
        if path.exists():
            return OmegaConf.load(path)
        return {}

    def load_model(self, model_name: str) -> Dict[str, Any]:
        """모델별 설정 로드"""
        path = self.config_root / "models" / f"{model_name}.yaml"
        if path.exists():
            return OmegaConf.load(path)
        return {}

    def load_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """전략 설정 로드"""
        path = self.config_root / "strategies" / f"{strategy_name}.yaml"
        if path.exists():
            return OmegaConf.load(path)
        return {}

    def load_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """실험 설정 로드"""
        path = self.config_root / "experiments" / f"{experiment_name}.yaml"
        return OmegaConf.load(path)

    def merge_configs(self, experiment_name: str) -> OmegaConf:
        """모든 Config를 계층적으로 병합"""
        exp_config = self.load_experiment(experiment_name)
        configs = [self.load_base()]

        model_type = exp_config.get('model', {}).get('type', 'encoder_decoder')
        configs.append(self.load_model_type(model_type))

        model_name = exp_config.get('model', {}).get('name', '')
        if model_name:
            configs.append(self.load_model(model_name))

        strategies = exp_config.get('strategies', {})
        for strategy_name, enabled in strategies.items():
            if enabled:
                configs.append(self.load_strategy(strategy_name))

        configs.append(exp_config)
        merged = OmegaConf.merge(*configs)

        return merged
```

### 2. CLI 인터페이스
```python
# train.py
import argparse
from src.config.loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='실험 config 이름'
    )
    parser.add_argument(
        '--override',
        type=str,
        nargs='*',
        help='오버라이드 설정'
    )
    args = parser.parse_args()

    loader = ConfigLoader()
    config = loader.merge_configs(args.config)

    if args.override:
        overrides = OmegaConf.from_dotlist(args.override)
        config = OmegaConf.merge(config, overrides)

    trainer = Trainer(config)
    trainer.train()
```

## 💡 Config 설계 Best Practices

### 1. 명확한 구분
```yaml
experiment:
  name: "baseline"

model:
  name: "kobart"

training:
  learning_rate: 1e-5
```

### 2. 검증된 기본값
```yaml
training:
  learning_rate: 1.0e-05
  per_device_train_batch_size: 50
```

### 3. 조건부 설정
```yaml
strategies:
  data_augmentation: false
  ensemble: false

data_augmentation:
  enabled: false
  methods: [...]
```

### 4. 타입 힌트 (주석)
```yaml
training:
  learning_rate: 1.0e-05
  per_device_train_batch_size: 50
  num_train_epochs: 20
```

## 🚀 실행 계획

### Week 1
- [x] Config 파일 분석 완료
- [ ] 계층적 Config 구조 설계
- [ ] `configs/base/` 작성
- [ ] `configs/models/` 작성
- [ ] `configs/experiments/baseline_kobart.yaml` 작성
- [ ] ConfigLoader 구현 및 테스트

### Week 2
- [ ] 베이스라인 재현
- [ ] `configs/base/causal_lm.yaml` 작성
- [ ] `configs/models/llama_3.2_3b.yaml` 작성
- [ ] `configs/experiments/llama_finetune.yaml` 작성
- [ ] Llama 파인튜닝 시작

### Week 3
- [ ] `configs/strategies/` 작성
- [ ] `configs/experiments/multi_model.yaml` 작성
- [ ] `configs/experiments/full_pipeline.yaml` 작성
- [ ] 최종 앙상블 실험

## 📊 예상 효과

### 재현성
완벽한 재현 가능

### 실험 속도
Config 변경으로 즉시 실행

### 유지보수성
중앙 집중식 Config 관리

### 협업
Config 파일만 공유

## 🔥 핵심 권장사항

1. 단계적 도입
2. 검증 우선
3. 문서화
4. 버전 관리
5. 네이밍 규칙
