# LLM 파인튜닝 시스템 상세 가이드

## 📋 목차
1. [개요](#개요)
2. [LoRA Loader](#lora-loader)
3. [LLM Dataset](#llm-dataset)
4. [LLM Trainer](#llm-trainer)
5. [사용 방법](#사용-방법)
6. [실행 명령어](#실행-명령어)

---

## 📝 개요

### 목적
- Causal LM (Llama, Qwen) 파인튜닝
- QLoRA 4-bit 양자화 지원
- LoRA (Low-Rank Adaptation) 지원
- Instruction/Chat Format 지원

### 핵심 기능
- ✅ QLoRA 4-bit 양자화
- ✅ LoRA 파라미터 효율적 학습
- ✅ Chat template 토큰 자동 추가
- ✅ Prompt truncation 방지
- ✅ Instruction Tuning 데이터 증강

---

## 🏗️ LoRA Loader

### 파일 위치
```
src/models/lora_loader.py
```

### 클래스 구조

```python
class LoRALoader:
    def __init__(self, config, logger=None)
    def load_model_and_tokenizer(use_lora=True, use_qlora=False)
    def _load_tokenizer(checkpoint)
    def _create_bnb_config()
    def _load_causal_lm(checkpoint, bnb_config)
    def _add_chat_tokens(model, tokenizer)
    def _apply_lora(model)
    def _configure_tokenizer(tokenizer)
```

### 주요 기능

#### 1. QLoRA 4-bit 양자화
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # Llama: bf16, Qwen: fp16
)
```

#### 2. LoRA 설정
```python
lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # alpha = r * 2
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
```

#### 3. Chat Template 토큰 추가
```python
# Llama 모델
chat_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]

# Qwen 모델
chat_tokens = ["<|im_start|>", "<|im_end|>"]

tokenizer.add_special_tokens({'additional_special_tokens': chat_tokens})
model.resize_token_embeddings(len(tokenizer))
```

#### 4. Prompt Truncation 방지
```python
# Left padding/truncation (Causal LM 필수)
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
```

---

## 📊 LLM Dataset

### 파일 위치
```
src/data/llm_dataset.py
```

### 클래스 구조

```python
class LLMSummarizationDataset(Dataset):
    def __init__(dialogues, summaries, tokenizer,
                 encoder_max_len=1024, decoder_max_len=200,
                 format_type="instruction")
    def __getitem__(idx)
    def _format_instruction(dialogue, summary)
    def _format_chat(dialogue, summary)

class InstructionAugmentedDataset(Dataset):
    # 5가지 instruction 템플릿으로 데이터 증강
```

### 데이터 포맷

#### Instruction Format
```
### Instruction:
다음 대화를 간결하게 요약해주세요.

### Input:
{dialogue}

### Response:
{summary}
```

#### Chat Format (Llama)
```
<|start_header_id|>system<|end_header_id|>
당신은 대화를 요약하는 전문가입니다.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
다음 대화를 요약해주세요:
{dialogue}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{summary}<|eot_id|>
```

---

## 🚀 LLM Trainer

### 파일 위치
```
src/training/llm_trainer.py
```

### 클래스 구조

```python
class LLMTrainer:
    def __init__(config, model, tokenizer, train_dataset, eval_dataset)
    def train()
    def evaluate()
    def _create_training_args()
    def _create_trainer()
```

### 학습 설정 (QLoRA 최적화)

```python
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,      # effective batch=64
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.1,
    max_grad_norm=1.2,

    # QLoRA 최적화
    optim="paged_adamw_32bit",
    bf16=True,                          # Llama: bf16
    gradient_checkpointing=True,

    # 평가 및 저장
    eval_strategy='epoch',
    save_strategy='epoch',
    metric_for_best_model='eval_loss',  # Causal LM은 loss 사용
    greater_is_better=False
)
```

---

## 💻 사용 방법

### 1. Config 기반 학습

```python
from src.config import load_config
from src.models.lora_loader import load_lora_model_and_tokenizer
from src.data.llm_dataset import create_llm_dataset
from src.training.llm_trainer import create_llm_trainer
import pandas as pd

# 1. Config 로드
config = load_config("llama_3.2_3b")

# 2. 모델 및 토크나이저 로드
model, tokenizer = load_lora_model_and_tokenizer(
    config,
    use_lora=True,
    use_qlora=True
)

# 3. 데이터 로드
train_df = pd.read_csv("data/raw/train.csv")
eval_df = pd.read_csv("data/raw/dev.csv")

# 4. Dataset 생성
train_dataset = create_llm_dataset(
    dialogues=train_df['dialogue'].tolist(),
    summaries=train_df['summary'].tolist(),
    tokenizer=tokenizer,
    encoder_max_len=1024,  # Prompt truncation 방지
    decoder_max_len=200,
    format_type="chat"
)

# 5. Trainer 생성 및 학습
trainer = create_llm_trainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 6. 학습 실행
results = trainer.train()
```

---

## 🔧 실행 명령어

### 1. 기본 LLM 학습 (Llama-3.2-3B + QLoRA)

```bash
python scripts/train_llm.py --experiment llama_3.2_3b --use_qlora
```

**결과 파일:**
- 모델: `outputs/llama_3.2_3b_qlora/final_model/`
- 로그: `logs/YYYYMMDD/train/train_llm_llama_3.2_3b_YYYYMMDD_HHMMSS.log`

### 2. Qwen 모델 학습

```bash
python scripts/train_llm.py --experiment qwen3_4b --use_qlora
```

**결과 파일:**
- 모델: `outputs/qwen3_4b_qlora/final_model/`
- 로그: `logs/YYYYMMDD/train/train_llm_qwen3_4b_YYYYMMDD_HHMMSS.log`

### 3. Instruction Tuning (데이터 5배 증강)

```bash
python scripts/train_llm.py --experiment llama_3.2_3b --use_qlora --use_instruction_augmentation
```

**효과:**
- 학습 데이터: 12,457개 → 62,285개 (5배)
- 5가지 instruction 템플릿 적용

### 4. 디버그 모드 (빠른 테스트)

```bash
python scripts/train_llm.py --experiment llama_3.2_3b --use_qlora --debug
```

**디버그 모드 설정:**
- 데이터: 학습 50개, 검증 10개
- 에포크: 1회
- 배치 크기: 2
- WandB: 비활성화

---

## 📂 Config 파일

### Causal LM 기본 설정
**파일:** `configs/base/causal_lm.yaml`

```yaml
model:
  type: causal_lm
  checkpoint: "Bllossom/llama-3.2-Korean-Bllossom-3B"

lora:
  r: 16
  alpha: 32
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  dropout: 0.05
  use_qlora: true

tokenizer:
  encoder_max_len: 1024  # Prompt truncation 방지
  decoder_max_len: 200

training:
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.1
  max_grad_norm: 1.2
  gradient_checkpointing: true
```

### Llama 모델 설정
**파일:** `configs/models/llama_3.2_3b.yaml`

```yaml
_base_: ../base/causal_lm.yaml

model:
  checkpoint: "Bllossom/llama-3.2-Korean-Bllossom-3B"
  size: "3B"
  dtype: "bf16"
  chat_template: "llama"

dataset:
  format_type: "chat"

experiment:
  name: "llama_3.2_3b_qlora"
  tags:
    - "llama"
    - "3b"
    - "qlora"
```

### Qwen 모델 설정
**파일:** `configs/models/qwen3_4b.yaml`

```yaml
_base_: ../base/causal_lm.yaml

model:
  checkpoint: "Qwen/Qwen3-4B-Instruct-2507"
  size: "4B"
  dtype: "fp16"
  chat_template: "qwen"

training:
  batch_size: 6
  gradient_accumulation_steps: 10

dataset:
  format_type: "chat"

experiment:
  name: "qwen3_4b_qlora"
  tags:
    - "qwen"
    - "4b"
    - "qlora"
```

---

## 🧪 테스트

### 테스트 파일 위치
```
src/tests/test_lora_loader.py
```

### 테스트 실행

```bash
python src/tests/test_lora_loader.py
```

### 테스트 항목 (총 4개)

1. ✅ LoRALoader 초기화
2. ✅ 토크나이저 로딩
3. ✅ LoRA Config 생성
4. ✅ 편의 함수

**결과:** 4/4 테스트 통과 (100%)

---

## 📊 예상 리소스

### GPU 메모리 사용량

| 모델 | 파라미터 | QLoRA (4-bit) | Full Fine-tuning |
|------|----------|---------------|------------------|
| Llama-3.2-3B | 3B | **8GB** | 24GB |
| Qwen3-4B | 4B | **10GB** | 32GB |
| Llama-3-8B | 8B | **16GB** | 64GB |

### 학습 시간 (A6000 기준)

| 모델 | 에포크 | 데이터 | 예상 시간 |
|------|--------|--------|-----------|
| Llama-3.2-3B | 3 | 12,457개 | 6-8시간 |
| Qwen3-4B | 3 | 12,457개 | 8-10시간 |

---

## 🎯 성능 목표

### Zero-shot 성능 (검증됨)

| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | **ROUGE Sum** |
|------|---------|---------|---------|---------------|
| Llama-3.2-Korean-3B | 26.96 | 11.08 | 24.22 | **49.52** (1위) |
| Qwen3-4B | 24.22 | 9.23 | 21.79 | **45.02** (4위) |

### 파인튜닝 목표

| 모델 | Zero-shot | 파인튜닝 목표 | 개선 목표 |
|------|-----------|---------------|-----------|
| Llama-3.2-Korean-3B | 49.52 | **95+** | +45 포인트 |
| Qwen3-4B | 45.02 | **95+** | +50 포인트 |

---

## 🔗 관련 파일

**소스 코드:**
- `src/models/lora_loader.py` - LoRA Loader
- `src/data/llm_dataset.py` - LLM Dataset
- `src/training/llm_trainer.py` - LLM Trainer

**Config:**
- `configs/base/causal_lm.yaml` - Causal LM 기본 설정
- `configs/models/llama_3.2_3b.yaml` - Llama 모델
- `configs/models/qwen3_4b.yaml` - Qwen 모델

**스크립트:**
- `scripts/train_llm.py` - LLM 학습 스크립트

**테스트:**
- `src/tests/test_lora_loader.py` - LoRA Loader 테스트
