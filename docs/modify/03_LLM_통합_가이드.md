# 🤖 LLM 파인튜닝 통합 가이드

**⚠️ 상태**: 선택적 (2025-10-11 업데이트)
**우선순위**: ⚠️ 선택적 고급 기능 (PRD 08번)
**예상 작업 시간**: 4-6시간
**난이도**: ★★★★☆
**선행 작업**: 02_실행_옵션_시스템_구현_가이드.md

> **📌 참고**: 이 문서는 선택적 고급 기능에 대한 가이드입니다.
> 현재 scripts/train_llm.py가 별도로 존재하여 LLM 파인튜닝이 가능하며,
> train.py와의 통합은 필요시 진행하면 됩니다.

---

## 📋 문제 정의

### 현재 상황
```
scripts/
├── train.py          # Encoder-Decoder 전용 (KoBART)
└── train_llm.py      # Causal LM 전용 (Llama, Qwen) - 분리됨!
```

**문제점**:
1. 두 개의 별도 스크립트 → 사용자 혼란
2. `train.py`에서 LLM 사용 불가능
3. 통합 파이프라인 구축 불가
4. PRD 08번 요구사항 미충족

### 목표 상태
```python
# 하나의 인터페이스로 모든 모델 사용
python train.py --mode single --models kobart
python train.py --mode single --models llama-3.2-korean-3b
python train.py --mode multi_model --models kobart llama-3.2-korean-3b qwen3-4b
```

---

## 🏗️ 통합 아키텍처

### 1. 모델 타입 분리

```
src/models/
├── __init__.py                    # 통합 인터페이스
├── model_loader.py                # 기존 (Encoder-Decoder)
├── llm_loader.py                  # 신규 (Causal LM)
├── model_config.py                # 모델별 설정
└── generation/
    ├── encoder_decoder_generator.py
    └── causal_lm_generator.py
```

### 2. Config 구조 확장

```yaml
# configs/models/kobart.yaml
model:
  name: "kobart"
  type: "encoder_decoder"  # ← 타입 추가!
  checkpoint: "digit82/kobart-summarization"

# configs/models/llama_3.2_3b.yaml
model:
  name: "llama-3.2-korean-3b"
  type: "causal_lm"  # ← 타입 추가!
  checkpoint: "Bllossom/llama-3.2-Korean-Bllossom-3B"

  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true

  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
```

---

## 📝 구현 단계

### Phase 1: LLM Loader 분리 (2시간)

#### src/models/llm_loader.py (신규 생성)

```python
# src/models/llm_loader.py
"""Causal LM 모델 로더 (Llama, Qwen, 등)"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_causal_lm(config, logger=None):
    """
    Causal LM 모델 로드 (QLoRA 포함)

    Args:
        config: 모델 설정
        logger: 로거

    Returns:
        model, tokenizer
    """
    if logger:
        logger.write(f"Loading Causal LM: {config.model.checkpoint}")

    # 1. Quantization 설정
    quantization_config = None
    if config.model.get('quantization'):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.model.quantization.get('load_in_4bit', True),
            bnb_4bit_compute_dtype=getattr(torch, config.model.quantization.get('bnb_4bit_compute_dtype', 'bfloat16')),
            bnb_4bit_quant_type=config.model.quantization.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=config.model.quantization.get('bnb_4bit_use_double_quant', True)
        )

    # 2. 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        config.model.checkpoint,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.training.get('bf16', True) else torch.float16,
        trust_remote_code=True
    )

    # 3. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.checkpoint,
        trust_remote_code=True
    )

    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Chat 템플릿 토큰 추가
    if config.tokenizer.get('chat_template_tokens'):
        special_tokens = {
            'additional_special_tokens': config.tokenizer.chat_template_tokens
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # 4. LoRA 설정
    if config.model.get('lora'):
        if logger:
            logger.write("Applying LoRA configuration...")

        # K-bit training 준비
        model = prepare_model_for_kbit_training(model)

        # LoRA Config
        lora_config = LoraConfig(
            r=config.model.lora.get('r', 16),
            lora_alpha=config.model.lora.get('lora_alpha', 32),
            lora_dropout=config.model.lora.get('lora_dropout', 0.05),
            bias=config.model.lora.get('bias', 'none'),
            task_type=config.model.lora.get('task_type', 'CAUSAL_LM'),
            target_modules=config.model.lora.get('target_modules', [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ])
        )

        # PEFT 모델 생성
        model = get_peft_model(model, lora_config)

        # 학습 가능한 파라미터 출력
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params

        if logger:
            logger.write(f"  Trainable params: {trainable_params:,} ({trainable_percentage:.2f}%)")
            logger.write(f"  Total params: {total_params:,}")

    # 5. Gradient Checkpointing
    if config.training.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Gradient checkpointing과 함께 사용 시 필수

    return model, tokenizer


def format_llm_prompt(dialogue: str, tokenizer) -> str:
    """
    LLM용 프롬프트 포맷팅

    Args:
        dialogue: 대화 원문
        tokenizer: 토크나이저

    Returns:
        포맷팅된 프롬프트
    """
    # Chat 템플릿 사용 (Llama 3.x 스타일)
    messages = [
        {
            "role": "system",
            "content": "You are an expert in dialogue summarization. Summarize the given dialogue concisely and accurately."
        },
        {
            "role": "user",
            "content": f"Dialogue:\n{dialogue}\n\nSummary:"
        }
    ]

    # Chat 템플릿 적용
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback: 수동 포맷팅
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{messages[0]['content']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{messages[1]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt
```

---

### Phase 2: 통합 Model Loader 수정 (1시간)

#### src/models/__init__.py 수정

```python
# src/models/__init__.py
"""모델 로더 통합 인터페이스"""

from src.models.model_loader import load_encoder_decoder
from src.models.llm_loader import load_causal_lm


def load_model_and_tokenizer(config, logger=None):
    """
    모델 타입에 따라 적절한 로더 선택

    Args:
        config: 모델 설정
        logger: 로거

    Returns:
        model, tokenizer

    Raises:
        ValueError: 지원하지 않는 모델 타입
    """
    model_type = config.model.get('type', 'encoder_decoder')

    if logger:
        logger.write(f"Model Type: {model_type}")

    if model_type == 'encoder_decoder':
        return load_encoder_decoder(config, logger)
    elif model_type == 'causal_lm':
        return load_causal_lm(config, logger)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


__all__ = [
    'load_model_and_tokenizer',
    'load_encoder_decoder',
    'load_causal_lm'
]
```

#### src/models/model_loader.py 함수명 변경

```python
# src/models/model_loader.py
"""Encoder-Decoder 모델 로더 (KoBART 등)"""

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)


def load_encoder_decoder(config, logger=None):
    """
    Encoder-Decoder 모델 로드

    Args:
        config: 모델 설정
        logger: 로거

    Returns:
        model, tokenizer
    """
    if logger:
        logger.write(f"Loading Encoder-Decoder: {config.model.checkpoint}")

    # 기존 load_model_and_tokenizer 코드와 동일
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.model.checkpoint)

    # Special tokens 추가
    if config.tokenizer.get('special_tokens'):
        special_tokens = {'additional_special_tokens': config.tokenizer.special_tokens}
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
```

---

### Phase 3: Dataset 및 Trainer 수정 (2시간)

#### src/data/dataset.py 수정 (LLM 지원)

```python
# src/data/dataset.py
"""Dataset 클래스 - Encoder-Decoder와 Causal LM 모두 지원"""

import torch
from torch.utils.data import Dataset


class DialogueSummarizationDataset(Dataset):
    """대화 요약 Dataset (Encoder-Decoder & Causal LM 지원)"""

    def __init__(
        self,
        dialogues,
        summaries,
        tokenizer,
        encoder_max_len=512,
        decoder_max_len=100,
        preprocess=True,
        model_type='encoder_decoder'  # ← 추가!
    ):
        self.dialogues = dialogues
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.preprocess = preprocess
        self.model_type = model_type  # ← 추가!

        # 전처리
        if self.preprocess:
            from src.data.preprocessor import preprocess_dialogue, preprocess_summary
            self.dialogues = [preprocess_dialogue(d) for d in dialogues]
            self.summaries = [preprocess_summary(s) for s in summaries]

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]

        if self.model_type == 'encoder_decoder':
            return self._get_encoder_decoder_item(dialogue, summary)
        elif self.model_type == 'causal_lm':
            return self._get_causal_lm_item(dialogue, summary)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _get_encoder_decoder_item(self, dialogue, summary):
        """Encoder-Decoder용 데이터"""
        # 기존 로직과 동일
        encoder_inputs = self.tokenizer(
            dialogue,
            max_length=self.encoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        decoder_inputs = self.tokenizer(
            summary,
            max_length=self.decoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = decoder_inputs['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': encoder_inputs['input_ids'].squeeze(),
            'attention_mask': encoder_inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

    def _get_causal_lm_item(self, dialogue, summary):
        """Causal LM용 데이터 (Instruction Tuning 스타일)"""
        from src.models.llm_loader import format_llm_prompt

        # 프롬프트 생성
        prompt = format_llm_prompt(dialogue, self.tokenizer)
        full_text = prompt + summary + self.tokenizer.eos_token

        # 토크나이징
        encoding = self.tokenizer(
            full_text,
            max_length=self.encoder_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Labels 생성 (프롬프트 부분은 -100으로 마스킹)
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.encoder_max_len,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]

        labels = encoding['input_ids'].clone()
        labels[:, :prompt_length] = -100  # 프롬프트 부분 마스킹
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
```

#### src/training/trainer_factory.py 수정

```python
# src/training/trainer_factory.py
"""Trainer 생성 팩토리"""

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments
)
from src.training.callbacks import create_callbacks


def create_trainer(
    config,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    use_wandb=False,
    logger=None,
    experiment_name=None
):
    """
    모델 타입에 따라 적절한 Trainer 생성

    Args:
        config: 학습 설정
        model: 모델
        tokenizer: 토크나이저
        train_dataset: 학습 데이터셋
        eval_dataset: 검증 데이터셋
        use_wandb: WandB 사용 여부
        logger: 로거
        experiment_name: 실험명

    Returns:
        Trainer
    """
    model_type = config.model.get('type', 'encoder_decoder')

    if model_type == 'encoder_decoder':
        return _create_seq2seq_trainer(
            config, model, tokenizer, train_dataset, eval_dataset,
            use_wandb, logger, experiment_name
        )
    elif model_type == 'causal_lm':
        return _create_causal_lm_trainer(
            config, model, tokenizer, train_dataset, eval_dataset,
            use_wandb, logger, experiment_name
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _create_seq2seq_trainer(config, model, tokenizer, train_dataset, eval_dataset,
                              use_wandb, logger, experiment_name):
    """Seq2Seq Trainer (기존 코드)"""
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.paths.model_save_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        # ... 기존 설정
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # ... 기존 설정
    )

    return trainer


def _create_causal_lm_trainer(config, model, tokenizer, train_dataset, eval_dataset,
                                use_wandb, logger, experiment_name):
    """Causal LM Trainer (신규)"""
    training_args = TrainingArguments(
        output_dir=config.paths.model_save_dir,
        num_train_epochs=config.training.get('num_train_epochs', 3),  # LLM은 3 epoch
        per_device_train_batch_size=config.training.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=config.training.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=config.training.get('gradient_accumulation_steps', 8),
        learning_rate=config.training.get('learning_rate', 2e-5),
        warmup_ratio=config.training.get('warmup_ratio', 0.1),
        weight_decay=config.training.get('weight_decay', 0.1),
        lr_scheduler_type=config.training.get('lr_scheduler_type', 'cosine'),
        max_grad_norm=config.training.get('max_grad_norm', 1.2),
        bf16=config.training.get('bf16', True),
        fp16=config.training.get('fp16', False),
        logging_steps=10,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=['wandb'] if use_wandb else [],
        run_name=experiment_name if use_wandb else None,
        gradient_checkpointing=config.training.get('gradient_checkpointing', True),
        optim=config.training.get('optim', 'paged_adamw_32bit'),
    )

    # Callbacks
    callbacks = create_callbacks(config, logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks
    )

    return trainer
```

---

## 🔧 구현 체크리스트

### Phase 1: LLM Loader 생성 (2시간)
- [ ] `src/models/llm_loader.py` 생성
- [ ] `load_causal_lm()` 함수 구현
- [ ] QLoRA 설정 구현
- [ ] Chat 템플릿 포맷팅 구현

### Phase 2: 통합 (1시간)
- [ ] `src/models/__init__.py` 수정
- [ ] `src/models/model_loader.py` 함수명 변경
- [ ] 모델 타입 기반 라우팅 구현

### Phase 3: Dataset & Trainer 수정 (2시간)
- [ ] `DialogueSummarizationDataset`에 `model_type` 파라미터 추가
- [ ] `_get_causal_lm_item()` 메서드 구현
- [ ] `create_trainer()`에 Causal LM 분기 추가
- [ ] `_create_causal_lm_trainer()` 구현

### Phase 4: Config 파일 생성 (1시간)
- [ ] `configs/models/llama_3.2_3b.yaml` 생성
- [ ] `configs/models/qwen3_4b.yaml` 생성
- [ ] `configs/base/causal_lm.yaml` 생성

---

## 🚀 즉시 시작

### 1. 파일 생성
```bash
cd /home/ieyeppo/AI_Lab/natural-language-processing-competition

touch src/models/llm_loader.py

mkdir -p configs/models
touch configs/models/llama_3.2_3b.yaml
touch configs/models/qwen3_4b.yaml

mkdir -p configs/base
touch configs/base/causal_lm.yaml
```

### 2. llm_loader.py 작성
위의 코드를 `src/models/llm_loader.py`에 붙여넣기

### 3. __init__.py 수정
위의 수정 사항을 `src/models/__init__.py`에 적용

---

## 📊 테스트

### 통합 후 테스트 명령어
```bash
# Encoder-Decoder (기존)
python train.py --mode single --models kobart --debug

# Causal LM (신규)
python train.py --mode single --models llama-3.2-korean-3b --debug

# 둘 다 앙상블
python train.py --mode multi_model --models kobart llama-3.2-korean-3b --debug
```

---

**다음 작업**: `04_나머지_모듈_구현_가이드.md` (Solar API, K-Fold, Ensemble, Optuna 등)
