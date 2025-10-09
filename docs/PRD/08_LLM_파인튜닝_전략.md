# 🤖 LLM 파인튜닝 전략

## 📚 접근 방식 비교

### 1. 인코더-디코더 모델 (기존 방식)
- **모델**: BART, T5, mBART
- **특징**: Seq2Seq 구조로 요약에 특화
- **장점**: 요약 태스크에 최적화됨
- **단점**: 모델 크기 제한, 컨텍스트 이해 한계

### 2. 디코더 전용 LLM 파인튜닝 (새로운 방식) ✨
- **모델**: GPT, LLaMA, Polyglot-Ko, SOLAR
- **특징**: Causal Language Modeling을 요약에 활용
- **장점**:
  - 더 깊은 문맥 이해
  - 대규모 사전학습 지식 활용
  - Instruction Following 능력
- **단점**:
  - 더 많은 컴퓨팅 리소스 필요
  - 토큰 사용량 증가

## 🎯 LLM 파인튜닝 구현 전략

### 1. 모델 선택
```python
# 권장 모델 (한국어 특화)
models = {
    'polyglot-ko-5.8b': {
        'size': '5.8B',
        'context': 2048,
        'korean_optimized': True
    },
    'kogpt': {
        'size': '6B',
        'context': 2048,
        'korean_optimized': True
    },
    'llama-2-ko': {
        'size': '7B',
        'context': 4096,
        'korean_optimized': True
    }
}
```

### 2. 데이터 포맷팅
```python
def format_for_llm_finetuning(dialogue, summary):
    """
    LLM 파인튜닝을 위한 데이터 포맷
    """
    instruction = "다음 대화를 간결하게 요약해주세요."

    # Option 1: Instruction Format
    prompt = f"""### Instruction:
{instruction}

### Input:
{dialogue}

### Response:
{summary}"""

    # Option 2: Chat Format
    prompt = f"""<|system|>
당신은 대화를 요약하는 전문가입니다.
<|user|>
다음 대화를 요약해주세요:
{dialogue}
<|assistant|>
{summary}"""

    return prompt
```

### 3. 파인튜닝 설정
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 설정 (효율적인 파인튜닝)
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./llm_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

### 4. Instruction Tuning
```python
# 다양한 Instruction 템플릿
instructions = [
    "다음 대화를 요약해주세요.",
    "아래 대화의 핵심 내용을 정리해주세요.",
    "주어진 대화를 간단히 요약하세요.",
    "다음 대화에서 중요한 정보를 추출해 요약하세요.",
    "대화 내용을 한 문단으로 요약해주세요."
]

def augment_with_instructions(data):
    """
    다양한 instruction으로 데이터 증강
    """
    augmented = []
    for dialogue, summary in data:
        for instruction in instructions:
            formatted = format_with_instruction(
                instruction, dialogue, summary
            )
            augmented.append(formatted)
    return augmented
```

## 💡 효율적인 파인튜닝 기법

### 1. LoRA (Low-Rank Adaptation)
- 전체 모델 대신 일부 파라미터만 학습
- 메모리 사용량 90% 감소
- 학습 속도 3-5배 향상

### 2. QLoRA (Quantized LoRA)
- 4-bit 양자화 + LoRA
- 더 큰 모델을 제한된 GPU에서 학습 가능

### 3. Gradient Checkpointing
- 메모리 사용량 감소
- 더 큰 배치 크기 사용 가능

## 📊 예상 성능

| 방식 | 모델 | 파라미터 | GPU 메모리 | 예상 ROUGE |
|------|------|----------|------------|------------|
| 기존 | KoBART | 124M | 8GB | 47-50 |
| LLM | Polyglot-Ko | 5.8B | 16GB | 55-60 |
| LLM + LoRA | Polyglot-Ko | 5.8B | 8GB | 53-58 |
| LLM + QLoRA | LLaMA-2-Ko | 7B | 8GB | 58-63 |

## 🔧 구현 예제

### 전체 파인튜닝 파이프라인
```python
class LLMFineTuner:
    def __init__(self, model_name, use_lora=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True if use_lora else False,
            device_map="auto"
        )

        if use_lora:
            self.model = get_peft_model(self.model, lora_config)

    def prepare_dataset(self, data):
        """데이터셋 준비"""
        formatted_data = []
        for item in data:
            text = self.format_for_training(
                item['dialogue'],
                item['summary']
            )
            formatted_data.append(text)
        return formatted_data

    def train(self, train_dataset, eval_dataset):
        """모델 학습"""
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()

    def generate_summary(self, dialogue):
        """요약 생성"""
        prompt = f"""### Instruction:
다음 대화를 요약해주세요.

### Input:
{dialogue}

### Response:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

        summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return summary.split("### Response:")[-1].strip()
```

## 📈 학습 전략

### Phase 1: 기본 파인튜닝
1. 전체 데이터로 3 epochs 학습
2. Learning rate: 2e-5
3. Batch size: 4 (gradient accumulation 사용)

### Phase 2: Instruction Tuning
1. 다양한 instruction 템플릿 적용
2. 데이터 증강 (5배)
3. 추가 2 epochs 학습

### Phase 3: RLHF (선택사항)
1. 생성된 요약에 대한 품질 평가
2. Reward model 학습
3. PPO를 통한 추가 최적화

## ⚡ 최적화 팁

### 1. Mixed Precision Training
```python
training_args = TrainingArguments(
    fp16=True,  # 또는 bf16=True
    # ...
)
```

### 2. Gradient Accumulation
```python
# 실제 배치 크기 = 4 * 4 = 16
per_device_train_batch_size=4,
gradient_accumulation_steps=4,
```

### 3. Learning Rate Schedule
```python
lr_scheduler_type="cosine",
warmup_ratio=0.1,
```

## 🎯 예상 결과

### 성능 향상 예측
- **베이스라인 (BART)**: 47.12
- **LLM 파인튜닝 (기본)**: 55-58
- **LLM + Instruction Tuning**: 58-62
- **LLM + 앙상블**: 62-65
- **최종 (후처리 포함)**: 65-70

### 리소스 사용량
- **학습 시간**: 5-10시간 (모델 크기에 따라)
- **GPU 메모리**: 8-16GB
- **디스크 공간**: 20-50GB (모델 체크포인트)

## 🚀 실행 계획

### Week 1
- [ ] LLM 모델 선택 및 환경 구성
- [ ] 데이터 포맷팅 스크립트 작성
- [ ] LoRA 설정 최적화

### Week 2
- [ ] 파인튜닝 실행
- [ ] Instruction tuning
- [ ] 성능 평가 및 비교

### Week 3
- [ ] 하이퍼파라미터 최적화
- [ ] 앙상블 준비
- [ ] 최종 모델 선정