# ==================== Causal LM 모델 로더 ==================== #
"""
Causal LM 모델 로더 (Llama, Qwen, 등)

PRD 08: LLM 파인튜닝 전략 구현
- QLoRA (4-bit quantization + LoRA)
- Chat Template 지원
- Gradient Checkpointing
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import torch

# ---------------------- 서드파티 라이브러리 ---------------------- #
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ==================== Causal LM 로더 ==================== #
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
    if hasattr(config.model, 'quantization') and config.model.quantization:
        if logger:
            logger.write("  Quantization 설정 적용...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.model.quantization.get('load_in_4bit', True),
            bnb_4bit_compute_dtype=getattr(
                torch,
                config.model.quantization.get('bnb_4bit_compute_dtype', 'float16')
            ),
            bnb_4bit_quant_type=config.model.quantization.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=config.model.quantization.get('bnb_4bit_use_double_quant', True)
        )

    # 2. 모델 로드
    if logger:
        logger.write("  모델 로딩 중...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model.checkpoint,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.training.get('bf16', True) else torch.float16,
        trust_remote_code=True
    )

    # 3. 토크나이저 로드
    if logger:
        logger.write("  토크나이저 로딩 중...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.checkpoint,
        trust_remote_code=True
    )

    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        if logger:
            logger.write(f"  패딩 토큰 설정: {tokenizer.eos_token}")

    # Chat 템플릿 토큰 추가
    if hasattr(config.tokenizer, 'chat_template_tokens') and config.tokenizer.chat_template_tokens:
        special_tokens = {
            'additional_special_tokens': config.tokenizer.chat_template_tokens
        }
        num_added = tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            if logger:
                logger.write(f"  Chat 템플릿 토큰 추가: {num_added}개")

    # 4. LoRA 설정
    if hasattr(config.model, 'lora') and config.model.lora:
        if logger:
            logger.write("  LoRA 설정 적용 중...")

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
            logger.write(f"  ✅ LoRA 적용 완료")
            logger.write(f"    학습 가능 파라미터: {trainable_params:,} ({trainable_percentage:.2f}%)")
            logger.write(f"    전체 파라미터: {total_params:,}")

    # 5. Gradient Checkpointing
    if config.training.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Gradient checkpointing과 함께 사용 시 필수
        if logger:
            logger.write("  ✅ Gradient Checkpointing 활성화")

    if logger:
        logger.write("  ✅ Causal LM 로드 완료")

    return model, tokenizer


# ==================== LLM 프롬프트 포맷팅 ==================== #
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
            "content": "You are an expert in dialogue summarization. Summarize the given dialogue concisely and accurately in Korean."
        },
        {
            "role": "user",
            "content": f"다음 대화를 요약해주세요:\n\n{dialogue}\n\n요약:"
        }
    ]

    # Chat 템플릿 적용
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback: 수동 포맷팅 (Llama 3 스타일)
            prompt = _manual_format_llama3(messages)
    else:
        # Fallback: 수동 포맷팅
        prompt = _manual_format_llama3(messages)

    return prompt


def _manual_format_llama3(messages):
    """
    Llama 3 스타일 수동 포맷팅

    Args:
        messages: 메시지 리스트

    Returns:
        포맷팅된 프롬프트
    """
    prompt = "<|begin_of_text|>"

    for message in messages:
        role = message['role']
        content = message['content']

        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    # Assistant 턴 시작
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt
