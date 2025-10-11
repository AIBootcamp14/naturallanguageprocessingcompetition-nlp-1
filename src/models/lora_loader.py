"""
LLM 파인튜닝을 위한 LoRA/QLoRA 모델 로더

PRD 08: LLM 파인튜닝 전략 구현
- QLoRA 4-bit 양자화 지원
- LoRA (Low-Rank Adaptation) 지원
- Causal LM (Llama, Qwen) 지원
- Chat template 토큰 자동 추가
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from omegaconf import DictConfig
from typing import Tuple, Optional, Dict, Any


class LoRALoader:
    """LoRA/QLoRA를 사용한 LLM 로더"""

    def __init__(self, config: DictConfig, logger=None):
        """
        Args:
            config: 모델 설정 (causal_lm 타입)
            logger: Logger 인스턴스
        """
        self.config = config
        self.logger = logger
        self._log("LoRALoader 초기화")

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def load_model_and_tokenizer(
        self,
        use_lora: bool = True,
        use_qlora: bool = False
    ) -> Tuple[Any, AutoTokenizer]:
        """
        Causal LM 모델 및 토크나이저 로드

        Args:
            use_lora: LoRA 사용 여부
            use_qlora: QLoRA (4-bit 양자화) 사용 여부

        Returns:
            (model, tokenizer)
        """
        checkpoint = self.config.model.checkpoint
        self._log(f"모델 로딩: {checkpoint}")

        # 1. 토크나이저 로드
        tokenizer = self._load_tokenizer(checkpoint)

        # 2. 양자화 설정 (QLoRA)
        bnb_config = None
        if use_qlora:
            bnb_config = self._create_bnb_config()

        # 3. 모델 로드
        model = self._load_causal_lm(checkpoint, bnb_config)

        # 4. Chat template 토큰 추가
        self._add_chat_tokens(model, tokenizer)

        # 5. LoRA 적용
        if use_lora or use_qlora:
            model = self._apply_lora(model)

        # 6. Left padding 설정 (Causal LM 추론 시 필수)
        self._configure_tokenizer(tokenizer)

        self._log(f"모델 로딩 완료: {checkpoint}")
        self._log(f"  - LoRA: {use_lora or use_qlora}")
        self._log(f"  - QLoRA (4-bit): {use_qlora}")
        self._log(f"  - Trainable params: {self._count_trainable_params(model)}")

        return model, tokenizer

    def _load_tokenizer(self, checkpoint: str) -> AutoTokenizer:
        """토크나이저 로드"""
        self._log(f"토크나이저 로딩: {checkpoint}")

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=True
        )

        # Padding 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _create_bnb_config(self) -> BitsAndBytesConfig:
        """QLoRA용 BitsAndBytes 설정 생성"""
        self._log("QLoRA 4-bit 양자화 설정 생성")

        # dtype 결정 (Llama: bf16, Qwen: fp16)
        compute_dtype = torch.bfloat16
        if 'qwen' in self.config.model.checkpoint.lower():
            compute_dtype = torch.float16
            self._log("  - Qwen 모델: fp16 사용")
        else:
            self._log("  - Llama 모델: bf16 사용")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )

        return bnb_config

    def _load_causal_lm(
        self,
        checkpoint: str,
        bnb_config: Optional[BitsAndBytesConfig]
    ) -> AutoModelForCausalLM:
        """Causal LM 모델 로드"""
        self._log(f"Causal LM 모델 로딩: {checkpoint}")

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bnb_config is None else None
        )

        return model

    def _add_chat_tokens(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """Chat template 토큰 추가"""
        checkpoint = self.config.model.checkpoint.lower()

        # Llama 모델
        if 'llama' in checkpoint:
            chat_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
            self._log(f"Llama chat 토큰 추가: {chat_tokens}")
        # Qwen 모델
        elif 'qwen' in checkpoint:
            chat_tokens = ["<|im_start|>", "<|im_end|>"]
            self._log(f"Qwen chat 토큰 추가: {chat_tokens}")
        else:
            self._log("알 수 없는 모델, chat 토큰 추가 생략")
            return

        # 토큰 추가
        num_added = tokenizer.add_special_tokens({'additional_special_tokens': chat_tokens})

        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            self._log(f"  - {num_added}개 토큰 추가됨, 임베딩 크기 조정 완료")

    def _apply_lora(self, model: AutoModelForCausalLM) -> PeftModel:
        """LoRA 적용"""
        self._log("LoRA 적용 중...")

        # LoRA Config 생성
        lora_config = LoraConfig(
            r=self.config.lora.get('r', 16),
            lora_alpha=self.config.lora.get('alpha', 32),
            target_modules=self.config.lora.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=self.config.lora.get('dropout', 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # LoRA 적용
        model = get_peft_model(model, lora_config)

        self._log("LoRA 적용 완료")
        self._log(f"  - rank: {lora_config.r}")
        self._log(f"  - alpha: {lora_config.lora_alpha}")
        self._log(f"  - target_modules: {lora_config.target_modules}")

        return model

    def _configure_tokenizer(self, tokenizer: AutoTokenizer):
        """토크나이저 설정 (Left padding for Causal LM)"""
        # PRD 08: Prompt truncation 방지
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        self._log("토크나이저 설정 완료 (Left padding/truncation)")

    def _count_trainable_params(self, model) -> str:
        """학습 가능한 파라미터 수 계산"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        return f"{trainable_params:,} / {all_params:,} ({trainable_percent:.2f}%)"


def load_lora_model_and_tokenizer(
    config: DictConfig,
    use_lora: bool = True,
    use_qlora: bool = False,
    logger=None
) -> Tuple[Any, AutoTokenizer]:
    """
    편의 함수: LoRA 모델 및 토크나이저 로드

    Args:
        config: 모델 설정
        use_lora: LoRA 사용 여부
        use_qlora: QLoRA 사용 여부
        logger: Logger 인스턴스

    Returns:
        (model, tokenizer)
    """
    loader = LoRALoader(config, logger=logger)
    return loader.load_model_and_tokenizer(use_lora=use_lora, use_qlora=use_qlora)
