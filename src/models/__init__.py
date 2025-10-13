# ==================== 모델 모듈 ==================== #
"""
모델 모듈

사전학습 모델 로딩 및 관리를 담당하는 모듈
- Encoder-Decoder 모델 로더 (KoBART 등)
- Causal LM 모델 로더 (Llama, Qwen 등)
- 통합 인터페이스: 모델 타입 자동 감지 및 로드

PRD 08: LLM 파인튜닝 전략 통합
"""

# ---------------------- 모델 모듈 Import ---------------------- #
from .model_loader import ModelLoader, load_model_and_tokenizer as load_encoder_decoder
from .llm_loader import load_causal_lm


# ==================== 통합 모델 로더 ==================== #
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
    # 모델 타입 감지 (기본값: encoder_decoder)
    model_type = config.model.get('type', 'encoder_decoder')

    if logger:
        logger.write(f"모델 타입: {model_type}")

    # 타입별 로더 선택
    if model_type == 'encoder_decoder':
        return load_encoder_decoder(config, logger)
    elif model_type == 'causal_lm':
        return load_causal_lm(config, logger)
    else:
        raise ValueError(
            f"지원하지 않는 모델 타입: {model_type}\n"
            f"사용 가능한 타입: encoder_decoder, causal_lm"
        )


# ---------------------- 외부 노출 모듈 정의 ---------------------- #
__all__ = [
    'ModelLoader',                  # 모델 로더 클래스
    'load_model_and_tokenizer',     # 통합 모델 로드 함수
    'load_encoder_decoder',         # Encoder-Decoder 로더
    'load_causal_lm'                # Causal LM 로더
]
