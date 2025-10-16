"""Generation configuration utilities for summarization models."""
from __future__ import annotations

from typing import Any, Dict

from transformers import PreTrainedTokenizerBase


def build_generation_kwargs(
    tokenizer: PreTrainedTokenizerBase,
    **overrides: Any,
) -> Dict[str, Any]:
    """Return decoding arguments tuned to reduce repetition in summaries.

    Args:
        tokenizer: Tokenizer supplying EOS/BOS/PAD token IDs.
        **overrides: Optional overrides for specific generation parameters.

    Returns:
        Dictionary ready to pass to ``model.generate``.
    """
    gen_kwargs: Dict[str, Any] = {
        #4가지 문장 경로를 동시에 탐색해요
        "num_beams": 4, #4
        #1.0이면 중립 (즉, 길이에 따른 불이익 없음), 1보다 작으면 → 짧은 문장 선호, 1보다 크면 → 긴 문장 선호
        "length_penalty": 1.0,

        "no_repeat_ngram_size": 3, #생성 중에 같은 3단어(trigram) 가 반복되지 않도록 막아요.
        "repetition_penalty": 1.2, #값이 클수록 → 반복 억제 강함
        "min_new_tokens": 12,#최소 12개 단어(토큰) 를 생성하기 전에는 절대 문장을 끝내지 못하게 해요.
        "max_new_tokens": 80, #모델이 한 번에 만들 수 있는 최대 단어(토큰)로 80개가 넘어가면 강제로 멈춰서 끝없는 문장 루프 방지.
        "early_stopping": True,
        "decoder_start_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    gen_kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return gen_kwargs


__all__ = ["build_generation_kwargs"]
