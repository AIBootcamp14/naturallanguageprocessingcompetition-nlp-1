#!/usr/bin/env python3
"""
Tokenizer 관련 유틸리티
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import List


def load_tokenizer(model_name: str, special_tokens: List[str] = None) -> PreTrainedTokenizerFast:
    """
    Tokenizer를 로드하고 special tokens를 추가합니다.

    Args:
        model_name: 모델 이름 (예: 'digit82/kobart-summarization')
        special_tokens: 추가할 특수 토큰 리스트

    Returns:
        설정된 tokenizer
    """
    # baseline.ipynb Cell 31 참조
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if special_tokens:
        special_tokens_dict = {
            'additional_special_tokens': special_tokens
        }
        tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


def tokenize_encoder_inputs(texts: List[str], tokenizer: PreTrainedTokenizerFast,
                            max_length: int = 512, return_tensors: str = "pt"):
    """
    Encoder 입력을 tokenize합니다.

    Args:
        texts: 입력 텍스트 리스트
        tokenizer: 사용할 tokenizer
        max_length: 최대 길이
        return_tensors: 반환 형식 ('pt' for PyTorch)

    Returns:
        Tokenized BatchEncoding
    """
    # baseline.ipynb Cell 27, 39 참조
    return tokenizer(
        texts,
        return_tensors=return_tensors,
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False
    )


def tokenize_decoder_inputs(texts: List[str], tokenizer: PreTrainedTokenizerFast,
                            max_length: int = 100, return_tensors: str = "pt"):
    """
    Decoder 입력을 tokenize합니다.

    Args:
        texts: 입력 텍스트 리스트 (BOS 토큰 포함)
        tokenizer: 사용할 tokenizer
        max_length: 최대 길이
        return_tensors: 반환 형식

    Returns:
        Tokenized BatchEncoding
    """
    # baseline.ipynb Cell 27 참조
    return tokenizer(
        texts,
        return_tensors=return_tensors,
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False
    )