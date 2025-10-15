#!/usr/bin/env python3
"""
모델 로딩 및 설정 유틸리티
"""

import torch
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    PreTrainedTokenizerFast
)
from typing import Tuple


def load_model_for_train(config: dict, tokenizer: PreTrainedTokenizerFast,
                         device: torch.device) -> BartForConditionalGeneration:
    """
    학습용 모델을 로드합니다.

    Args:
        config: 설정 딕셔너리
        tokenizer: 설정된 tokenizer (special tokens 추가 완료)
        device: 사용할 디바이스

    Returns:
        설정된 모델
    """
    # baseline.ipynb Cell 31 참조
    model_name = config['general']['model_name']

    # Config 로드
    bart_config = BartConfig.from_pretrained(model_name)

    # 모델 로드
    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        config=bart_config
    )

    # Vocab size 조정 (special tokens 추가 반영)
    model.resize_token_embeddings(len(tokenizer))

    # Device로 이동
    model.to(device)

    return model


def load_model_for_inference(checkpoint_path: str, tokenizer: PreTrainedTokenizerFast,
                             device: torch.device) -> BartForConditionalGeneration:
    """
    추론용 모델을 체크포인트에서 로드합니다.

    Args:
        checkpoint_path: 체크포인트 디렉토리 경로
        tokenizer: 설정된 tokenizer
        device: 사용할 디바이스

    Returns:
        로드된 모델
    """
    # baseline.ipynb Cell 41 참조
    model = BartForConditionalGeneration.from_pretrained(
        checkpoint_path,
        local_files_only=True
    )

    # Vocab size 조정
    model.resize_token_embeddings(len(tokenizer))

    # Device로 이동
    model.to(device)

    return model


def get_model_info(model: BartForConditionalGeneration) -> dict:
    """
    모델 정보를 반환합니다.

    Args:
        model: BART 모델

    Returns:
        모델 정보 딕셔너리
    """
    return {
        'vocab_size': model.config.vocab_size,
        'max_position_embeddings': model.config.max_position_embeddings,
        'd_model': model.config.d_model,
        'encoder_layers': model.config.encoder_layers,
        'decoder_layers': model.config.decoder_layers,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }