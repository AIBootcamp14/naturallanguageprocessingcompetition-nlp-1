#!/usr/bin/env python3
"""
모델 로딩 및 관리 모듈
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

import torch
from typing import Dict
from model_utils import load_model_for_train, load_model_for_inference
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast


class ModelManager:
    """
    모델 로딩 및 관리 클래스
    """

    def __init__(self, config: Dict, tokenizer: PreTrainedTokenizerFast, device: torch.device):
        """
        ModelManager 초기화

        Args:
            config: 설정 딕셔너리
            tokenizer: 토크나이저
            device: 사용할 디바이스
        """
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    def load_model_for_training(self) -> BartForConditionalGeneration:
        """
        학습용 모델을 로드합니다 (scripts/model_utils.py 활용)

        Returns:
            학습용 BART 모델
        """
        print("\n" + "=" * 80)
        print("📦 모델 로딩 중...")
        print("=" * 80)

        model = load_model_for_train(self.config, self.tokenizer, self.device)

        print(f"✅ 모델 로드 완료")
        print(f"   모델: {self.config['general']['model_name']}")
        print(f"   Vocab size: {len(self.tokenizer)}")
        print(f"   Device: {self.device}")
        print("=" * 80)

        return model

    def load_model_for_inference(self, checkpoint_path: str) -> BartForConditionalGeneration:
        """
        추론용 모델을 체크포인트에서 로드합니다 (scripts/model_utils.py 활용)

        Args:
            checkpoint_path: 체크포인트 디렉토리 경로

        Returns:
            추론용 BART 모델
        """
        print("\n" + "=" * 80)
        print("📦 체크포인트 로딩 중...")
        print("=" * 80)

        model = load_model_for_inference(checkpoint_path, self.tokenizer, self.device)

        print(f"✅ 체크포인트 로드 완료")
        print(f"   경로: {checkpoint_path}")
        print(f"   Vocab size: {len(self.tokenizer)}")
        print(f"   Device: {self.device}")
        print("=" * 80)

        return model
