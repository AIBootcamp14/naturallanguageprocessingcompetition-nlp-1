"""
모델 매니저

PRD 12: 다중 모델 앙상블 전략 구현
- 여러 모델 로드 및 관리
- 앙상블 실행
"""

import os
from typing import List, Dict, Optional, Literal
from pathlib import Path


class ModelManager:
    """여러 모델을 관리하는 매니저"""

    def __init__(self, logger=None):
        """
        Args:
            logger: Logger 인스턴스
        """
        self.models = []
        self.tokenizers = []
        self.model_names = []
        self.logger = logger

        self._log("ModelManager 초기화")

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def load_model(
        self,
        model_path: str,
        model_name: Optional[str] = None
    ):
        """
        모델 로드

        Args:
            model_path: 모델 경로
            model_name: 모델 이름 (표시용)
        """
        from transformers import (
            AutoConfig,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
            AutoTokenizer
        )

        if model_name is None:
            model_name = Path(model_path).name

        self._log(f"\n모델 로드 중: {model_name}")
        self._log(f"  - 경로: {model_path}")

        # 모델 타입 자동 감지
        config = AutoConfig.from_pretrained(model_path)
        is_encoder_decoder = config.is_encoder_decoder if hasattr(config, 'is_encoder_decoder') else False

        # 모델 및 토크나이저 로드
        if is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Decoder-only 모델의 경우 left padding 설정
        if not is_encoder_decoder:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # GPU 사용 가능 시 이동
        import torch
        if torch.cuda.is_available():
            model = model.cuda()
            self._log(f"  - GPU로 이동 완료")

        # 평가 모드
        model.eval()

        # 저장
        self.models.append(model)
        self.tokenizers.append(tokenizer)
        self.model_names.append(model_name)

        self._log(f"모델 로드 완료: {model_name}")

    def load_models(
        self,
        model_paths: List[str],
        model_names: Optional[List[str]] = None
    ):
        """
        여러 모델 로드

        Args:
            model_paths: 모델 경로 리스트
            model_names: 모델 이름 리스트
        """
        if model_names is None:
            model_names = [None] * len(model_paths)

        for path, name in zip(model_paths, model_names):
            self.load_model(path, name)

    def create_ensemble(
        self,
        ensemble_type: Literal["weighted", "voting"] = "weighted",
        weights: Optional[List[float]] = None,
        voting: Literal["hard", "soft"] = "hard"
    ):
        """
        앙상블 생성

        Args:
            ensemble_type: 앙상블 타입 ("weighted" 또는 "voting")
            weights: 가중치 리스트 (weighted일 때만)
            voting: 투표 방식 (voting일 때만)

        Returns:
            앙상블 인스턴스
        """
        assert len(self.models) > 0, "로드된 모델이 없음"

        self._log(f"\n앙상블 생성: {ensemble_type}")

        if ensemble_type == "weighted":
            from .weighted import WeightedEnsemble
            ensemble = WeightedEnsemble(
                self.models,
                self.tokenizers,
                weights,
                self.logger
            )
        elif ensemble_type == "voting":
            from .voting import VotingEnsemble
            ensemble = VotingEnsemble(
                self.models,
                self.tokenizers,
                voting,
                self.logger
            )
        else:
            raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")

        return ensemble

    def get_info(self) -> Dict:
        """
        로드된 모델 정보 반환

        Returns:
            모델 정보 딕셔너리
        """
        return {
            "num_models": len(self.models),
            "model_names": self.model_names,
        }


def create_model_manager(logger=None) -> ModelManager:
    """
    편의 함수: 모델 매니저 생성

    Args:
        logger: Logger 인스턴스

    Returns:
        ModelManager 인스턴스
    """
    return ModelManager(logger)
