"""
가중치 기반 앙상블

PRD 12: 다중 모델 앙상블 전략 구현
- 가중 평균 앙상블
- 모델별 가중치 설정
"""

import numpy as np
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer


class WeightedEnsemble:
    """가중치 기반 앙상블"""

    def __init__(
        self,
        models: List,
        tokenizers: List[PreTrainedTokenizer],
        weights: Optional[List[float]] = None,
        logger=None
    ):
        """
        Args:
            models: 모델 리스트
            tokenizers: 토크나이저 리스트
            weights: 모델별 가중치 (None이면 균등 가중치)
            logger: Logger 인스턴스
        """
        assert len(models) == len(tokenizers), "모델과 토크나이저 개수가 다름"

        self.models = models
        self.tokenizers = tokenizers
        self.logger = logger

        # 가중치 설정
        if weights is None:
            # 균등 가중치
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "가중치 개수가 모델 개수와 다름"
            # 정규화
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self._log(f"WeightedEnsemble 초기화: {len(models)}개 모델")
        self._log(f"  - 가중치: {[f'{w:.3f}' for w in self.weights]}")

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def predict(
        self,
        dialogues: List[str],
        max_new_tokens: int = 200,
        min_new_tokens: int = 30,
        num_beams: int = 4,
        batch_size: int = 8
    ) -> List[str]:
        """
        가중 앙상블 예측

        Args:
            dialogues: 입력 대화 리스트
            max_new_tokens: 생성할 최대 토큰 수
            min_new_tokens: 생성할 최소 토큰 수
            num_beams: Beam search 빔 개수
            batch_size: 배치 크기

        Returns:
            예측 요약 리스트
        """
        self._log(f"\n가중 앙상블 예측 시작")
        self._log(f"  - 입력 데이터: {len(dialogues)}개")
        self._log(f"  - 모델 수: {len(self.models)}개")

        # 각 모델의 예측 수집
        all_predictions = []

        for idx, (model, tokenizer, weight) in enumerate(zip(self.models, self.tokenizers, self.weights)):
            # FIXME: Corrupted log message

            predictions = []

            # 배치 단위로 예측
            for i in range(0, len(dialogues), batch_size):
                batch_dialogues = dialogues[i:i + batch_size]

                # 토큰화
                inputs = tokenizer(
                    batch_dialogues,
                    max_length=1024,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )

                # GPU 사용 가능 시 이동
                if next(model.parameters()).is_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # 예측
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

                # 디코딩
                batch_predictions = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )

                predictions.extend(batch_predictions)

            all_predictions.append((predictions, weight))

        # 가중 평균 (텍스트 기반 투표)
        # 각 샘플별로 가장 높은 가중치를 가진 예측 선택
        final_predictions = []

        for i in range(len(dialogues)):
            # 각 모델의 i번째 예측과 가중치
            candidates = [(pred[i], weight) for pred, weight in all_predictions]

            # 가장 높은 가중치의 예측 선택
            best_pred = max(candidates, key=lambda x: x[1])[0]
            final_predictions.append(best_pred)

        self._log(f"\n가중 앙상블 예측 완료: {len(final_predictions)}개")

        return final_predictions


def create_weighted_ensemble(
    models: List,
    tokenizers: List,
    weights: Optional[List[float]] = None,
    logger=None
) -> WeightedEnsemble:
    """
    편의 함수: 가중치 앙상블 생성

    Args:
        models: 모델 리스트
        tokenizers: 토크나이저 리스트
        weights: 가중치 리스트
        logger: Logger 인스턴스

    Returns:
        WeightedEnsemble 인스턴스
    """
    return WeightedEnsemble(models, tokenizers, weights, logger)
