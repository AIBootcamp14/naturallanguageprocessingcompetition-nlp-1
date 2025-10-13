"""
투표 기반 앙상블

PRD 12: 다중 모델 앙상블 전략 구현
- Soft Voting (확률 기반)
- Hard Voting (다수결)
"""

import numpy as np
from typing import List, Literal
from transformers import PreTrainedTokenizer
from collections import Counter


class VotingEnsemble:
    """투표 기반 앙상블"""

    def __init__(
        self,
        models: List,
        tokenizers: List[PreTrainedTokenizer],
        voting: Literal["hard", "soft"] = "hard",
        logger=None
    ):
        """
        Args:
            models: 모델 리스트
            tokenizers: 토크나이저 리스트
            voting: 투표 방식 ("hard" 또는 "soft")
            logger: Logger 인스턴스
        """
        assert len(models) == len(tokenizers), "모델과 토크나이저 개수가 다름"

        self.models = models
        self.tokenizers = tokenizers
        self.voting = voting
        self.logger = logger

        self._log(f"VotingEnsemble 초기화: {len(models)}개 모델")
        self._log(f"  - 투표 방식: {voting}")

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def predict(
        self,
        dialogues: List[str],
        max_length: int = 200,
        num_beams: int = 4,
        batch_size: int = 8
    ) -> List[str]:
        """
        투표 앙상블 예측

        Args:
            dialogues: 입력 대화 리스트
            max_length: 최대 생성 길이
            num_beams: Beam search 빔 개수
            batch_size: 배치 크기

        Returns:
            예측 요약 리스트
        """
        # FIXME: Corrupted log message
        self._log(f"  - 입력 데이터: {len(dialogues)}개")
        self._log(f"  - 모델 수: {len(self.models)}개")

        # 각 모델의 예측 수집
        all_predictions = []

        for idx, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
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
                    max_length=max_length,
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

            all_predictions.append(predictions)

        # 투표
        if self.voting == "hard":
            final_predictions = self._hard_voting(all_predictions)
        else:
            # Soft voting은 텍스트 생성 태스크에서 구현이 복잡함
            # 여기서는 hard voting으로 대체
            self._log("  - 경고: Soft voting은 텍스트 생성에서 지원하지 않음. Hard voting 사용.")
            final_predictions = self._hard_voting(all_predictions)

        self._log(f"\n투표 앙상블 예측 완료: {len(final_predictions)}개")

        return final_predictions

    def _hard_voting(self, all_predictions: List[List[str]]) -> List[str]:
        """
        Hard Voting (다수결)

        Args:
            all_predictions: 모델별 예측 리스트

        Returns:
            최종 예측 리스트
        """
        final_predictions = []

        num_samples = len(all_predictions[0])

        for i in range(num_samples):
            # 각 모델의 i번째 예측
            candidates = [preds[i] for preds in all_predictions]

            # 가장 많이 나온 예측 선택
            counter = Counter(candidates)
            most_common = counter.most_common(1)[0][0]

            final_predictions.append(most_common)

        return final_predictions


def create_voting_ensemble(
    models: List,
    tokenizers: List,
    voting: Literal["hard", "soft"] = "hard",
    logger=None
) -> VotingEnsemble:
    """
    편의 함수: 투표 앙상블 생성

    Args:
        models: 모델 리스트
        tokenizers: 토크나이저 리스트
        voting: 투표 방식
        logger: Logger 인스턴스

    Returns:
        VotingEnsemble 인스턴스
    """
    return VotingEnsemble(models, tokenizers, voting, logger)
