# ==================== Blending 앙상블 ==================== #
"""
Blending 앙상블 전략

PRD 12: 다중 모델 앙상블 전략
Validation 세트 기반 가중치 최적화 시스템
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from typing import List, Dict, Any, Optional
from pathlib import Path

# ---------------------- 서드파티 라이브러리 ---------------------- #
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge import Rouge


# ==================== BlendingEnsemble ==================== #
class BlendingEnsemble:
    """
    Blending 앙상블 - Validation 기반 가중치 학습

    Stacking과 유사하지만, 별도의 Validation 세트로 가중치를 학습
    """

    def __init__(
        self,
        base_models: List[AutoModelForSeq2SeqLM],
        tokenizers: List[AutoTokenizer],
        model_names: List[str],
        logger=None
    ):
        """
        Args:
            base_models: 베이스 모델 리스트
            tokenizers: 토크나이저 리스트
            model_names: 모델 이름 리스트
            logger: Logger 인스턴스
        """
        self.base_models = base_models
        self.tokenizers = tokenizers
        self.model_names = model_names
        self.logger = logger

        # 모델별 가중치 (초기값: 균등)
        self.weights = np.ones(len(base_models)) / len(base_models)
        self.is_trained = False

        self._log(f"BlendingEnsemble 초기화 완료")
        self._log(f"  베이스 모델 수: {len(base_models)}")
        self._log(f"  초기 가중치: 균등 ({1/len(base_models):.3f})")

    def _log(self, msg: str):
        """로깅 유틸리티"""
        if self.logger:
            if hasattr(self.logger, 'write'):
                self.logger.write(msg)
            elif hasattr(self.logger, 'info'):
                self.logger.info(msg)
        else:
            print(msg)

    def train_weights(
        self,
        val_dialogues: List[str],
        val_summaries: List[str],
        max_length: int = 200,
        num_beams: int = 4
    ):
        """
        Validation 세트로 가중치 학습

        Args:
            val_dialogues: 검증용 대화 리스트
            val_summaries: 검증용 요약 리스트 (정답)
            max_length: 최대 생성 길이
            num_beams: Beam search 크기
        """
        self._log("\nBlending 가중치 학습 시작...")
        self._log(f"  검증 샘플 수: {len(val_dialogues)}")

        # 1단계: 각 베이스 모델의 예측 수집
        base_predictions = self._get_base_predictions(
            val_dialogues,
            max_length=max_length,
            num_beams=num_beams
        )

        # 2단계: 각 모델의 ROUGE 점수 계산
        rouge = Rouge()
        model_scores = []

        for idx, pred_list in enumerate(base_predictions):
            scores_sum = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
            valid_count = 0

            for pred, ref in zip(pred_list, val_summaries):
                try:
                    scores = rouge.get_scores(pred, ref)[0]
                    scores_sum['rouge-1'] += scores['rouge-1']['f']
                    scores_sum['rouge-2'] += scores['rouge-2']['f']
                    scores_sum['rouge-l'] += scores['rouge-l']['f']
                    valid_count += 1
                except:
                    pass

            # 평균 ROUGE-L 점수를 가중치로 사용
            avg_score = scores_sum['rouge-l'] / max(valid_count, 1)
            model_scores.append(avg_score)

            self._log(f"  모델 {idx+1} ({self.model_names[idx]}): ROUGE-L = {avg_score:.4f}")

        # 3단계: 점수를 가중치로 변환 (정규화)
        model_scores = np.array(model_scores)
        if model_scores.sum() > 0:
            self.weights = model_scores / model_scores.sum()
        else:
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)

        self.is_trained = True

        self._log("\n  ✅ 가중치 학습 완료:")
        for name, weight in zip(self.model_names, self.weights):
            self._log(f"    {name}: {weight:.4f}")

    def _get_base_predictions(
        self,
        dialogues: List[str],
        max_length: int = 200,
        num_beams: int = 4
    ) -> List[List[str]]:
        """베이스 모델들의 예측 수집"""
        all_predictions = []

        for idx, (model, tokenizer, name) in enumerate(zip(
            self.base_models, self.tokenizers, self.model_names
        )):
            self._log(f"  모델 {idx+1}/{len(self.base_models)} 예측 중: {name}")

            predictions = []
            model.eval()

            with torch.no_grad():
                for dialogue in dialogues:
                    inputs = tokenizer(
                        dialogue,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    )

                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        model = model.cuda()

                    outputs = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )

                    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predictions.append(summary)

            all_predictions.append(predictions)

        return all_predictions

    def predict(
        self,
        dialogues: List[str],
        max_length: int = 200,
        num_beams: int = 4,
        batch_size: int = 8
    ) -> List[str]:
        """
        Blending 앙상블 예측

        Args:
            dialogues: 대화 리스트
            max_length: 최대 생성 길이
            num_beams: Beam search 크기
            batch_size: 배치 크기

        Returns:
            예측된 요약 리스트
        """
        if not self.is_trained:
            self._log("⚠️  가중치가 학습되지 않았습니다. 균등 가중치 사용")

        # 1단계: 베이스 모델 예측
        base_predictions = self._get_base_predictions(
            dialogues,
            max_length=max_length,
            num_beams=num_beams
        )

        # 2단계: 가중 투표로 최종 예측 선택
        final_predictions = self._weighted_selection(base_predictions)

        return final_predictions

    def _weighted_selection(
        self,
        base_predictions: List[List[str]]
    ) -> List[str]:
        """가중치 기반 예측 선택"""
        final_predictions = []
        rouge = Rouge()

        for i in range(len(base_predictions[0])):
            # 각 모델의 예측 수집
            candidates = [pred_list[i] for pred_list in base_predictions]

            # 가중치 기반 점수 계산
            weighted_scores = []
            for j, candidate in enumerate(candidates):
                # 간단한 방법: 가중치를 직접 점수로 사용
                score = self.weights[j]
                weighted_scores.append(score)

            # 가장 높은 가중치를 가진 모델의 예측 선택
            best_idx = np.argmax(weighted_scores)
            final_predictions.append(candidates[best_idx])

        return final_predictions

    def save(self, output_dir: str):
        """앙상블 저장"""
        import joblib

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 가중치 저장
        if self.is_trained:
            weights_path = output_path / "blending_weights.pkl"
            joblib.dump({
                'weights': self.weights,
                'model_names': self.model_names
            }, weights_path)
            self._log(f"  Blending 가중치 저장: {weights_path}")

    def load(self, output_dir: str):
        """앙상블 로드"""
        import joblib

        weights_path = Path(output_dir) / "blending_weights.pkl"
        if weights_path.exists():
            data = joblib.load(weights_path)
            self.weights = data['weights']
            self.is_trained = True
            self._log(f"  Blending 가중치 로드: {weights_path}")


# ==================== 팩토리 함수 ==================== #
def create_blending_ensemble(
    base_models: List[AutoModelForSeq2SeqLM],
    tokenizers: List[AutoTokenizer],
    model_names: List[str],
    logger=None
) -> BlendingEnsemble:
    """
    BlendingEnsemble 생성 팩토리 함수

    Args:
        base_models: 베이스 모델 리스트
        tokenizers: 토크나이저 리스트
        model_names: 모델 이름 리스트
        logger: Logger 인스턴스

    Returns:
        BlendingEnsemble 인스턴스
    """
    return BlendingEnsemble(
        base_models=base_models,
        tokenizers=tokenizers,
        model_names=model_names,
        logger=logger
    )
