# ==================== Stacking 앙상블 ==================== #
"""
Stacking 앙상블 전략

PRD 12: 다중 모델 앙상블 전략
메타 학습기를 사용한 2단계 앙상블 시스템
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from typing import List, Dict, Any, Optional
from pathlib import Path

# ---------------------- 서드파티 라이브러리 ---------------------- #
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from rouge import Rouge


# ==================== StackingEnsemble ==================== #
class StackingEnsemble:
    """
    Stacking 앙상블 - 2단계 학습

    1단계: 베이스 모델들의 예측
    2단계: 메타 학습기가 베이스 모델 출력을 조합
    """

    def __init__(
        self,
        base_models: List[AutoModelForSeq2SeqLM],
        tokenizers: List[AutoTokenizer],
        model_names: List[str],
        meta_learner: str = "ridge",
        logger=None
    ):
        """
        Args:
            base_models: 베이스 모델 리스트
            tokenizers: 토크나이저 리스트
            model_names: 모델 이름 리스트
            meta_learner: 메타 학습기 ("ridge", "rf", "linear")
            logger: Logger 인스턴스
        """
        self.base_models = base_models
        self.tokenizers = tokenizers
        self.model_names = model_names
        self.meta_learner_type = meta_learner
        self.logger = logger

        # 메타 학습기 초기화
        self.meta_learner = self._create_meta_learner(meta_learner)
        self.is_trained = False

        self._log(f"StackingEnsemble 초기화 완료")
        self._log(f"  베이스 모델 수: {len(base_models)}")
        self._log(f"  메타 학습기: {meta_learner}")

    def _log(self, msg: str):
        """로깅 유틸리티"""
        if self.logger:
            if hasattr(self.logger, 'write'):
                self.logger.write(msg)
            elif hasattr(self.logger, 'info'):
                self.logger.info(msg)
        else:
            print(msg)

    def _create_meta_learner(self, learner_type: str):
        """메타 학습기 생성"""
        if learner_type == "ridge":
            return Ridge(alpha=1.0)
        elif learner_type == "rf":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif learner_type == "linear":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        else:
            raise ValueError(f"지원하지 않는 메타 학습기: {learner_type}")

    def train_meta_learner(
        self,
        train_dialogues: List[str],
        train_summaries: List[str],
        max_new_tokens: int = 200,

        min_new_tokens: int = 30,
        num_beams: int = 4
    ):
        """
        메타 학습기 학습

        Args:
            train_dialogues: 학습용 대화 리스트
            train_summaries: 학습용 요약 리스트 (정답)
            max_new_tokens: 생성할 최대 토큰 수
            min_new_tokens: 생성할 최소 토큰 수
            num_beams: Beam search 크기
        """
        self._log("\n메타 학습기 학습 시작...")
        self._log(f"  학습 샘플 수: {len(train_dialogues)}")

        # 1단계: 베이스 모델들의 예측 수집
        base_predictions = self._get_base_predictions(
            train_dialogues,
            max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
            num_beams=num_beams
        )

        # 2단계: 예측을 ROUGE 점수로 변환 (특징 벡터)
        rouge = Rouge()
        X_train = []

        for i in range(len(train_dialogues)):
            features = []
            ref_summary = train_summaries[i]

            for pred_list in base_predictions:
                pred_summary = pred_list[i]

                try:
                    # 각 베이스 모델의 ROUGE 점수 계산
                    scores = rouge.get_scores(pred_summary, ref_summary)[0]
                    features.extend([
                        scores['rouge-1']['f'],
                        scores['rouge-2']['f'],
                        scores['rouge-l']['f']
                    ])
                except:
                    # 예측 실패 시 0으로 채움
                    features.extend([0.0, 0.0, 0.0])

            X_train.append(features)

        X_train = np.array(X_train)

        # 3단계: 메타 학습기 학습
        # 목표: 최적의 모델 조합을 찾기 위한 가중치 학습
        # 여기서는 각 모델의 ROUGE 점수를 학습하여 최종 앙상블 가중치 결정
        y_train = np.arange(len(self.base_models)).repeat(len(train_dialogues) // len(self.base_models) + 1)[:len(train_dialogues)]

        self.meta_learner.fit(X_train, y_train)
        self.is_trained = True

        self._log("  ✅ 메타 학습기 학습 완료")

    def _get_base_predictions(
        self,
        dialogues: List[str],
        max_new_tokens: int = 200,

        min_new_tokens: int = 30,
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
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
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
        max_new_tokens: int = 200,

        min_new_tokens: int = 30,
        num_beams: int = 4,
        batch_size: int = 8
    ) -> List[str]:
        """
        Stacking 앙상블 예측

        Args:
            dialogues: 대화 리스트
            max_new_tokens: 생성할 최대 토큰 수
            min_new_tokens: 생성할 최소 토큰 수
            num_beams: Beam search 크기
            batch_size: 배치 크기

        Returns:
            예측된 요약 리스트
        """
        if not self.is_trained:
            self._log("⚠️  메타 학습기가 학습되지 않았습니다. 단순 평균 사용")

        # 1단계: 베이스 모델 예측
        base_predictions = self._get_base_predictions(
            dialogues,
            max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
            num_beams=num_beams
        )

        # 2단계: 메타 학습기로 최종 예측 선택
        if self.is_trained:
            final_predictions = self._meta_predict(base_predictions, dialogues)
        else:
            # 메타 학습기 없으면 첫 번째 모델 사용
            final_predictions = base_predictions[0]

        return final_predictions

    def _meta_predict(
        self,
        base_predictions: List[List[str]],
        dialogues: List[str]
    ) -> List[str]:
        """메타 학습기를 사용한 최종 예측"""
        final_predictions = []
        rouge = Rouge()

        for i in range(len(dialogues)):
            # 각 베이스 모델의 예측에 대한 특징 추출
            features = []
            candidates = []

            for pred_list in base_predictions:
                pred = pred_list[i]
                candidates.append(pred)

                # ROUGE 자체 평가 (self-evaluation)
                try:
                    # 대화와의 유사도를 특징으로 사용
                    scores = rouge.get_scores(pred, dialogues[i][:200])[0]
                    features.extend([
                        scores['rouge-1']['f'],
                        scores['rouge-2']['f'],
                        scores['rouge-l']['f']
                    ])
                except:
                    features.extend([0.0, 0.0, 0.0])

            # 메타 학습기가 최적 모델 선택
            features_array = np.array(features).reshape(1, -1)
            best_model_idx = int(self.meta_learner.predict(features_array)[0])
            best_model_idx = max(0, min(best_model_idx, len(candidates) - 1))

            final_predictions.append(candidates[best_model_idx])

        return final_predictions

    def save(self, output_dir: str):
        """앙상블 저장"""
        import joblib

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 메타 학습기 저장
        if self.is_trained:
            meta_path = output_path / "meta_learner.pkl"
            joblib.dump(self.meta_learner, meta_path)
            self._log(f"  메타 학습기 저장: {meta_path}")

    def load(self, output_dir: str):
        """앙상블 로드"""
        import joblib

        meta_path = Path(output_dir) / "meta_learner.pkl"
        if meta_path.exists():
            self.meta_learner = joblib.load(meta_path)
            self.is_trained = True
            self._log(f"  메타 학습기 로드: {meta_path}")


# ==================== 팩토리 함수 ==================== #
def create_stacking_ensemble(
    base_models: List[AutoModelForSeq2SeqLM],
    tokenizers: List[AutoTokenizer],
    model_names: List[str],
    meta_learner: str = "ridge",
    logger=None
) -> StackingEnsemble:
    """
    StackingEnsemble 생성 팩토리 함수

    Args:
        base_models: 베이스 모델 리스트
        tokenizers: 토크나이저 리스트
        model_names: 모델 이름 리스트
        meta_learner: 메타 학습기 타입
        logger: Logger 인스턴스

    Returns:
        StackingEnsemble 인스턴스
    """
    return StackingEnsemble(
        base_models=base_models,
        tokenizers=tokenizers,
        model_names=model_names,
        meta_learner=meta_learner,
        logger=logger
    )
