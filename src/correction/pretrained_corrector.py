"""
사전학습 모델 보정기

PRD 04, 12: 추론 최적화 및 앙상블 전략 구현
허깅페이스 사전학습 모델을 활용한 요약 보정
"""

# ------------------------- 표준 라이브러리 ------------------------- #
from typing import List, Dict, Optional

# ------------------------- 서드파티 라이브러리 ------------------------- #
import torch


# ==================== 사전학습 모델 보정기 클래스 ==================== #
class PretrainedCorrector:
    """
    허깅페이스 사전학습 모델을 활용한 요약 보정

    주요 기능:
    1. 여러 사전학습 모델 로드 및 관리
    2. 참조 요약 생성
    3. 품질 평가 및 보정
    4. 앙상블 전략 적용

    사용 예시:
        corrector = PretrainedCorrector(
            model_names=["gogamza/kobart-base-v2", "digit82/kobart-summarization"],
            correction_strategy="quality_based",
            quality_threshold=0.3
        )
        corrected = corrector.correct_batch(dialogues, candidate_summaries)
    """

    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(
        self,
        model_names: List[str],
        correction_strategy: str = "quality_based",
        quality_threshold: float = 0.3,
        device: Optional[torch.device] = None,
        logger=None
    ):
        """
        Args:
            model_names: 허깅페이스 모델 이름 리스트
                예: ["gogamza/kobart-base-v2", "digit82/kobart-summarization"]
            correction_strategy: 보정 전략
                - "threshold": 임계값 기반
                - "voting": 투표 기반
                - "weighted": 가중 평균
                - "quality_based": 품질 기반 (추천)
            quality_threshold: 품질 임계값 (0.0~1.0)
            device: 추론 디바이스 (None이면 자동 감지)
            logger: Logger 인스턴스
        """
        self.model_names = model_names
        self.correction_strategy = correction_strategy
        self.quality_threshold = quality_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        # -------------- 모델 로더 초기화 -------------- #
        from src.correction.model_loader import HuggingFaceModelLoader
        self.model_loader = HuggingFaceModelLoader(device=self.device, logger=logger)

        # -------------- 모델 및 토크나이저 저장소 -------------- #
        self.models = {}                                # 모델 딕셔너리
        self.tokenizers = {}                            # 토크나이저 딕셔너리
        self._load_all_models()                         # 모든 모델 로드

        # -------------- 품질 평가기 초기화 -------------- #
        from src.correction.quality_evaluator import QualityEvaluator
        self.evaluator = QualityEvaluator(logger=logger)

        # -------------- 앙상블 전략 초기화 -------------- #
        from src.correction.ensemble_strategies import get_ensemble_strategy
        self.ensemble = get_ensemble_strategy(correction_strategy)

    # ---------------------- 모든 모델 로드 메서드 ---------------------- #
    def _load_all_models(self):
        """
        모든 허깅페이스 모델 로드

        실패한 모델은 건너뜀 (graceful degradation)
        """
        for model_name in self.model_names:
            try:
                model, tokenizer = self.model_loader.load_model(model_name)
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
            except Exception as e:
                self._log(f"⚠️  모델 로드 실패, 건너뜀: {model_name}")
                self._log(f"   에러: {str(e)}")

    # ---------------------- 배치 보정 메서드 ---------------------- #
    def correct_batch(
        self,
        dialogues: List[str],
        candidate_summaries: List[str],
        batch_size: int = 16,
        **generation_kwargs
    ) -> List[str]:
        """
        배치 보정

        Args:
            dialogues: 입력 대화 리스트
            candidate_summaries: KoBART가 생성한 초안 요약 리스트
            batch_size: 배치 크기
            **generation_kwargs: 생성 파라미터 (max_new_tokens, num_beams 등)

        Returns:
            보정된 요약 리스트
        """
        # -------------- 로드된 모델 확인 -------------- #
        if not self.models:
            self._log("⚠️  로드된 참조 모델이 없음. 원본 요약 반환")
            return candidate_summaries

        # -------------- 보정 시작 로그 -------------- #
        self._log("=" * 60)
        self._log("사전학습 모델 보정 시작")
        self._log(f"  - 샘플 수: {len(dialogues)}")
        self._log(f"  - 참조 모델 수: {len(self.models)}")
        self._log(f"  - 보정 전략: {self.correction_strategy}")
        self._log(f"  - 품질 임계값: {self.quality_threshold}")
        self._log("=" * 60)

        # -------------- 단계 1: 참조 요약 생성 -------------- #
        # 각 허깅페이스 모델로 참조 요약 생성
        reference_summaries = {}
        for model_name, model in self.models.items():
            self._log(f"\n[1/3] 참조 요약 생성 중: {model_name}")
            tokenizer = self.tokenizers[model_name]
            summaries = self._generate_summaries(
                dialogues, model, tokenizer, batch_size, **generation_kwargs
            )
            reference_summaries[model_name] = summaries
            self._log(f"  ✅ 완료: {len(summaries)}개 요약 생성")

        # -------------- 단계 2: 품질 평가 -------------- #
        self._log(f"\n[2/3] 품질 평가 중...")
        quality_scores = self.evaluator.evaluate_all(
            candidate_summaries=candidate_summaries,
            reference_summaries=reference_summaries,
            dialogues=dialogues
        )
        self._log(f"  ✅ 평가 완료")

        # -------------- 단계 3: 보정 전략 적용 -------------- #
        self._log(f"\n[3/3] 보정 전략 적용 중: {self.correction_strategy}")
        corrected_summaries = self.ensemble.select(
            candidate_summaries=candidate_summaries,
            reference_summaries=reference_summaries,
            quality_scores=quality_scores,
            threshold=self.quality_threshold
        )
        self._log(f"  ✅ 보정 완료")

        # -------------- 보정 통계 출력 -------------- #
        num_corrected = sum([
            1 for orig, corr in zip(candidate_summaries, corrected_summaries)
            if orig != corr
        ])
        self._log(f"\n📊 보정 통계:")
        self._log(f"  - 전체: {len(dialogues)}개")
        self._log(f"  - 보정됨: {num_corrected}개 ({num_corrected/len(dialogues)*100:.1f}%)")
        self._log(f"  - 유지됨: {len(dialogues)-num_corrected}개")
        self._log("=" * 60)

        return corrected_summaries

    # ---------------------- 단일 모델 요약 생성 메서드 ---------------------- #
    def _generate_summaries(
        self,
        dialogues: List[str],
        model,
        tokenizer,
        batch_size: int = 16,
        **generation_kwargs
    ) -> List[str]:
        """
        단일 모델로 배치 요약 생성

        Args:
            dialogues: 대화 리스트
            model: HuggingFace 모델
            tokenizer: HuggingFace 토크나이저
            batch_size: 배치 크기
            **generation_kwargs: 생성 파라미터

        Returns:
            요약 리스트
        """
        from src.inference import create_predictor

        # -------------- Predictor 생성 -------------- #
        # 기존 코드 재사용
        predictor = create_predictor(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            logger=None                                 # 너무 많은 로그 방지
        )

        # -------------- 배치 예측 -------------- #
        summaries = predictor.predict_batch(
            dialogues=dialogues,
            batch_size=batch_size,
            show_progress=False,                        # 진행바 비활성화
            **generation_kwargs
        )

        return summaries

    # ---------------------- 로깅 헬퍼 메서드 ---------------------- #
    def _log(self, msg: str):
        """
        로깅 헬퍼 함수

        Args:
            msg: 로그 메시지
        """
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)
