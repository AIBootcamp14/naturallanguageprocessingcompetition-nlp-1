"""
사전학습 모델 보정 모듈

PRD 04, 12: 추론 최적화 및 앙상블 전략 구현
"""

# ------------------------- 프로젝트 모듈 ------------------------- #
from src.correction.pretrained_corrector import PretrainedCorrector
from src.correction.model_loader import HuggingFaceModelLoader
from src.correction.quality_evaluator import QualityEvaluator
from src.correction.ensemble_strategies import (
    EnsembleStrategy,
    QualityBasedStrategy,
    ThresholdStrategy,
    VotingStrategy,
    get_ensemble_strategy
)

__all__ = [
    "PretrainedCorrector",
    "HuggingFaceModelLoader",
    "QualityEvaluator",
    "EnsembleStrategy",
    "QualityBasedStrategy",
    "ThresholdStrategy",
    "VotingStrategy",
    "get_ensemble_strategy",
    "create_pretrained_corrector",
]


# ==================== 팩토리 함수 ==================== #
# ---------------------- PretrainedCorrector 생성 함수 ---------------------- #
def create_pretrained_corrector(
    model_names,
    correction_strategy="quality_based",
    quality_threshold=0.3,
    device=None,
    logger=None,
    checkpoint_dir=None
):
    """
    편의 함수: PretrainedCorrector 인스턴스 생성

    Args:
        model_names: 허깅페이스 모델 이름 리스트
        correction_strategy: 보정 전략 (quality_based, threshold, voting, weighted)
        quality_threshold: 품질 임계값 (0.0~1.0)
        device: 추론 디바이스
        logger: Logger 인스턴스
        checkpoint_dir: 체크포인트 디렉토리 (선택)

    Returns:
        PretrainedCorrector 인스턴스
    """
    return PretrainedCorrector(
        model_names=model_names,
        correction_strategy=correction_strategy,
        quality_threshold=quality_threshold,
        device=device,
        logger=logger,
        checkpoint_dir=checkpoint_dir
    )
