"""
앙상블 전략

PRD 12: 다중 모델 앙상블 전략
여러 모델의 요약 중 최적 요약 선택
"""

# ------------------------- 표준 라이브러리 ------------------------- #
from abc import ABC, abstractmethod
from typing import List, Dict


# ==================== 앙상블 전략 베이스 클래스 ==================== #
class EnsembleStrategy(ABC):
    """
    앙상블 전략 베이스 클래스

    모든 앙상블 전략이 구현해야 하는 인터페이스 정의
    """

    # ---------------------- 추상 메서드: 요약 선택 ---------------------- #
    @abstractmethod
    def select(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        quality_scores: Dict[str, List[float]],
        threshold: float
    ) -> List[str]:
        """
        최종 요약 선택

        Args:
            candidate_summaries: KoBART가 생성한 요약 리스트
            reference_summaries: 참조 모델별 요약 딕셔너리
            quality_scores: 품질 점수 딕셔너리
            threshold: 임계값 (전략별 의미 다름)

        Returns:
            최종 선택된 요약 리스트
        """
        pass


# ==================== 품질 기반 선택 전략 (추천) ==================== #
class QualityBasedStrategy(EnsembleStrategy):
    """
    품질 기반 선택 전략 (추천)

    로직:
    1. KoBART 품질이 임계값 이상이면 KoBART 사용
    2. 아니면 가장 품질 높은 참조 모델 사용
    """

    # ---------------------- 요약 선택 메서드 ---------------------- #
    def select(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        quality_scores: Dict[str, List[float]],
        threshold: float
    ) -> List[str]:
        """
        품질이 가장 높은 요약 선택 (개선)
        """
        final_summaries = []

        for i in range(len(candidate_summaries)):
            candidate_quality = quality_scores["candidate_quality"][i]

            # -------------- KoBART 품질 확인 -------------- #
            if candidate_quality >= threshold:
                # 품질이 충분히 높으면 KoBART 사용
                final_summaries.append(candidate_summaries[i])
                continue

            # -------------- 최고 품질 참조 모델 선택 -------------- #
            best_model = None
            best_quality = -1

            for model_name in reference_summaries.keys():
                quality_key = f"{model_name}_quality"
                if quality_key in quality_scores:
                    quality = quality_scores[quality_key][i]

                    # ✅ 추가: 참조 요약이 비어있거나 너무 짧으면 제외
                    ref_summary = reference_summaries[model_name][i]
                    if not ref_summary.strip() or len(ref_summary) < 10:
                        continue

                    if quality > best_quality:
                        best_quality = quality
                        best_model = model_name

            # -------------- 최종 선택 (개선) -------------- #
            if best_model and best_quality > threshold:
                # ✅ 수정: 품질이 임계값보다 높을 때만 참조 모델 사용
                final_summaries.append(reference_summaries[best_model][i])
            else:
                # ✅ 개선: 참조 모델 품질이 낮으면 KoBART 사용
                final_summaries.append(candidate_summaries[i])

        return final_summaries


# ==================== 임계값 기반 전략 ==================== #
class ThresholdStrategy(EnsembleStrategy):
    """
    임계값 기반 전략

    로직:
    - 모델 간 합의도(agreement)가 임계값 이하면 참조 모델 사용
    - 아니면 KoBART 사용
    """

    # ---------------------- 요약 선택 메서드 ---------------------- #
    def select(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        quality_scores: Dict[str, List[float]],
        threshold: float
    ) -> List[str]:
        """
        임계값 기반 선택
        """
        final_summaries = []

        # -------------- 참조 모델 확인 -------------- #
        if not reference_summaries:
            return candidate_summaries                  # 참조 없으면 KoBART 반환

        first_ref_model = list(reference_summaries.keys())[0]

        # -------------- 샘플별 선택 -------------- #
        for i in range(len(candidate_summaries)):
            agreement = quality_scores.get("candidate_agreement", [0.5] * len(candidate_summaries))[i]

            if agreement <= threshold:
                # 합의도 낮음 → 참조 모델 사용
                final_summaries.append(reference_summaries[first_ref_model][i])
            else:
                # 합의도 높음 → KoBART 사용
                final_summaries.append(candidate_summaries[i])

        return final_summaries


# ==================== 투표 기반 전략 ==================== #
class VotingStrategy(EnsembleStrategy):
    """
    투표 기반 전략

    로직:
    - 모든 요약(KoBART + 참조 모델들) 중 품질이 가장 높은 요약 선택
    """

    # ---------------------- 요약 선택 메서드 ---------------------- #
    def select(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        quality_scores: Dict[str, List[float]],
        threshold: float
    ) -> List[str]:
        """
        투표로 선택 (품질 점수 기반)
        """
        final_summaries = []

        for i in range(len(candidate_summaries)):
            # -------------- 모든 후보 수집 -------------- #
            all_candidates = {
                "candidate": (candidate_summaries[i], quality_scores["candidate_quality"][i])
            }

            for model_name, summaries in reference_summaries.items():
                quality_key = f"{model_name}_quality"
                if quality_key in quality_scores:
                    all_candidates[model_name] = (summaries[i], quality_scores[quality_key][i])

            # -------------- 최고 품질 선택 -------------- #
            best_name = max(all_candidates.keys(), key=lambda k: all_candidates[k][1])
            final_summaries.append(all_candidates[best_name][0])

        return final_summaries


# ==================== 가중 평균 전략 ==================== #
class WeightedStrategy(EnsembleStrategy):
    """
    가중 평균 전략

    주의: 문장 단위 가중 평균은 어려우므로 Quality-based 전략으로 폴백
    """

    # ---------------------- 요약 선택 메서드 ---------------------- #
    def select(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        quality_scores: Dict[str, List[float]],
        threshold: float
    ) -> List[str]:
        """
        가중 평균 (실제로는 Quality-based와 동일)

        로직:
        - 품질 점수를 가중치로 사용하여 가장 높은 것 선택
        - 문장 앙상블이 불가능하므로 Quality-based 전략 사용
        """
        # 문장 앙상블은 어려우므로 Quality-based 전략 사용
        quality_based = QualityBasedStrategy()
        return quality_based.select(
            candidate_summaries,
            reference_summaries,
            quality_scores,
            threshold
        )


# ==================== 전략 팩토리 함수 ==================== #
# ---------------------- 전략 생성 함수 ---------------------- #
def get_ensemble_strategy(strategy_name: str) -> EnsembleStrategy:
    """
    앙상블 전략 팩토리

    Args:
        strategy_name: 전략 이름
            - "quality_based": 품질 기반 선택 (추천)
            - "threshold": 임계값 기반 선택
            - "voting": 투표 기반 선택
            - "weighted": 가중 평균 (Quality-based로 폴백)

    Returns:
        EnsembleStrategy 인스턴스

    Raises:
        ValueError: 지원하지 않는 전략 이름
    """
    # -------------- 전략 매핑 -------------- #
    strategies = {
        "quality_based": QualityBasedStrategy,
        "threshold": ThresholdStrategy,
        "voting": VotingStrategy,
        "weighted": WeightedStrategy,
    }

    # -------------- 전략 검증 및 생성 -------------- #
    if strategy_name not in strategies:
        raise ValueError(f"지원하지 않는 전략: {strategy_name}. 사용 가능: {list(strategies.keys())}")

    return strategies[strategy_name]()
