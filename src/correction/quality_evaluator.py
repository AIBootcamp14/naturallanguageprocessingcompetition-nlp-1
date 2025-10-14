"""
품질 평가기

PRD 04: 추론 최적화
ROUGE 메트릭 기반 요약 품질 평가
"""

# ------------------------- 표준 라이브러리 ------------------------- #
from typing import List, Dict, Optional

# ------------------------- 서드파티 라이브러리 ------------------------- #
import numpy as np


# ==================== 요약 품질 평가 클래스 ==================== #
class QualityEvaluator:
    """
    요약 품질 평가기

    지원 메트릭:
    - ROUGE-L F1 (단일 요약 품질)
    - Self-ROUGE (여러 요약 간 일치도)
    - Length Ratio (길이 적절성, 선택적)

    주요 기능:
    - KoBART와 참조 모델 간 품질 비교
    - 여러 모델 요약의 일치도 측정
    - 최적 요약 선택을 위한 품질 점수 제공
    """

    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(self, logger=None):
        """
        Args:
            logger: Logger 인스턴스
        """
        self.logger = logger

        # -------------- ROUGE 라이브러리 로드 -------------- #
        try:
            from rouge import Rouge
            self.rouge = Rouge()                        # ROUGE 평가기 초기화
        except ImportError:
            self._log("⚠️  rouge 라이브러리가 설치되지 않음 (pip install rouge)")
            self.rouge = None

    # ---------------------- 전체 품질 평가 메서드 ---------------------- #
    def evaluate_all(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        dialogues: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        전체 품질 평가

        Args:
            candidate_summaries: KoBART가 생성한 요약 리스트
            reference_summaries: 각 모델별 참조 요약
                예: {"model1": ["요약1", ...], "model2": [...]}
            dialogues: 원본 대화 리스트 (선택적, 현재 미사용)

        Returns:
            품질 점수 딕셔너리
                {
                    "candidate_quality": [0.5, 0.7, ...],           # KoBART 품질
                    "model1_quality": [0.6, 0.8, ...],              # 참조 모델1 품질
                    "model2_quality": [...],                        # 참조 모델2 품질
                    "candidate_agreement": [0.4, 0.6, ...],         # Self-ROUGE (모델 간 일치도)
                }
        """
        # -------------- ROUGE 사용 가능 여부 확인 -------------- #
        if self.rouge is None:
            self._log("❌ ROUGE 평가기를 사용할 수 없음")
            return self._create_dummy_scores(len(candidate_summaries), reference_summaries)

        self._log("품질 평가 시작...")

        quality_scores = {}

        # -------------- 단계 1: Self-ROUGE 계산 -------------- #
        # 여러 요약 간 일치도 계산 (모든 모델 포함)
        self._log("  [1/3] Self-ROUGE 계산 중...")
        all_summaries = [candidate_summaries]           # KoBART 요약 추가
        for model_name, summaries in reference_summaries.items():
            all_summaries.append(summaries)             # 각 참조 모델 요약 추가

        agreement_scores = []
        for i in range(len(candidate_summaries)):
            # i번째 샘플의 모든 요약들 수집
            sample_summaries = [summaries[i] for summaries in all_summaries]
            agreement = self._compute_self_rouge(sample_summaries)
            agreement_scores.append(agreement)

        quality_scores["candidate_agreement"] = agreement_scores

        # -------------- 단계 2: 각 참조 모델의 품질 계산 -------------- #
        # 참조 모델이 다른 모델들과 얼마나 일치하는지 평가
        self._log("  [2/3] 모델별 품질 계산 중...")
        for model_name, summaries in reference_summaries.items():
            model_quality = []
            for i in range(len(summaries)):
                # 이 모델을 제외한 다른 모델들과 비교
                other_summaries = [
                    s[i] for name, s in reference_summaries.items()
                    if name != model_name
                ] + [candidate_summaries[i]]            # KoBART도 포함

                if other_summaries:
                    # 다른 모든 요약과의 평균 ROUGE
                    avg_rouge = np.mean([
                        self._compute_rouge_f1(summaries[i], other)
                        for other in other_summaries
                    ])
                else:
                    avg_rouge = 0.5                     # 기본값

                model_quality.append(avg_rouge)

            quality_scores[f"{model_name}_quality"] = model_quality

        # -------------- 단계 3: KoBART (Candidate) 품질 계산 -------------- #
        # KoBART가 참조 모델들과 얼마나 일치하는지 평가
        self._log("  [3/3] Candidate 품질 계산 중...")
        candidate_quality = []
        for i in range(len(candidate_summaries)):
            # 모든 참조 모델과 비교
            ref_list = [
                ref[i] for ref in reference_summaries.values()
            ]
            if ref_list:
                # 모든 참조 요약과의 평균 ROUGE
                avg_rouge = np.mean([
                    self._compute_rouge_f1(candidate_summaries[i], ref)
                    for ref in ref_list
                ])
            else:
                avg_rouge = 0.5                         # 기본값

            candidate_quality.append(avg_rouge)

        quality_scores["candidate_quality"] = candidate_quality

        # -------------- 단계 4: dialogue 유사도 페널티 적용 -------------- #
        # dialogue와 너무 유사한 요약에 페널티 부과
        if dialogues:
            self._log("  [4/4] dialogue 유사도 페널티 적용 중...")
            for model_name, summaries in reference_summaries.items():
                quality_key = f"{model_name}_quality"
                if quality_key in quality_scores:
                    for i in range(len(summaries)):
                        # 요약과 dialogue의 ROUGE 계산
                        dialogue_similarity = self._compute_rouge_f1(summaries[i], dialogues[i])

                        # dialogue와 너무 유사하면 품질 점수 대폭 감소
                        if dialogue_similarity > 0.8:
                            quality_scores[quality_key][i] *= 0.1  # 90% 페널티

        self._log("  ✅ 평가 완료")
        return quality_scores

    # ---------------------- ROUGE-L F1 계산 메서드 ---------------------- #
    def _compute_rouge_f1(self, hypothesis: str, reference: str) -> float:
        """
        ROUGE-L F1 점수 계산

        Args:
            hypothesis: 평가 대상 요약
            reference: 참조 요약

        Returns:
            ROUGE-L F1 점수 (0.0~1.0)
        """
        if self.rouge is None:
            return 0.5                                  # ROUGE 없으면 기본값

        try:
            # -------------- 빈 문자열 체크 -------------- #
            if not hypothesis.strip() or not reference.strip():
                return 0.0

            # -------------- ROUGE 계산 -------------- #
            scores = self.rouge.get_scores(hypothesis, reference)[0]
            return scores['rouge-l']['f']               # ROUGE-L F1 반환

        except Exception as e:
            # ROUGE 계산 실패 시 기본값 반환
            return 0.0

    # ---------------------- Self-ROUGE 계산 메서드 ---------------------- #
    def _compute_self_rouge(self, summaries: List[str]) -> float:
        """
        Self-ROUGE 계산: 여러 요약 간 평균 ROUGE

        Args:
            summaries: 요약 리스트 (2개 이상)

        Returns:
            Self-ROUGE 점수 (0.0~1.0)
        """
        # -------------- 요약 개수 확인 -------------- #
        if len(summaries) < 2:
            return 1.0                                  # 요약이 1개면 완벽한 일치

        # -------------- 모든 쌍에 대해 ROUGE 계산 -------------- #
        scores = []
        for i in range(len(summaries)):
            for j in range(i+1, len(summaries)):
                score = self._compute_rouge_f1(summaries[i], summaries[j])
                scores.append(score)

        return float(np.mean(scores)) if scores else 0.0

    # ---------------------- 더미 점수 생성 메서드 ---------------------- #
    def _create_dummy_scores(
        self,
        num_samples: int,
        reference_summaries: Dict[str, List[str]]
    ) -> Dict[str, List[float]]:
        """
        ROUGE 사용 불가능 시 더미 점수 생성

        Args:
            num_samples: 샘플 개수
            reference_summaries: 참조 요약 딕셔너리

        Returns:
            더미 품질 점수 딕셔너리 (모든 값 0.5)
        """
        quality_scores = {
            "candidate_quality": [0.5] * num_samples,   # KoBART 기본값
            "candidate_agreement": [0.5] * num_samples, # Self-ROUGE 기본값
        }

        # 각 참조 모델에 대한 기본값 생성
        for model_name in reference_summaries.keys():
            quality_scores[f"{model_name}_quality"] = [0.5] * num_samples

        return quality_scores

    # ---------------------- 로깅 헬퍼 메서드 ---------------------- #
    def _log(self, msg: str):
        """
        로깅 헬퍼 함수

        Args:
            msg: 로그 메시지
        """
        if self.logger:
            self.logger.write(msg)
