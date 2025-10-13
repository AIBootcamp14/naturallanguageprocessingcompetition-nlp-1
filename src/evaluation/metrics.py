# ==================== 평가 지표 모듈 ==================== #
"""
평가 지표 계산 모듈

ROUGE 점수 계산을 위한 모듈
- ROUGE-1/2/L F1 점수 계산 (경진대회 평가 기준)
- Multi-reference 지원
- 배치 계산 및 통계 정보
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from typing import List, Dict, Union, Optional
import warnings

# ---------------------- 서드파티 라이브러리 ---------------------- #
import numpy as np
from rouge_score import rouge_scorer


# ==================== RougeCalculator 클래스 정의 ==================== #
class RougeCalculator:
    """ROUGE 점수 계산 클래스"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(
        self,
        rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL'],  # ROUGE 타입
        use_stemmer: bool = False                       # 형태소 분석기 사용 여부
    ):
        """
        Args:
            rouge_types: 계산할 ROUGE 타입 리스트
            use_stemmer: 형태소 분석기 사용 여부 (한국어는 False 권장)
        """
        self.rouge_types = rouge_types                  # ROUGE 타입 저장
        self.use_stemmer = use_stemmer                  # 형태소 분석기 설정 저장

        # -------------- ROUGE Scorer 초기화 -------------- #
        try:
            self.scorer = rouge_scorer.RougeScorer(     # ROUGE Scorer 생성
                rouge_types=rouge_types,
                use_stemmer=use_stemmer
            )
        except Exception as e:
            warnings.warn(f"RougeScorer 초기화 실패: {e}")
            self.scorer = None


    # ---------------------- 단일 샘플 ROUGE 계산 함수 ---------------------- #
    def calculate_single(
        self,
        prediction: str,                                # 예측 요약
        reference: Union[str, List[str]]                # 정답 요약 (단일 또는 다중)
    ) -> Dict[str, Dict[str, float]]:
        """
        단일 샘플의 ROUGE 점수 계산

        Args:
            prediction: 예측 요약
            reference: 정답 요약 (str 또는 List[str])

        Returns:
            Dict[str, Dict[str, float]]: {
                'rouge1': {'precision': 0.5, 'recall': 0.6, 'fmeasure': 0.55},
                'rouge2': {...},
                'rougeL': {...}
            }
        """
        # -------------- 입력 검증 -------------- #
        if not prediction or not reference:             # 빈 입력 확인
            return self._empty_scores()                 # 빈 점수 반환

        # -------------- Multi-reference 처리 -------------- #
        if isinstance(reference, list):                 # 다중 정답인 경우
            # 각 정답에 대해 ROUGE 계산 후 최대값 선택
            all_scores = []                             # 모든 점수 저장

            for ref in reference:                       # 각 정답 반복
                if ref:                                 # 빈 문자열이 아닌 경우
                    scores = self.scorer.score(prediction, ref)  # ROUGE 계산
                    all_scores.append(scores)           # 점수 추가

            # 모든 정답이 비어있는 경우
            if not all_scores:
                return self._empty_scores()

            # 각 ROUGE 타입별로 최대 F1 점수 선택
            result = {}
            for rouge_type in self.rouge_types:         # 각 ROUGE 타입
                max_score = max(                        # 최대 점수 선택
                    all_scores,
                    key=lambda x: x[rouge_type].fmeasure  # F1 점수 기준
                )[rouge_type]

                result[rouge_type] = {                  # 결과 저장
                    'precision': max_score.precision,
                    'recall': max_score.recall,
                    'fmeasure': max_score.fmeasure
                }

            return result

        # -------------- 단일 정답 처리 -------------- #
        else:
            scores = self.scorer.score(prediction, reference)  # ROUGE 계산

            # 결과를 딕셔너리로 변환
            result = {}
            for rouge_type in self.rouge_types:         # 각 ROUGE 타입
                result[rouge_type] = {                  # 결과 저장
                    'precision': scores[rouge_type].precision,
                    'recall': scores[rouge_type].recall,
                    'fmeasure': scores[rouge_type].fmeasure
                }

            return result


    # ---------------------- 배치 ROUGE 계산 함수 ---------------------- #
    def calculate_batch(
        self,
        predictions: List[str],                         # 예측 요약 리스트
        references: Union[List[str], List[List[str]]]   # 정답 요약 리스트
    ) -> Dict[str, Dict[str, float]]:
        """
        배치 샘플의 ROUGE 점수 평균 계산

        Args:
            predictions: 예측 요약 리스트
            references: 정답 요약 리스트 (각 요소가 str 또는 List[str])

        Returns:
            Dict[str, Dict[str, float]]: {
                'rouge1': {'precision': 0.5, 'recall': 0.6, 'fmeasure': 0.55, 'std': 0.1},
                'rouge2': {...},
                'rougeL': {...}
            }
        """
        # -------------- 입력 검증 -------------- #
        if len(predictions) != len(references):         # 길이 불일치 확인
            raise ValueError(
                f"예측과 정답의 개수가 다릅니다: {len(predictions)} vs {len(references)}"
            )

        # -------------- 각 샘플의 ROUGE 계산 -------------- #
        all_scores = {rouge_type: [] for rouge_type in self.rouge_types}  # 점수 저장 딕셔너리

        for pred, ref in zip(predictions, references):  # 각 샘플 반복
            sample_scores = self.calculate_single(pred, ref)  # 단일 샘플 ROUGE 계산

            # 각 ROUGE 타입별 F1 점수 저장
            for rouge_type in self.rouge_types:
                all_scores[rouge_type].append(
                    sample_scores[rouge_type]['fmeasure']  # F1 점수만 저장
                )

        # -------------- 평균 및 표준편차 계산 -------------- #
        result = {}
        for rouge_type in self.rouge_types:             # 각 ROUGE 타입
            scores = all_scores[rouge_type]             # 해당 타입의 모든 점수

            result[rouge_type] = {                      # 결과 저장
                'fmeasure': float(np.mean(scores)),     # 평균 F1
                'std': float(np.std(scores)),           # 표준편차
                'min': float(np.min(scores)),           # 최소값
                'max': float(np.max(scores))            # 최대값
            }

        # -------------- ROUGE Sum 계산 (경진대회 기준) -------------- #
        # ROUGE-1, ROUGE-2, ROUGE-L F1의 합
        rouge_sum = sum(
            result[rouge_type]['fmeasure']
            for rouge_type in self.rouge_types
        )
        result['rouge_sum'] = {                         # ROUGE Sum 저장
            'fmeasure': rouge_sum,
            'std': 0.0,                                 # Sum은 표준편차 없음
            'min': 0.0,
            'max': 0.0
        }

        return result


    # ---------------------- 빈 점수 반환 함수 ---------------------- #
    def _empty_scores(self) -> Dict[str, Dict[str, float]]:
        """
        빈 입력에 대한 기본 점수 반환

        Returns:
            Dict[str, Dict[str, float]]: 모든 점수가 0인 딕셔너리
        """
        result = {}
        for rouge_type in self.rouge_types:             # 각 ROUGE 타입
            result[rouge_type] = {                      # 0 점수
                'precision': 0.0,
                'recall': 0.0,
                'fmeasure': 0.0
            }

        return result


# ==================== 편의 함수 ==================== #
# ---------------------- ROUGE 점수 계산 편의 함수 ---------------------- #
def calculate_rouge_scores(
    predictions: Union[str, List[str]],                 # 예측 요약
    references: Union[str, List[str], List[List[str]]],  # 정답 요약
    rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL'],  # ROUGE 타입
    use_stemmer: bool = False                           # 형태소 분석기 사용 여부
) -> Dict[str, Dict[str, float]]:
    """
    ROUGE 점수 계산 편의 함수

    Args:
        predictions: 예측 요약 (단일 또는 리스트)
        references: 정답 요약 (단일, 리스트, 또는 다중 정답 리스트)
        rouge_types: 계산할 ROUGE 타입
        use_stemmer: 형태소 분석기 사용 여부

    Returns:
        Dict[str, Dict[str, float]]: ROUGE 점수 딕셔너리
    """
    # -------------- ROUGE Calculator 생성 -------------- #
    calculator = RougeCalculator(rouge_types, use_stemmer)  # 계산기 생성

    # -------------- 단일 샘플 처리 -------------- #
    if isinstance(predictions, str):                    # 예측이 단일 문자열인 경우
        return calculator.calculate_single(predictions, references)  # 단일 계산

    # -------------- 배치 처리 -------------- #
    else:
        return calculator.calculate_batch(predictions, references)  # 배치 계산


# ==================== 점수 포맷팅 함수 ==================== #
# ---------------------- ROUGE 점수 출력 포맷팅 ---------------------- #
def format_rouge_scores(
    scores: Dict[str, Dict[str, float]],                # ROUGE 점수 딕셔너리
    decimal_places: int = 4                             # 소수점 자리수
) -> str:
    """
    ROUGE 점수를 보기 좋게 포맷팅

    Args:
        scores: ROUGE 점수 딕셔너리
        decimal_places: 소수점 자리수

    Returns:
        str: 포맷팅된 문자열
    """
    lines = []                                          # 출력 라인 리스트

    # -------------- 각 ROUGE 타입 출력 -------------- #
    for rouge_type, metrics in scores.items():          # 각 ROUGE 타입
        # ROUGE 타입 헤더
        lines.append(f"{rouge_type.upper()}:")

        # 각 메트릭 출력
        for metric_name, value in metrics.items():      # 각 메트릭
            formatted_value = f"{value:.{decimal_places}f}"  # 소수점 포맷팅
            lines.append(f"  {metric_name}: {formatted_value}")

        lines.append("")  # 빈 줄 추가

    return "\n".join(lines)                             # 문자열 결합
