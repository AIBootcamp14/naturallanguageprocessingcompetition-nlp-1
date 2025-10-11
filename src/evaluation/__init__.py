# ==================== 평가 모듈 ==================== #
"""
평가 모듈

모델 성능 평가 지표 계산을 담당하는 모듈
- ROUGE 점수 계산
- Multi-reference 지원
- 배치 평가
"""

# ---------------------- 평가 모듈 Import ---------------------- #
from .metrics import RougeCalculator, calculate_rouge_scores  # ROUGE 계산 클래스 및 함수

# ---------------------- 외부 노출 모듈 정의 ---------------------- #
__all__ = [
    'RougeCalculator',              # ROUGE 계산 클래스
    'calculate_rouge_scores'        # ROUGE 계산 함수
]
