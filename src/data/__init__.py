# ==================== 데이터 처리 모듈 ==================== #
"""
데이터 처리 모듈

데이터 전처리 및 데이터셋 클래스를 제공하는 모듈
- 전처리: 노이즈 제거, 정규화, 화자 추출
- 데이터셋: PyTorch Dataset 클래스
"""

# ---------------------- 데이터 모듈 Import ---------------------- #
from .preprocessor import DialoguePreprocessor            # 대화 전처리 클래스
from .dataset import (                                    # 데이터셋 클래스들
    DialogueSummarizationDataset,                         # 학습/검증용 데이터셋
    InferenceDataset                                      # 추론용 데이터셋
)

# ---------------------- 외부 노출 모듈 정의 ---------------------- #
__all__ = [
    'DialoguePreprocessor',          # 대화 전처리 클래스
    'DialogueSummarizationDataset',  # 학습/검증 데이터셋
    'InferenceDataset'               # 추론 데이터셋
]
