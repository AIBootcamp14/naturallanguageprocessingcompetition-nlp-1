"""
허깅페이스 모델 로더

PRD 04: 추론 최적화
사전학습 모델 로드 및 캐싱 관리
"""

# ------------------------- 표준 라이브러리 ------------------------- #
from typing import Tuple, Optional

# ------------------------- 서드파티 라이브러리 ------------------------- #
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


# ==================== HuggingFace 모델 로더 클래스 ==================== #
class HuggingFaceModelLoader:
    """
    허깅페이스 사전학습 모델 로더

    지원 모델:
    - KoBART 기반 (gogamza/kobart-base-v2, digit82/kobart-summarization)
    - T5 기반
    - BART 기반

    주요 기능:
    - 모델 및 토크나이저 로드
    - 메모리 캐싱으로 중복 로드 방지
    - GPU/CPU 디바이스 자동 설정
    """

    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(self, device=None, logger=None):
        """
        Args:
            device: 추론 디바이스 (None이면 자동 감지)
            logger: Logger 인스턴스
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.cache = {}                                 # 모델 캐시 딕셔너리

    # ---------------------- 모델 로드 메서드 ---------------------- #
    def load_model(
        self,
        model_name: str,
        use_cache: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        허깅페이스 모델 로드

        Args:
            model_name: 허깅페이스 모델 이름 (예: gogamza/kobart-base-v2)
            use_cache: 캐시 사용 여부

        Returns:
            (model, tokenizer) 튜플
        """
        # -------------- 캐시 확인 -------------- #
        if use_cache and model_name in self.cache:
            self._log(f"캐시에서 로드: {model_name}")
            return self.cache[model_name]

        self._log(f"모델 로드 중: {model_name}")

        # -------------- 모델 및 토크나이저 로드 -------------- #
        try:
            # Transformers 모듈 임포트
            from transformers import (
                AutoConfig,
                AutoModelForSeq2SeqLM,
                AutoTokenizer
            )

            # Config 로드
            config = AutoConfig.from_pretrained(model_name)

            # 모델 로드 (Seq2Seq 전용)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = model.to(self.device)               # GPU/CPU 전송
            model.eval()                                # 평가 모드 설정

            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # -------------- 캐시 저장 -------------- #
            if use_cache:
                self.cache[model_name] = (model, tokenizer)

            self._log(f"  ✅ 로드 완료: {model_name}")
            return model, tokenizer

        # -------------- 예외 처리 -------------- #
        except Exception as e:
            self._log(f"  ❌ 로드 실패: {model_name}")
            self._log(f"     에러: {str(e)}")
            raise

    # ---------------------- 캐시 초기화 메서드 ---------------------- #
    def clear_cache(self):
        """
        캐시 초기화 및 GPU 메모리 해제
        """
        self.cache.clear()                              # 딕셔너리 캐시 초기화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                    # GPU 메모리 해제

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
