"""
텍스트 후처리 시스템

주요 기능:
- PRD 04: 성능 개선 전략 - 후처리
- 문장 부호 정규화, 길이 조절 등 텍스트 품질 개선
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import re
from typing import List


# ==================== TextPostprocessor 클래스 ==================== #
class TextPostprocessor:
    """텍스트 후처리 클래스"""

    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(self, logger=None):
        """
        Args:
            logger: 로거 객체
        """
        self.logger = logger            # 로거 저장

    # ---------------------- 로그 출력 메서드 ---------------------- #
    def _log(self, msg: str):
        """
        로그 메시지 출력

        Args:
            msg: 출력할 메시지
        """
        # -------------- 로거 사용 여부 확인 -------------- #
        # 로거가 있으면 로거로 출력
        if self.logger:
            self.logger.write(msg)
        # 로거가 없으면 print로 출력
        else:
            print(msg)

    # ---------------------- 문장 부호 정규화 ---------------------- #
    def normalize_punctuation(self, text: str) -> str:
        """
        문장 부호 정규화 처리

        Args:
            text: 원본 텍스트

        Returns:
            정규화된 텍스트
        """
        # -------------- 공백과 문장 부호 사이 정규화 -------------- #
        # 문장 부호 앞의 불필요한 공백 제거
        text = re.sub(r'\s+([.,!?])', r'\1', text)

        # -------------- 중복 문장 부호 제거 -------------- #
        # 연속된 동일 문장 부호를 하나로 축약
        text = re.sub(r'([.,!?])+', r'\1', text)

        return text

    # ---------------------- 길이 조절 메서드 ---------------------- #
    def adjust_length(self, text: str, max_length: int = 200) -> str:
        """
        텍스트 길이 조절 (최대 길이 제한)

        Args:
            text: 원본 텍스트
            max_length: 최대 허용 길이

        Returns:
            길이가 조절된 텍스트
        """
        # -------------- 길이 초과 여부 확인 -------------- #
        # 최대 길이를 초과하는 경우에만 처리
        if len(text) > max_length:
            # -------------- 문장 단위로 분리 -------------- #
            # 문장 부호를 기준으로 텍스트 분리
            sentences = re.split(r'([.!?])', text)

            # -------------- 길이 제한 내에서 문장 추가 -------------- #
            result = []                 # 결과 저장 리스트
            current_length = 0          # 현재 누적 길이

            # 각 문장을 순회하며 추가
            for sent in sentences:
                # 현재 문장을 추가해도 길이 제한 내인 경우
                if current_length + len(sent) <= max_length:
                    result.append(sent)
                    current_length += len(sent)
                # 길이 제한 초과 시 중단
                else:
                    break

            # 추가된 문장들을 연결하여 반환
            return ''.join(result)

        # 길이 제한 이내면 원본 반환
        return text

    # ---------------------- 종합 후처리 메서드 ---------------------- #
    def process(self, text: str, max_length: int = 200) -> str:
        """
        텍스트 종합 후처리 파이프라인

        Args:
            text: 원본 텍스트
            max_length: 최대 허용 길이

        Returns:
            후처리된 텍스트
        """
        # -------------- 후처리 단계별 실행 -------------- #
        # 1단계: 문장 부호 정규화
        text = self.normalize_punctuation(text)

        # 2단계: 길이 조절
        text = self.adjust_length(text, max_length)

        # 앞뒤 공백 제거 후 반환
        return text.strip()

    # ---------------------- 배치 후처리 메서드 ---------------------- #
    def batch_process(self, texts: List[str], max_length: int = 200) -> List[str]:
        """
        여러 텍스트를 배치로 후처리

        Args:
            texts: 텍스트 리스트
            max_length: 최대 허용 길이

        Returns:
            후처리된 텍스트 리스트
        """
        # 각 텍스트에 대해 후처리 실행
        return [self.process(text, max_length) for text in texts]


# ==================== 헬퍼 함수 ==================== #
# ---------------------- TextPostprocessor 생성 함수 ---------------------- #
def create_postprocessor(logger=None) -> TextPostprocessor:
    """
    TextPostprocessor 인스턴스 생성 헬퍼 함수

    Args:
        logger: 로거 객체

    Returns:
        초기화된 TextPostprocessor 인스턴스
    """
    return TextPostprocessor(logger=logger)
