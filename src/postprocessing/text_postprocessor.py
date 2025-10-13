"""
텍스트 후처리 시스템

PRD 04: 성능 개선 전략 - 후처리
"""

import re
from typing import List


class TextPostprocessor:
    """텍스트 후처리 클래스"""

    def __init__(self, logger=None):
        self.logger = logger

    def _log(self, msg: str):
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def normalize_punctuation(self, text: str) -> str:
        """문장 부호 정규화"""
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])+', r'\1', text)
        return text

    def adjust_length(self, text: str, max_length: int = 200) -> str:
        """길이 조절"""
        if len(text) > max_length:
            sentences = re.split(r'([.!?])', text)
            result = []
            current_length = 0
            
            for sent in sentences:
                if current_length + len(sent) <= max_length:
                    result.append(sent)
                    current_length += len(sent)
                else:
                    break
            
            return ''.join(result)
        return text

    def process(self, text: str, max_length: int = 200) -> str:
        """종합 후처리"""
        text = self.normalize_punctuation(text)
        text = self.adjust_length(text, max_length)
        return text.strip()

    def batch_process(self, texts: List[str], max_length: int = 200) -> List[str]:
        """배치 후처리"""
        return [self.process(text, max_length) for text in texts]


def create_postprocessor(logger=None) -> TextPostprocessor:
    return TextPostprocessor(logger=logger)
