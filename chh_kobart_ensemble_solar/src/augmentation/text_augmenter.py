# ==================== 텍스트 증강 시스템 ==================== #
"""
데이터 증강 시스템

주요 기능:
- PRD 04: 성능 개선 전략 - 데이터 증강
- 문장 순서 섞기, 배치 증강 등 다양한 증강 기법 제공
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import random
import re
from typing import List, Dict, Optional, Tuple

# ---------------------- 서드파티 라이브러리 ---------------------- #
import numpy as np


# ==================== TextAugmenter 클래스 ==================== #
class TextAugmenter:
    """텍스트 증강 통합 클래스"""

    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(self, seed: int = 42, logger=None):
        """
        Args:
            seed: 랜덤 시드
            logger: 로거 객체
        """
        self.seed = seed                # 랜덤 시드 저장
        self.logger = logger            # 로거 저장

        # 랜덤 시드 설정
        random.seed(seed)               # Python 랜덤 시드 설정
        np.random.seed(seed)            # NumPy 랜덤 시드 설정

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

    # ---------------------- 문장 순서 섞기 ---------------------- #
    def shuffle_sentences(self, dialogue: str, shuffle_prob: float = 0.5) -> str:
        """
        대화 턴 순서 무작위 섞기

        Args:
            dialogue: 원본 대화 텍스트
            shuffle_prob: 섞기 실행 확률 (0.0 ~ 1.0)

        Returns:
            순서가 섞인 대화 텍스트
        """
        # -------------- 확률적 실행 여부 결정 -------------- #
        # 확률에 따라 원본 그대로 반환
        if random.random() > shuffle_prob:
            return dialogue

        # -------------- 대화 턴 분리 -------------- #
        # #Person1#, #Person2# 등의 화자 태그로 분리
        turns = re.split(r'(#Person\d+#:)', dialogue)
        turns = [t.strip() for t in turns if t.strip()]

        # -------------- 턴 쌍 생성 -------------- #
        # 화자 태그와 발화 내용을 쌍으로 묶기
        turn_pairs = []
        for i in range(0, len(turns), 2):
            # 화자 태그와 발화가 모두 있는 경우
            if i + 1 < len(turns):
                turn_pairs.append(turns[i] + ' ' + turns[i+1])

        # -------------- 턴 순서 섞기 -------------- #
        random.shuffle(turn_pairs)      # 턴 쌍 순서 무작위 섞기

        # 섞인 턴들을 공백으로 연결하여 반환
        return ' '.join(turn_pairs)

    # ---------------------- 배치 증강 메서드 ---------------------- #
    def batch_augment(
        self,
        dialogues: List[str],
        summaries: List[str],
        n_augmentations_per_sample: int = 2
    ) -> Tuple[List[str], List[str]]:
        """
        배치 단위로 데이터 증강 수행

        Args:
            dialogues: 대화 텍스트 리스트
            summaries: 요약 텍스트 리스트
            n_augmentations_per_sample: 샘플당 생성할 증강 데이터 수

        Returns:
            증강된 대화 리스트, 증강된 요약 리스트의 튜플
        """
        # -------------- 결과 저장 리스트 초기화 -------------- #
        aug_dialogues = []              # 증강된 대화 저장
        aug_summaries = []              # 증강된 요약 저장

        # -------------- 각 샘플별 증강 수행 -------------- #
        for dialogue, summary in zip(dialogues, summaries):
            # 원본 데이터 추가
            aug_dialogues.append(dialogue)
            aug_summaries.append(summary)

            # -------------- 증강 데이터 생성 -------------- #
            # 지정된 횟수만큼 증강 반복
            for _ in range(n_augmentations_per_sample):
                # 문장 순서 섞기로 증강
                aug_dialogue = self.shuffle_sentences(dialogue)
                aug_dialogues.append(aug_dialogue)
                aug_summaries.append(summary)

        # 증강된 대화와 요약 반환
        return aug_dialogues, aug_summaries


# ==================== 헬퍼 함수 ==================== #
# ---------------------- TextAugmenter 생성 함수 ---------------------- #
def create_augmenter(seed: int = 42, logger=None) -> TextAugmenter:
    """
    TextAugmenter 인스턴스 생성 헬퍼 함수

    Args:
        seed: 랜덤 시드
        logger: 로거 객체

    Returns:
        초기화된 TextAugmenter 인스턴스
    """
    return TextAugmenter(seed=seed, logger=logger)
