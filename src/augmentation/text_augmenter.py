# ==================== 텍스트 증강 시스템 ==================== #
"""
데이터 증강 시스템

PRD 04: 성능 개선 전략 - 데이터 증강
"""

from typing import List, Dict, Optional, Tuple
import random
import re
import numpy as np


class TextAugmenter:
    """텍스트 증강 통합 클래스"""

    def __init__(self, seed: int = 42, logger=None):
        self.seed = seed
        self.logger = logger
        random.seed(seed)
        np.random.seed(seed)

    def _log(self, msg: str):
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def shuffle_sentences(self, dialogue: str, shuffle_prob: float = 0.5) -> str:
        """문장 순서 섞기"""
        if random.random() > shuffle_prob:
            return dialogue

        turns = re.split(r'(#Person\d+#:)', dialogue)
        turns = [t.strip() for t in turns if t.strip()]

        turn_pairs = []
        for i in range(0, len(turns), 2):
            if i + 1 < len(turns):
                turn_pairs.append(turns[i] + ' ' + turns[i+1])

        random.shuffle(turn_pairs)
        return ' '.join(turn_pairs)

    def batch_augment(
        self,
        dialogues: List[str],
        summaries: List[str],
        n_augmentations_per_sample: int = 2
    ) -> Tuple[List[str], List[str]]:
        """배치 증강"""
        aug_dialogues = []
        aug_summaries = []

        for dialogue, summary in zip(dialogues, summaries):
            aug_dialogues.append(dialogue)
            aug_summaries.append(summary)
            
            for _ in range(n_augmentations_per_sample):
                aug_dialogue = self.shuffle_sentences(dialogue)
                aug_dialogues.append(aug_dialogue)
                aug_summaries.append(summary)

        return aug_dialogues, aug_summaries


def create_augmenter(seed: int = 42, logger=None) -> TextAugmenter:
    return TextAugmenter(seed=seed, logger=logger)
