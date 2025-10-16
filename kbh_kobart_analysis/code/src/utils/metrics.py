#!/usr/bin/env python3
"""
메트릭 계산 유틸리티
"""

from rouge import Rouge
from typing import List, Dict


def compute_rouge_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    ROUGE 메트릭을 계산합니다.

    Args:
        predictions: 예측 요약문 리스트
        references: 참조 요약문 리스트

    Returns:
        ROUGE F1 scores 딕셔너리
        {
            'rouge-1': float,
            'rouge-2': float,
            'rouge-l': float
        }

    Example:
        >>> preds = ['오늘 날씨가 좋습니다.', '내일은 비가 옵니다.']
        >>> refs = ['오늘 날씨 좋음.', '내일 비 예상.']
        >>> scores = compute_rouge_metrics(preds, refs)
        >>> print(scores['rouge-1'])
        0.65
    """
    rouge = Rouge()

    try:
        results = rouge.get_scores(predictions, references, avg=True)
        # F1 score만 반환
        return {key: value["f"] for key, value in results.items()}
    except Exception as e:
        print(f"ROUGE 계산 오류: {e}")
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}


def format_rouge_scores(scores: Dict[str, float]) -> str:
    """
    ROUGE 점수를 포맷팅된 문자열로 변환합니다.

    Args:
        scores: ROUGE 점수 딕셔너리

    Returns:
        포맷팅된 문자열

    Example:
        >>> scores = {'rouge-1': 0.65, 'rouge-2': 0.45, 'rouge-l': 0.60}
        >>> print(format_rouge_scores(scores))
        ROUGE-1: 65.00% | ROUGE-2: 45.00% | ROUGE-L: 60.00%
    """
    return (
        f"ROUGE-1: {scores['rouge-1']*100:.2f}% | "
        f"ROUGE-2: {scores['rouge-2']*100:.2f}% | "
        f"ROUGE-L: {scores['rouge-l']*100:.2f}%"
    )
