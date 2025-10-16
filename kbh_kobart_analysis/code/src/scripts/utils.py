#!/usr/bin/env python3
"""
공통 유틸리티 함수 모음
"""

import yaml
import torch
import random
import numpy as np
from typing import List, Dict
import pandas as pd
import os


def load_config(config_path: str) -> dict:
    """
    YAML 설정 파일을 로드합니다.

    Args:
        config_path: config.yaml 파일 경로

    Returns:
        설정 딕셔너리

    Examples:
        >>> config = load_config('./config.yaml')
        >>> print(config['general']['model_name'])
    """
    # baseline.ipynb Cell 12 참조
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    return loaded_config


def save_config(config_data: dict, config_path: str) -> None:
    """
    설정을 YAML 파일로 저장합니다.

    Args:
        config_data: 설정 딕셔너리
        config_path: 저장할 파일 경로

    Examples:
        >>> config_data = {'general': {'model_name': 'digit82/kobart-summarization'}}
        >>> save_config(config_data, './new_config.yaml')
    """
    with open(config_path, "w") as file:
        yaml.dump(config_data, file, allow_unicode=True)


def get_device() -> torch.device:
    """
    사용 가능한 디바이스를 반환합니다.

    Returns:
        torch.device ('cuda:0' 또는 'cpu')

    Note:
        baseline.ipynb에서는 'cuda:0'으로 고정되어 있음
        개선: torch.cuda.is_available() 체크 추가

    Examples:
        >>> device = get_device()
        >>> print(device)  # cuda:0 or cpu
    """
    # baseline.ipynb에서는 'cuda:0'으로 고정되어 있음
    # 개선: torch.cuda.is_available() 체크
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def set_seed(seed: int) -> None:
    """
    재현성을 위해 랜덤 시드를 설정합니다.

    Args:
        seed: 시드 값

    Examples:
        >>> set_seed(42)
        >>> # 이제 모든 랜덤 연산이 재현 가능합니다
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_special_tokens(texts: List[str], tokens: List[str]) -> List[str]:
    """
    텍스트 리스트에서 특수 토큰을 제거합니다.

    Args:
        texts: 텍스트 리스트
        tokens: 제거할 토큰 리스트 (예: ['<s>', '</s>', '<pad>'])

    Returns:
        특수 토큰이 제거된 텍스트 리스트

    Examples:
        >>> texts = ['<s> Hello world </s>', '<s> Good morning <pad>']
        >>> tokens = ['<s>', '</s>', '<pad>']
        >>> cleaned = remove_special_tokens(texts, tokens)
        >>> print(cleaned)  # ['  Hello world  ', '  Good morning  ']
    """
    # baseline.ipynb Cell 42 참조
    cleaned_texts = texts.copy()
    for token in tokens:
        cleaned_texts = [s.replace(token, " ") for s in cleaned_texts]
    return cleaned_texts


def validate_csv(csv_path: str) -> Dict[str, any]:
    """
    제출용 CSV 파일의 유효성을 검증합니다.

    Args:
        csv_path: CSV 파일 경로

    Returns:
        검증 결과 딕셔너리
        {
            'valid': bool,
            'num_samples': int,
            'has_index': bool,
            'columns': list,
            'errors': list
        }

    Examples:
        >>> result = validate_csv('./output.csv')
        >>> if result['valid']:
        ...     print("CSV 파일이 유효합니다.")
        ... else:
        ...     print(f"오류: {result['errors']}")
    """
    result = {
        'valid': True,
        'errors': []
    }

    if not os.path.exists(csv_path):
        result['valid'] = False
        result['errors'].append(f"파일이 존재하지 않음: {csv_path}")
        return result

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"CSV 읽기 오류: {str(e)}")
        return result

    result['num_samples'] = len(df)
    result['columns'] = df.columns.tolist()

    # 필수 컬럼 확인
    if 'fname' not in df.columns or 'summary' not in df.columns:
        result['valid'] = False
        result['errors'].append("필수 컬럼 누락: fname, summary")

    # 인덱스 컬럼 확인 (submission 파일은 index를 포함해야 함)
    # CSV 파일에 unnamed 컬럼이 있으면 인덱스가 저장된 것으로 판단
    has_index_col = any('unnamed' in col.lower() for col in df.columns)
    result['has_index'] = has_index_col

    # 샘플 수 확인 (test는 499개)
    if len(df) != 499:
        result['errors'].append(f"샘플 수 오류: {len(df)} (기대값: 499)")

    # 빈 값 확인
    if df['fname'].isna().any():
        result['valid'] = False
        result['errors'].append("fname 컬럼에 빈 값이 존재합니다")

    if df['summary'].isna().any():
        result['valid'] = False
        result['errors'].append("summary 컬럼에 빈 값이 존재합니다")

    return result


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Utils 모듈 테스트 ===")

    # 1. get_device 테스트
    device = get_device()
    print(f"1. Device: {device}")

    # 2. set_seed 테스트
    set_seed(42)
    print(f"2. Seed set to 42")

    # 3. remove_special_tokens 테스트
    texts = ['<s> 안녕하세요 </s>', '<s> 반갑습니다 <pad>']
    tokens = ['<s>', '</s>', '<pad>']
    cleaned = remove_special_tokens(texts, tokens)
    print(f"3. Cleaned texts: {cleaned}")

    print("\n✅ 모든 테스트 완료")