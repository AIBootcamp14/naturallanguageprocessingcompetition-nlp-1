#!/usr/bin/env python3
"""
Config 파싱 유틸리티
"""

import yaml
from typing import Dict
import copy


def load_experiment_config(config_path: str, experiment_name: str) -> Dict:
    """
    experiments.yaml에서 특정 실험 설정을 로드하고 defaults와 병합합니다.

    Args:
        config_path: experiments.yaml 파일 경로
        experiment_name: 실험 이름 (예: 'exp7a', 'exp7f')

    Returns:
        병합된 설정 딕셔너리

    Raises:
        ValueError: 실험 이름이 존재하지 않는 경우

    Example:
        >>> config = load_experiment_config('./config/experiments.yaml', 'exp7a')
        >>> print(config['general']['output_dir'])
        '/Competition/NLP/.../submission_exp7a'
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)

    if experiment_name not in all_configs['experiments']:
        raise ValueError(
            f"실험 '{experiment_name}'이(가) 존재하지 않습니다. "
            f"사용 가능한 실험: {list(all_configs['experiments'].keys())}"
        )

    # defaults 복사
    config = copy.deepcopy(all_configs['defaults'])

    # 실험 설정 병합
    experiment_config = all_configs['experiments'][experiment_name]
    config = merge_configs(config, experiment_config)

    # 메타데이터 추가
    config['experiment_name'] = experiment_name
    config['description'] = experiment_config.get('description', '')

    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    두 config 딕셔너리를 깊은 병합합니다.

    Args:
        base: 기본 설정 (defaults)
        override: 오버라이드할 설정 (experiment config)

    Returns:
        병합된 설정 딕셔너리

    Note:
        - 중첩된 딕셔너리도 재귀적으로 병합됩니다.
        - override의 값이 우선순위를 가집니다.

    Example:
        >>> base = {'a': {'b': 1, 'c': 2}, 'd': 3}
        >>> override = {'a': {'c': 99}, 'e': 4}
        >>> result = merge_configs(base, override)
        >>> print(result)
        {'a': {'b': 1, 'c': 99}, 'd': 3, 'e': 4}
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 중첩된 딕셔너리인 경우 재귀적으로 병합
            result[key] = merge_configs(result[key], value)
        else:
            # 그 외의 경우 override 값 사용
            result[key] = copy.deepcopy(value)

    return result


def validate_config(config: Dict) -> bool:
    """
    설정의 유효성을 검사합니다.

    Args:
        config: 검사할 설정 딕셔너리

    Returns:
        유효하면 True, 아니면 False

    Raises:
        ValueError: 필수 설정이 누락된 경우
    """
    required_sections = ['general', 'tokenizer', 'training', 'inference']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"필수 섹션 '{section}'이(가) 누락되었습니다.")

    # 필수 필드 검사
    if 'model_name' not in config['general']:
        raise ValueError("general.model_name이 필요합니다.")

    if 'output_dir' not in config['general']:
        raise ValueError("general.output_dir이 필요합니다.")

    return True
