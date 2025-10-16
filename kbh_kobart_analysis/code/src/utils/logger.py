#!/usr/bin/env python3
"""
로깅 유틸리티
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Optional


def setup_logger(name: str = 'train', level: int = logging.INFO) -> logging.Logger:
    """
    로거를 설정합니다.

    Args:
        name: 로거 이름
        level: 로깅 레벨 (기본값: INFO)

    Returns:
        설정된 로거

    Example:
        >>> logger = setup_logger('train')
        >>> logger.info('학습 시작')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 기존 핸들러 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 콘솔 핸들러
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # 포맷 설정
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def log_experiment_start(experiment_name: str, config: Dict, logger: Optional[logging.Logger] = None):
    """
    실험 시작 로그를 출력합니다.

    Args:
        experiment_name: 실험 이름
        config: 설정 딕셔너리
        logger: 로거 (None이면 print 사용)

    Example:
        >>> log_experiment_start('exp7a', config, logger)
    """
    log_func = logger.info if logger else print

    log_func("=" * 80)
    log_func(f"실험 시작: {experiment_name}")
    log_func("=" * 80)
    log_func(f"설명: {config.get('description', 'N/A')}")
    log_func(f"모델: {config['general']['model_name']}")
    log_func(f"출력 디렉토리: {config['general']['output_dir']}")
    log_func(f"Encoder Max Length: {config['tokenizer']['encoder_max_len']}")
    log_func(f"Train Batch Size: {config['training']['per_device_train_batch_size']}")
    log_func(f"Gradient Accumulation: {config['training']['gradient_accumulation_steps']}")
    log_func(f"Effective Batch Size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    log_func(f"Learning Rate: {config['training']['learning_rate']}")
    log_func(f"Epochs: {config['training']['num_train_epochs']}")
    log_func(f"Length Penalty: {config['inference'].get('length_penalty', 1.0)}")

    # 가중치 샘플링 정보
    if config.get('data', {}).get('use_weights', False):
        log_func("\n가중치 샘플링: 활성화")
        weight_config = config['data'].get('weight_config', {})
        log_func(f"  - 도메인 임계값: {weight_config.get('domain_threshold', 'N/A')}")
        log_func(f"  - 서브클러스터 임계값: {weight_config.get('subcluster_threshold', 'N/A')}")
    else:
        log_func("\n가중치 샘플링: 비활성화 (자연 분포)")

    log_func("=" * 80)


def log_experiment_end(experiment_name: str, status: str = 'success', logger: Optional[logging.Logger] = None):
    """
    실험 종료 로그를 출력합니다.

    Args:
        experiment_name: 실험 이름
        status: 상태 ('success' 또는 'failed')
        logger: 로거 (None이면 print 사용)

    Example:
        >>> log_experiment_end('exp7a', 'success', logger)
    """
    log_func = logger.info if logger else print

    log_func("\n" + "=" * 80)
    if status == 'success':
        log_func(f"✅ 실험 완료: {experiment_name}")
    else:
        log_func(f"❌ 실험 실패: {experiment_name}")
    log_func("=" * 80)
    log_func(f"완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_func("=" * 80)


def print_separator(title: str = "", logger: Optional[logging.Logger] = None):
    """
    구분선을 출력합니다.

    Args:
        title: 구분선 중앙에 표시할 제목
        logger: 로거 (None이면 print 사용)

    Example:
        >>> print_separator("데이터 로딩", logger)
        ========== 데이터 로딩 ==========
    """
    log_func = logger.info if logger else print

    if title:
        log_func(f"\n{'=' * 10} {title} {'=' * 10}")
    else:
        log_func("\n" + "=" * 80)
