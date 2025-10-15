#!/usr/bin/env python3
"""
학습 CLI 진입점

사용법:
    python train.py --experiment exp7a
    python train.py --experiment exp7f

특징:
    - config/experiments.yaml에서 실험 설정 자동 로드
    - 가중치 샘플링 자동 적용 (use_weights=true인 경우)
    - scripts/ 디렉토리의 기존 유틸리티 활용
"""

import sys
import os
import argparse

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(os.path.dirname(current_dir), 'scripts')
sys.path.append(scripts_dir)

# Wandb 환경변수 로드
from dotenv import load_dotenv
load_dotenv('/Competition/NLP/.env')

import torch
from utils.config import load_experiment_config, validate_config
from utils.logger import setup_logger, log_experiment_start, log_experiment_end

# scripts 디렉토리의 기존 유틸리티
from utils import set_seed
from tokenizer_utils import load_tokenizer
from wandb_utils import init_wandb, finish_run

# core 모듈
from core.data import DataManager
from core.model import ModelManager
from core.trainer import Trainer


def parse_args():
    """
    CLI 인자를 파싱합니다
    """
    parser = argparse.ArgumentParser(description='KoBART 대화 요약 학습')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='실험 이름 (예: exp7a, exp7f)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/experiments.yaml',
        help='실험 설정 파일 경로 (기본값: ./config/experiments.yaml)'
    )
    return parser.parse_args()


def setup_wandb(config: dict, experiment_name: str):
    """
    Wandb를 초기화합니다

    Args:
        config: 설정 딕셔너리
        experiment_name: 실험 이름

    Returns:
        wandb run 객체
    """
    wandb_config_base = config.get('wandb', {})

    # 실험별 wandb 설정
    wandb_config = {
        "experiment": experiment_name,
        "description": config.get('description', ''),
        "data_source": config['data'].get('train_file', 'train.csv'),
        "sampling_strategy": "weighted" if config.get('data', {}).get('use_weights', False) else "natural",
        "weighted_sampling": config.get('data', {}).get('use_weights', False),
        "encoder_max_len": config['tokenizer']['encoder_max_len'],
        "decoder_max_len": config['tokenizer']['decoder_max_len'],
        "train_batch_size": config['training']['per_device_train_batch_size'],
        "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
        "effective_batch_size": config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'],
        "learning_rate": config['training']['learning_rate'],
        "num_epochs": config['training']['num_train_epochs'],
        "length_penalty": config['inference'].get('length_penalty', 1.0),
        "gradient_checkpointing": config['training']['gradient_checkpointing'],
    }

    # 가중치 설정 추가 (사용 시)
    if config.get('data', {}).get('use_weights', False):
        weight_config = config['data']['weight_config']
        wandb_config.update({
            "domain_threshold": weight_config.get('domain_threshold'),
            "subcluster_threshold": weight_config.get('subcluster_threshold'),
        })

    run = init_wandb(
        experiment_name=wandb_config_base.get('name', experiment_name),
        config=wandb_config,
        tags=wandb_config_base.get('tags', [experiment_name]),
        group=wandb_config_base.get('group', 'experiments'),
        notes=wandb_config_base.get('notes', config.get('description', ''))
    )

    return run


def main():
    """
    메인 학습 파이프라인
    """
    # 1. 인자 파싱
    args = parse_args()

    # 2. Logger 설정
    logger = setup_logger('train')

    # 3. Config 로드
    logger.info(f"실험 설정 로딩: {args.experiment}")
    config = load_experiment_config(args.config, args.experiment)
    validate_config(config)

    # 4. 실험 시작 로그
    log_experiment_start(args.experiment, config, logger)

    # 5. 시드 설정
    set_seed(config['training']['seed'])
    logger.info(f"Random seed 설정: {config['training']['seed']}")

    # 6. Wandb 초기화
    logger.info("Wandb 초기화 중...")
    run = setup_wandb(config, args.experiment)
    logger.info(f"Wandb Dashboard: {run.url}")

    try:
        # 7. Device 설정
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {device}")

        # 8. Tokenizer 로드
        logger.info("Tokenizer 로딩 중...")
        tokenizer = load_tokenizer(
            config['general']['model_name'],
            config['tokenizer']['special_tokens']
        )
        logger.info(f"Tokenizer 로드 완료 (vocab size: {len(tokenizer)})")

        # 9. 데이터 준비
        logger.info("데이터셋 준비 중...")
        data_manager = DataManager(config, tokenizer)
        train_dataset, val_dataset, sampler = data_manager.prepare_data()

        # 10. 모델 로드
        logger.info("모델 로딩 중...")
        model_manager = ModelManager(config, tokenizer, device)
        model = model_manager.load_model_for_training()

        # 11. 학습
        logger.info("학습 시작...")
        trainer = Trainer(config, args.experiment)
        trainer.train(model, tokenizer, train_dataset, val_dataset, sampler)

        # 12. 완료
        log_experiment_end(args.experiment, 'success', logger)

        # Wandb summary
        finish_run(summary_metrics={
            "status": "success",
            "experiment": args.experiment,
        })

    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}", exc_info=True)
        log_experiment_end(args.experiment, 'failed', logger)

        # Wandb summary
        finish_run(summary_metrics={
            "status": "failed",
            "error": str(e)
        })

        raise


if __name__ == "__main__":
    main()
