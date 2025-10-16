#!/usr/bin/env python3
"""
추론 CLI 진입점

사용법:
    python inference.py --experiment exp7a --checkpoint checkpoint-2068
    python inference.py --experiment exp7f --checkpoint checkpoint-1880 --output custom.csv

특징:
    - config/experiments.yaml에서 실험 설정 자동 로드
    - 체크포인트 경로 자동 생성 (output_dir/checkpoint_name)
    - Competition 제출 형식으로 자동 저장 (index 포함)
"""

import sys
import os
import argparse

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(os.path.dirname(current_dir), 'scripts')
sys.path.append(scripts_dir)

import torch
from src.utils.config import load_experiment_config, validate_config
from src.utils.logger import setup_logger

# scripts 디렉토리의 기존 유틸리티
from tokenizer_utils import load_tokenizer

# core 모듈
from src.core.model import ModelManager
from src.core.inference import Inferencer


def parse_args():
    """
    CLI 인자를 파싱합니다
    """
    parser = argparse.ArgumentParser(description='KoBART 대화 요약 추론')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='실험 이름 (예: exp7a, exp7f)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='체크포인트 이름 (예: checkpoint-2068)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/experiments.yaml',
        help='실험 설정 파일 경로 (기본값: ./config/experiments.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 파일명 (기본값: submission_{experiment}.csv)'
    )
    return parser.parse_args()


def main():
    """
    메인 추론 파이프라인
    """
    # 1. 인자 파싱
    args = parse_args()

    # 2. Logger 설정
    logger = setup_logger('inference')

    # 3. Config 로드
    logger.info(f"실험 설정 로딩: {args.experiment}")
    config = load_experiment_config(args.config, args.experiment)
    validate_config(config)

    logger.info("=" * 80)
    logger.info(f"추론 실험: {args.experiment}")
    logger.info(f"체크포인트: {args.checkpoint}")
    logger.info("=" * 80)

    try:
        # 4. Device 설정
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {device}")

        # 5. Tokenizer 로드
        logger.info("Tokenizer 로딩 중...")
        tokenizer = load_tokenizer(
            config['general']['model_name'],
            config['tokenizer']['special_tokens']
        )
        logger.info(f"Tokenizer 로드 완료 (vocab size: {len(tokenizer)})")

        # 6. 모델 로드
        model_manager = ModelManager(config, tokenizer, device)
        checkpoint_path = os.path.join(config['general']['output_dir'], args.checkpoint)
        model = model_manager.load_model_for_inference(checkpoint_path)

        # 7. 출력 경로 설정
        if args.output is None:
            # 기본값: /Competition/NLP/.../submission_{experiment}.csv
            output_path = f"/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_{args.experiment}.csv"
        else:
            # 사용자 지정 경로
            if not args.output.startswith('/'):
                # 상대 경로면 절대 경로로 변환
                output_path = os.path.abspath(args.output)
            else:
                output_path = args.output

        logger.info(f"출력 경로: {output_path}")

        # 8. 추론 실행
        inferencer = Inferencer(config, args.experiment, args.checkpoint)
        result_df = inferencer.run(model, tokenizer, output_path)

        logger.info("=" * 80)
        logger.info("✅ 추론 완료!")
        logger.info("=" * 80)
        logger.info(f"제출 파일: {output_path}")
        logger.info(f"총 {len(result_df)}개 요약 생성")

    except Exception as e:
        logger.error(f"추론 중 오류 발생: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
