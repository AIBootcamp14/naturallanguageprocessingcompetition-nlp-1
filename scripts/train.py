# ==================== 학습 실행 스크립트 ==================== #
"""
전체 학습 파이프라인 실행 스크립트

사용법:
    python scripts/train.py --experiment baseline_kobart
    python scripts/train.py --experiment baseline_kobart --debug
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 서드파티 라이브러리 ---------------------- #
import pandas as pd

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.config import load_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer
from src.utils.config.seed import set_seed
from src.logging.logger import Logger
from src.utils.core.common import create_log_path
from src.utils.gpu_optimization.team_gpu_check import (
    get_gpu_info,
    check_gpu_tier,
    get_optimal_batch_size
)


# ==================== 메인 함수 ==================== #
def main():
    # -------------- 인자 파싱 -------------- #
    parser = argparse.ArgumentParser(description="모델 학습 스크립트")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="실험 이름 (예: baseline_kobart)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 (작은 데이터셋으로 빠른 테스트)"
    )
    args = parser.parse_args()

    # -------------- Logger 초기화 -------------- #
    log_path = create_log_path("outputs/logs", f"train_{args.experiment}")
    logger = Logger(log_path, print_also=True)
    logger.start_redirect()

    try:
        logger.write("=" * 60)
        logger.write(f"학습 시작: {args.experiment}")
        logger.write("=" * 60)

        # -------------- GPU 정보 출력 -------------- #
        logger.write("\n[GPU 정보]")
        gpu_info = get_gpu_info()
        for key, value in gpu_info.items():
            logger.write(f"  {key}: {value}")

        gpu_tier = check_gpu_tier()
        logger.write(f"  GPU Tier: {gpu_tier}")

        # -------------- 1. Config 로드 -------------- #
        logger.write("\n[1/6] Config 로딩...")
        config = load_config(args.experiment)

        # 디버그 모드 설정
        if args.debug:
            logger.write("  ⚠️ 디버그 모드 활성화")
            config.training.epochs = 2
            config.training.batch_size = 4
            config.wandb.enabled = False
        else:
            # GPU tier에 따른 배치 크기 최적화 제안
            optimal_batch_size = get_optimal_batch_size("kobart", gpu_tier)
            if config.training.batch_size != optimal_batch_size:
                logger.write(f"  💡 추천 배치 크기: {optimal_batch_size} (현재: {config.training.batch_size})")

        # 시드 설정
        set_seed(config.experiment.seed)
        logger.write(f"  ✅ Config 로드 완료 (seed: {config.experiment.seed})")

        # -------------- 2. 데이터 로드 -------------- #
        logger.write("\n[2/6] 데이터 로딩...")
        train_df = pd.read_csv(config.paths.train_data)
        eval_df = pd.read_csv(config.paths.dev_data)

        # 디버그 모드: 데이터 축소
        if args.debug:
            train_df = train_df.head(100)
            eval_df = eval_df.head(20)
            logger.write(f"  ⚠️ 디버그: 학습 {len(train_df)}개, 검증 {len(eval_df)}개")
        else:
            logger.write(f"  ✅ 학습 데이터: {len(train_df)}개")
            logger.write(f"  ✅ 검증 데이터: {len(eval_df)}개")

        # -------------- 3. 모델 및 토크나이저 로드 -------------- #
        logger.write("\n[3/6] 모델 로딩...")
        model, tokenizer = load_model_and_tokenizer(config, logger=logger)
        logger.write("  ✅ 모델 로드 완료")

        # -------------- 4. Dataset 생성 -------------- #
        logger.write("\n[4/6] Dataset 생성...")
        train_dataset = DialogueSummarizationDataset(
            dialogues=train_df['dialogue'].tolist(),
            summaries=train_df['summary'].tolist(),
            tokenizer=tokenizer,
            encoder_max_len=config.tokenizer.encoder_max_len,
            decoder_max_len=config.tokenizer.decoder_max_len,
            preprocess=True
        )

        eval_dataset = DialogueSummarizationDataset(
            dialogues=eval_df['dialogue'].tolist(),
            summaries=eval_df['summary'].tolist(),
            tokenizer=tokenizer,
            encoder_max_len=config.tokenizer.encoder_max_len,
            decoder_max_len=config.tokenizer.decoder_max_len,
            preprocess=True
        )

        logger.write(f"  ✅ 학습 Dataset: {len(train_dataset)}개")
        logger.write(f"  ✅ 검증 Dataset: {len(eval_dataset)}개")

        # -------------- 5. Trainer 생성 및 학습 -------------- #
        logger.write("\n[5/6] 학습 시작...")
        trainer = create_trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            use_wandb=config.wandb.enabled and not args.debug,
            logger=logger
        )

        # 학습 실행
        results = trainer.train()

        # -------------- 6. 결과 출력 -------------- #
        logger.write("\n[6/6] 학습 완료!")
        logger.write(f"  최종 모델 저장: {results['final_model_path']}")
        if 'best_model_checkpoint' in results:
            logger.write(f"  최상 체크포인트: {results['best_model_checkpoint']}")

        if 'eval_metrics' in results and results['eval_metrics']:
            logger.write("\n  최종 평가 결과:")
            for key, value in results['eval_metrics'].items():
                if 'rouge' in key:
                    logger.write(f"    {key}: {value:.4f}")

        logger.write("\n" + "=" * 60)
        logger.write("🎉 학습 완료!")
        logger.write("=" * 60)

    finally:
        # Logger 정리
        logger.stop_redirect()
        logger.close()


# ==================== 실행부 ==================== #
if __name__ == "__main__":
    main()
