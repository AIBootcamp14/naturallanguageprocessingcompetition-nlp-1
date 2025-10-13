"""
LLM 파인튜닝 학습 스크립트

PRD 08: LLM 파인튜닝 전략 구현
- QLoRA 4-bit 양자화
- Llama/Qwen 모델 지원
- Instruction/Chat format 지원
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.models.lora_loader import load_lora_model_and_tokenizer
from src.data.llm_dataset import create_llm_dataset
from src.training.llm_trainer import create_llm_trainer
from src.logging.logger import Logger
from src.utils.core.common import create_log_path, now
from src.utils.gpu_optimization.team_gpu_check import get_gpu_info, check_gpu_tier


def main():
    parser = argparse.ArgumentParser(description="LLM 파인튜닝 학습")
    parser.add_argument(
        "--experiment",
        type=str,
        default="llama_3.2_3b",
        help="실험 이름 (configs/models/ 또는 configs/experiments/의 파일명)"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        default=True,
        help="QLoRA (4-bit 양자화) 사용"
    )
    parser.add_argument(
        "--use_instruction_augmentation",
        action="store_true",
        help="Instruction Tuning 데이터 증강 사용"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 (작은 데이터셋, 짧은 학습)"
    )

    args = parser.parse_args()

    # Logger 초기화
    log_path = create_log_path("train", f"train_llm_{args.experiment}_{now('%Y%m%d_%H%M%S')}.log")
    logger = Logger(log_path, print_also=True)
    logger.start_redirect()

    try:
        logger.write("=" * 60)
        logger.write("LLM 파인튜닝 학습 시작")
        logger.write("=" * 60)
        logger.write(f"실험 이름: {args.experiment}")
        logger.write(f"QLoRA 사용: {args.use_qlora}")
        logger.write(f"Instruction 증강: {args.use_instruction_augmentation}")
        logger.write(f"디버그 모드: {args.debug}")

        # GPU 정보 출력
        logger.write("\n[GPU 정보]")
        gpu_info = get_gpu_info()
        for key, value in gpu_info.items():
            logger.write(f"  {key}: {value}")

        gpu_tier = check_gpu_tier()
        logger.write(f"  GPU Tier: {gpu_tier}")

        # Config 로드
        logger.write("\n[Config 로드]")
        config = load_config(args.experiment)
        logger.write(f"Config 로드 완료: {args.experiment}")

        # 디버그 모드 설정
        if args.debug:
            logger.write("\n[디버그 모드]")
            config.training.epochs = 1
            config.training.batch_size = 2
            config.training.gradient_accumulation_steps = 2
            config.wandb.enabled = False
            logger.write("  - Epochs: 1")
            logger.write("  - Batch size: 2")
            logger.write("  - Gradient accumulation: 2")
            logger.write("  - WandB: 비활성화")

        # 모델 및 토크나이저 로드
        logger.write("\n[모델 로드]")
        model, tokenizer = load_lora_model_and_tokenizer(
            config,
            use_lora=True,
            use_qlora=args.use_qlora,
            logger=logger
        )

        # 데이터 로드
        logger.write("\n[데이터 로드]")
        train_df = pd.read_csv("data/raw/train.csv")
        eval_df = pd.read_csv("data/raw/dev.csv")

        if args.debug:
            train_df = train_df.head(50)
            eval_df = eval_df.head(10)
            logger.write(f"  - 디버그: 학습 {len(train_df)}개, 검증 {len(eval_df)}개")
        else:
            logger.write(f"  - 학습: {len(train_df)}개")
            logger.write(f"  - 검증: {len(eval_df)}개")

        # Dataset 생성
        logger.write("\n[Dataset 생성]")
        train_dataset = create_llm_dataset(
            dialogues=train_df['dialogue'].tolist(),
            summaries=train_df['summary'].tolist(),
            tokenizer=tokenizer,
            encoder_max_len=config.tokenizer.encoder_max_len,
            decoder_max_len=config.tokenizer.decoder_max_len,
            format_type=config.dataset.format_type,
            use_instruction_augmentation=args.use_instruction_augmentation
        )

        eval_dataset = create_llm_dataset(
            dialogues=eval_df['dialogue'].tolist(),
            summaries=eval_df['summary'].tolist(),
            tokenizer=tokenizer,
            encoder_max_len=config.tokenizer.encoder_max_len,
            decoder_max_len=config.tokenizer.decoder_max_len,
            format_type=config.dataset.format_type,
            use_instruction_augmentation=False  # 평가는 증강 안 함
        )

        logger.write(f"Dataset 생성 완료")
        logger.write(f"  - 학습: {len(train_dataset)}개")
        logger.write(f"  - 검증: {len(eval_dataset)}개")
        logger.write(f"  - Format: {config.dataset.format_type}")
        logger.write(f"  - Instruction 증강: {args.use_instruction_augmentation}")

        # Trainer 생성
        logger.write("\n[Trainer 생성]")
        trainer = create_llm_trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            use_wandb=config.wandb.enabled and not args.debug,
            logger=logger
        )

        # 학습 실행
        logger.write("\n[학습 시작]")
        results = trainer.train()

        # 결과 출력
        logger.write("\n" + "=" * 60)
        logger.write("학습 완료")
        logger.write("=" * 60)
        logger.write(f"최종 모델 저장: {results['final_model_path']}")
        logger.write(f"학습 Loss: {results['train_loss']:.4f}")
        logger.write(f"학습 시간: {results['train_runtime']:.2f}초")

        if results['eval_metrics']:
            logger.write("\n평가 메트릭:")
            for key, value in results['eval_metrics'].items():
                logger.write(f"  {key}: {value:.4f}")

        logger.write("\n학습 로그 저장: " + log_path)
        logger.write("=" * 60)

    except Exception as e:
        logger.write("\n" + "=" * 60)
        logger.write("에러 발생")
        logger.write("=" * 60)
        logger.write(f"에러 메시지: {str(e)}")
        import traceback
        logger.write(traceback.format_exc())
        raise

    finally:
        logger.stop_redirect()
        logger.close()


if __name__ == "__main__":
    main()
