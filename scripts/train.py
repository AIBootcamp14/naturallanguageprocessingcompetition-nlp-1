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

    print("=" * 60)
    print(f"학습 시작: {args.experiment}")
    print("=" * 60)

    # -------------- 1. Config 로드 -------------- #
    print("\n[1/6] Config 로딩...")
    config = load_config(args.experiment)

    # 디버그 모드 설정
    if args.debug:
        print("  ⚠️ 디버그 모드 활성화")
        config.training.epochs = 2
        config.training.batch_size = 4
        config.wandb.enabled = False

    # 시드 설정
    set_seed(config.experiment.seed)
    print(f"  ✅ Config 로드 완료 (seed: {config.experiment.seed})")

    # -------------- 2. 데이터 로드 -------------- #
    print("\n[2/6] 데이터 로딩...")
    train_df = pd.read_csv(config.paths.train_data)
    eval_df = pd.read_csv(config.paths.dev_data)

    # 디버그 모드: 데이터 축소
    if args.debug:
        train_df = train_df.head(100)
        eval_df = eval_df.head(20)
        print(f"  ⚠️ 디버그: 학습 {len(train_df)}개, 검증 {len(eval_df)}개")
    else:
        print(f"  ✅ 학습 데이터: {len(train_df)}개")
        print(f"  ✅ 검증 데이터: {len(eval_df)}개")

    # -------------- 3. 모델 및 토크나이저 로드 -------------- #
    print("\n[3/6] 모델 로딩...")
    model, tokenizer = load_model_and_tokenizer(config)
    print("  ✅ 모델 로드 완료")

    # -------------- 4. Dataset 생성 -------------- #
    print("\n[4/6] Dataset 생성...")
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

    print(f"  ✅ 학습 Dataset: {len(train_dataset)}개")
    print(f"  ✅ 검증 Dataset: {len(eval_dataset)}개")

    # -------------- 5. Trainer 생성 및 학습 -------------- #
    print("\n[5/6] 학습 시작...")
    trainer = create_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        use_wandb=config.wandb.enabled and not args.debug
    )

    # 학습 실행
    results = trainer.train()

    # -------------- 6. 결과 출력 -------------- #
    print("\n[6/6] 학습 완료!")
    print(f"  최종 모델 저장: {results['final_model_path']}")
    if 'best_model_checkpoint' in results:
        print(f"  최상 체크포인트: {results['best_model_checkpoint']}")

    if 'eval_metrics' in results and results['eval_metrics']:
        print("\n  최종 평가 결과:")
        for key, value in results['eval_metrics'].items():
            if 'rouge' in key:
                print(f"    {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("🎉 학습 완료!")
    print("=" * 60)


# ==================== 실행부 ==================== #
if __name__ == "__main__":
    main()
