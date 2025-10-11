#!/usr/bin/env python3
# ==================== NLP 대화 요약 통합 학습 스크립트 ==================== #
"""
NLP 대화 요약 통합 학습 스크립트
PRD 14번 "실행 옵션 시스템" 구현

사용법:
    # 단일 모델
    python scripts/train.py --mode single --models kobart

    # K-Fold 교차검증
    python scripts/train.py --mode kfold --models solar-10.7b --k_folds 5

    # 다중 모델 앙상블
    python scripts/train.py --mode multi_model --models kobart llama-3.2-3b

    # Optuna 최적화
    python scripts/train.py --mode optuna --optuna_trials 50

    # 풀 파이프라인
    python scripts/train.py --mode full --models all --use_tta
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.logging.logger import Logger
from src.utils.config.seed import set_seed


# ==================== 인자 파싱 ==================== #
def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='NLP 대화 요약 모델 학습 - 유연한 실행 옵션',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ==================== 기본 설정 ====================
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'kfold', 'multi_model', 'optuna', 'full'],
        help='''실행 모드 선택:
        single: 단일 모델 학습 (빠른 실험)
        kfold: K-Fold 교차 검증 (안정성)
        multi_model: 다중 모델 앙상블 (성능)
        optuna: 하이퍼파라미터 최적화 (자동화)
        full: 전체 파이프라인 (최종 제출)'''
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='설정 파일 경로'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='실험명 (자동 생성: {mode}_{model}_{timestamp})'
    )

    # ==================== 모델 선택 ====================
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['kobart'],
        choices=[
            'kobart',
            'solar-10.7b',
            'polyglot-ko-12.8b',
            'llama-3.2-korean-3b',
            'qwen3-4b',
            'kullm-v2',
            'all'  # 모든 모델
        ],
        help='사용할 모델 (multi_model 모드에서 여러 개 선택 가능)'
    )

    # ==================== 학습 설정 ====================
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='에폭 수 (None: config 파일 값 사용)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='배치 크기 (None: config 파일 값 사용 또는 자동 탐색)'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='학습률 (None: config 파일 값 사용)'
    )

    # ==================== K-Fold 설정 ====================
    parser.add_argument(
        '--k_folds',
        type=int,
        default=5,
        help='K-Fold 수 (kfold 모드)'
    )

    parser.add_argument(
        '--fold_seed',
        type=int,
        default=42,
        help='Fold 분할 시드'
    )

    # ==================== 앙상블 설정 ====================
    parser.add_argument(
        '--ensemble_strategy',
        type=str,
        default='weighted_avg',
        choices=[
            'averaging',
            'weighted_avg',
            'majority_vote',
            'stacking',
            'blending',
            'rouge_weighted'
        ],
        help='앙상블 전략'
    )

    parser.add_argument(
        '--ensemble_weights',
        type=float,
        nargs='+',
        default=None,
        help='모델별 가중치 (자동 최적화 가능)'
    )

    # ==================== TTA 설정 ====================
    parser.add_argument(
        '--use_tta',
        action='store_true',
        help='Test Time Augmentation 사용'
    )

    parser.add_argument(
        '--tta_strategies',
        type=str,
        nargs='+',
        default=['paraphrase'],
        choices=['paraphrase', 'reorder', 'synonym', 'mask'],
        help='TTA 전략'
    )

    parser.add_argument(
        '--tta_num_aug',
        type=int,
        default=3,
        help='TTA 증강 수'
    )

    # ==================== Optuna 설정 ====================
    parser.add_argument(
        '--optuna_trials',
        type=int,
        default=100,
        help='Optuna 시도 횟수'
    )

    parser.add_argument(
        '--optuna_timeout',
        type=int,
        default=7200,
        help='Optuna 제한 시간 (초)'
    )

    parser.add_argument(
        '--optuna_sampler',
        type=str,
        default='tpe',
        choices=['tpe', 'gp', 'random', 'cmaes'],
        help='Optuna 샘플러'
    )

    parser.add_argument(
        '--optuna_pruner',
        type=str,
        default='median',
        choices=['median', 'percentile', 'hyperband'],
        help='Optuna 가지치기'
    )

    # ==================== 로깅 및 모니터링 ====================
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='WandB 사용'
    )

    parser.add_argument(
        '--wandb_project',
        type=str,
        default='dialogue-summarization',
        help='WandB 프로젝트명'
    )

    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='시각화 저장'
    )

    # ==================== 기타 옵션 ====================
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='랜덤 시드'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 (적은 데이터)'
    )

    # ==================== 데이터 경로 ====================
    parser.add_argument(
        '--train_data',
        type=str,
        default='data/raw/train.csv',
        help='학습 데이터 경로'
    )

    parser.add_argument(
        '--dev_data',
        type=str,
        default='data/raw/dev.csv',
        help='검증 데이터 경로'
    )

    # ==================== 출력 경로 ====================
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='출력 디렉토리 (None: 자동 생성)'
    )

    return parser.parse_args()


# ==================== 환경 설정 ====================
def setup_environment(args):
    """환경 설정"""
    # 시드 설정
    set_seed(args.seed)

    # 실험명 자동 생성
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.models[0].replace('-', '_') if args.models else 'default'
        args.experiment_name = f"{args.mode}_{model_name}_{timestamp}"

    # 출력 디렉토리 생성
    if args.output_dir is None:
        output_dir = Path(f"experiments/{args.experiment_name}")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    # 로거 설정
    log_path = output_dir / "train.log"
    logger = Logger(log_path, print_also=True)
    logger.start_redirect()

    return logger


# ==================== Trainer 선택 ====================
def get_trainer(args, logger):
    """모드에 따른 Trainer 선택"""
    if args.mode == 'single':
        from src.trainers import SingleModelTrainer
        return SingleModelTrainer(args, logger)

    elif args.mode == 'kfold':
        from src.trainers import KFoldTrainer
        return KFoldTrainer(args, logger)

    elif args.mode == 'multi_model':
        from src.trainers import MultiModelEnsembleTrainer
        return MultiModelEnsembleTrainer(args, logger)

    elif args.mode == 'optuna':
        from src.trainers import OptunaTrainer
        return OptunaTrainer(args, logger)

    elif args.mode == 'full':
        from src.trainers import FullPipelineTrainer
        return FullPipelineTrainer(args, logger)

    else:
        raise ValueError(f"지원하지 않는 모드: {args.mode}")


# ==================== 메인 함수 ====================
def main():
    """메인 실행 함수"""
    # 인자 파싱
    args = parse_arguments()

    print("=" * 60)
    print("🚀 NLP 대화 요약 학습 시작")
    print(f"📋 실행 모드: {args.mode}")
    print(f"🤖 모델: {', '.join(args.models)}")
    print(f"📁 실험명: {args.experiment_name or '(자동 생성)'}")
    print("=" * 60)

    # 환경 설정
    logger = setup_environment(args)

    try:
        # Trainer 생성
        trainer = get_trainer(args, logger)

        # 학습 실행
        logger.write(f"\n📊 {args.mode.upper()} 모드 실행 중...")
        results = trainer.train()

        # 결과 저장
        trainer.save_results(results)

        # 시각화 (옵션)
        if args.save_visualizations:
            logger.write("\n📈 시각화 생성 중...")
            try:
                from src.utils.visualizations import create_training_visualizations
                create_training_visualizations(
                    results=results,
                    output_dir=args.output_dir
                )
                logger.write("  ✅ 시각화 저장 완료")
            except ImportError:
                logger.write("  ⚠️ 시각화 모듈 없음 (추후 구현 예정)")
            except Exception as e:
                logger.write(f"  ⚠️ 시각화 생성 실패: {e}")

        print("\n" + "=" * 60)
        print("✅ 학습 완료!")
        print(f"📁 결과 저장: {args.output_dir}")
        print("=" * 60)

    except Exception as e:
        logger.write(f"\n❌ 오류 발생: {e}", print_error=True)
        raise

    finally:
        # 정리
        logger.stop_redirect()
        logger.close()


# ==================== 실행부 ==================== #
if __name__ == "__main__":
    main()
