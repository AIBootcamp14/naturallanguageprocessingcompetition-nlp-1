#!/usr/bin/env python3
# ==================== K-Fold 앙상블 추론 스크립트 ==================== #
"""
K-Fold 모델 앙상블 추론 스크립트

사용법:
    python scripts/kfold_ensemble_inference.py \
        --experiment_dir experiments/20251014/20251014_183206_kobart_ultimate_kfold \
        --test_data data/raw/test.csv \
        --ensemble_method soft_voting \
        --use_pretrained_correction \
        --output submissions/kfold_ensemble.csv
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import pickle

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 파일에서 환경 변수 로드
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드 성공: {env_path}")
    else:
        print(f"⚠️  .env 파일 없음: {env_path}")
except ImportError:
    print("⚠️  python-dotenv 미설치 - 환경 변수 수동 설정 필요")

# ---------------------- 서드파티 라이브러리 ---------------------- #
import warnings
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

# Transformers 경고 메시지 필터링
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*num_labels.*id2label.*")

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.config import load_config
from src.inference import create_predictor
from src.logging.logger import Logger
from src.utils.core.common import now, ensure_dir
from src.utils.gpu_optimization.team_gpu_check import get_gpu_info, check_gpu_tier


# ==================== 체크포인트 관리 함수 ==================== #
def save_inference_checkpoint(checkpoint_dir, stage, data, logger=None):
    """
    추론 체크포인트 저장

    Args:
        checkpoint_dir: 체크포인트 디렉토리
        stage: 단계 이름 ('kfold', 'hf_correction', 'solar_api')
        data: 저장할 데이터 (dict)
        logger: Logger 인스턴스
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / f"{stage}_checkpoint.pkl"

    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)

        if logger:
            logger.write(f"💾 체크포인트 저장: {checkpoint_file}")
    except Exception as e:
        if logger:
            logger.write(f"⚠️  체크포인트 저장 실패: {e}")


def load_inference_checkpoint(checkpoint_dir, stage, logger=None):
    """
    추론 체크포인트 로드

    Args:
        checkpoint_dir: 체크포인트 디렉토리
        stage: 단계 이름 ('kfold', 'hf_correction', 'solar_api')
        logger: Logger 인스턴스

    Returns:
        dict: 체크포인트 데이터, 없으면 None
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_file = checkpoint_dir / f"{stage}_checkpoint.pkl"

    if not checkpoint_file.exists():
        return None

    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)

        if logger:
            logger.write(f"📂 체크포인트 로드: {checkpoint_file}")

        return data
    except Exception as e:
        if logger:
            logger.write(f"⚠️  체크포인트 로드 실패: {e}")
        return None


def remove_inference_checkpoint(checkpoint_dir, stage, logger=None):
    """체크포인트 파일 삭제"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_file = checkpoint_dir / f"{stage}_checkpoint.pkl"

    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            if logger:
                logger.write(f"🗑️  체크포인트 삭제: {checkpoint_file}")
        except Exception as e:
            if logger:
                logger.write(f"⚠️  체크포인트 삭제 실패: {e}")


# ==================== K-Fold 앙상블 클래스 ==================== #
class KFoldEnsemblePredictor:
    """K-Fold 모델 앙상블 예측기"""

    def __init__(self, fold_model_dirs, ensemble_method='soft_voting', logger=None):
        """
        Args:
            fold_model_dirs: Fold별 모델 디렉토리 리스트
            ensemble_method: 앙상블 방법 ('soft_voting', 'hard_voting', 'averaging')
            logger: Logger 인스턴스
        """
        self.fold_model_dirs = fold_model_dirs
        self.ensemble_method = ensemble_method
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 각 Fold 모델 로드
        self.models = []
        self.tokenizers = []
        self._load_all_fold_models()

    def _log(self, msg):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def _load_all_fold_models(self):
        """모든 Fold 모델 로드"""
        self._log(f"\n🔄 {len(self.fold_model_dirs)}개 Fold 모델 로딩 중...")

        for i, model_dir in enumerate(self.fold_model_dirs):
            self._log(f"  [Fold {i+1}/{len(self.fold_model_dirs)}] 로딩: {model_dir}")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = model.to(self.device)
                model.eval()

                self.models.append(model)
                self.tokenizers.append(tokenizer)
                self._log(f"    ✅ 완료")
            except Exception as e:
                self._log(f"    ❌ 실패: {e}")
                raise

        self._log(f"✅ 전체 {len(self.models)}개 모델 로드 완료\n")

    def predict_batch(
        self,
        dialogues,
        batch_size=16,
        show_progress=True,
        **generation_kwargs
    ):
        """
        배치 앙상블 예측

        Args:
            dialogues: 입력 대화 리스트
            batch_size: 배치 크기
            show_progress: 진행바 표시 여부
            **generation_kwargs: 생성 파라미터

        Returns:
            List[str]: 앙상블된 요약 리스트
        """
        self._log(f"🔮 K-Fold 앙상블 추론 시작 ({self.ensemble_method})")
        self._log(f"  - 샘플 수: {len(dialogues)}")
        self._log(f"  - Fold 수: {len(self.models)}")
        self._log(f"  - 배치 크기: {batch_size}")

        # 각 Fold별로 예측 수행
        all_fold_summaries = []

        for fold_idx, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            self._log(f"\n[Fold {fold_idx+1}/{len(self.models)}] 예측 중...")

            # Predictor 생성
            predictor = create_predictor(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                logger=None  # 너무 많은 로그 방지
            )

            # 배치 예측
            fold_summaries = predictor.predict_batch(
                dialogues=dialogues,
                batch_size=batch_size,
                show_progress=show_progress,
                **generation_kwargs
            )

            all_fold_summaries.append(fold_summaries)
            self._log(f"  ✅ Fold {fold_idx+1} 예측 완료 ({len(fold_summaries)}개)")

        # 앙상블 수행
        self._log(f"\n🔄 앙상블 수행 중 ({self.ensemble_method})...")
        ensemble_summaries = self._ensemble(all_fold_summaries)
        self._log(f"  ✅ 앙상블 완료\n")

        return ensemble_summaries

    def _ensemble(self, all_fold_summaries):
        """
        앙상블 수행

        Args:
            all_fold_summaries: Fold별 요약 리스트의 리스트
                [[fold0_summary0, fold0_summary1, ...],
                 [fold1_summary0, fold1_summary1, ...],
                 ...]

        Returns:
            List[str]: 앙상블된 요약 리스트
        """
        n_samples = len(all_fold_summaries[0])
        ensemble_summaries = []

        for i in range(n_samples):
            # i번째 샘플의 모든 Fold 요약 수집
            sample_summaries = [fold_summaries[i] for fold_summaries in all_fold_summaries]

            if self.ensemble_method == 'soft_voting':
                # Soft Voting: 가장 긴 요약 선택 (정보량 최대화)
                ensemble_summary = max(sample_summaries, key=len)

            elif self.ensemble_method == 'hard_voting':
                # Hard Voting: 가장 빈번한 요약 선택
                from collections import Counter
                counter = Counter(sample_summaries)
                ensemble_summary = counter.most_common(1)[0][0]

            elif self.ensemble_method == 'averaging':
                # Averaging: 중간 길이 요약 선택
                sample_summaries_sorted = sorted(sample_summaries, key=len)
                median_idx = len(sample_summaries_sorted) // 2
                ensemble_summary = sample_summaries_sorted[median_idx]

            else:
                # 기본값: 첫 번째 Fold 사용
                ensemble_summary = sample_summaries[0]

            ensemble_summaries.append(ensemble_summary)

        return ensemble_summaries


# ==================== 메인 함수 ==================== #
def main():
    # -------------- 인자 파싱 -------------- #
    parser = argparse.ArgumentParser(description="K-Fold 앙상블 추론 스크립트")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="K-Fold 학습 실험 디렉토리 (예: experiments/20251014/20251014_183206_kobart_ultimate_kfold)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/raw/test.csv",
        help="테스트 데이터 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="제출 파일 출력 경로 (미지정 시 자동 생성)"
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        default="soft_voting",
        choices=["soft_voting", "hard_voting", "averaging"],
        help="앙상블 방법"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="추론 배치 크기"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam search 빔 개수"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="생성할 최대 토큰 수"
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=None,
        help="생성할 최소 토큰 수"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="반복 억제 강도"
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        help="반복 금지 n-gram 크기"
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=None,
        help="길이 페널티"
    )
    parser.add_argument(
        "--use_pretrained_correction",
        action="store_true",
        help="HuggingFace 사전학습 모델 보정 사용"
    )
    parser.add_argument(
        "--correction_models",
        type=str,
        nargs="+",
        default=["gogamza/kobart-base-v2", "digit82/kobart-summarization"],
        help="보정에 사용할 HuggingFace 모델 리스트"
    )
    parser.add_argument(
        "--correction_strategy",
        type=str,
        default="quality_based",
        choices=["quality_based", "threshold", "voting", "weighted"],
        help="보정 전략"
    )
    parser.add_argument(
        "--correction_threshold",
        type=float,
        default=0.3,
        help="품질 임계값"
    )
    parser.add_argument(
        "--use_solar_api",
        action="store_true",
        help="Solar API 앙상블 사용"
    )
    parser.add_argument(
        "--solar_api_key",
        type=str,
        default=None,
        help="Solar API 키 (환경변수 SOLAR_API_KEY 사용 가능)"
    )
    parser.add_argument(
        "--solar_model",
        type=str,
        default="solar-1-mini-chat",
        help="Solar 모델 선택"
    )
    parser.add_argument(
        "--solar_temperature",
        type=float,
        default=0.2,
        help="Solar API 생성 온도"
    )
    parser.add_argument(
        "--solar_batch_size",
        type=int,
        default=10,
        help="Solar API 배치 크기"
    )
    parser.add_argument(
        "--solar_delay",
        type=float,
        default=1.0,
        help="Solar API 배치 간 대기 시간 (초)"
    )
    parser.add_argument(
        "--solar_use_voting",
        action="store_true",
        help="Solar API K-Fold 방식 다중 샘플링 사용"
    )
    parser.add_argument(
        "--solar_n_samples",
        type=int,
        default=3,
        help="Solar API 샘플링 횟수 (voting 사용 시)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="체크포인트에서 이어서 실행"
    )
    parser.add_argument(
        "--skip_kfold",
        action="store_true",
        help="K-Fold 앙상블 건너뛰기 (체크포인트 필수)"
    )
    parser.add_argument(
        "--kfold_checkpoint",
        type=str,
        default=None,
        help="재사용할 K-Fold 체크포인트 경로 (예: experiments/.../checkpoints/kfold_checkpoint.pkl)"
    )

    args = parser.parse_args()

    # -------------- 출력 디렉토리 설정 -------------- #
    timestamp = now('%Y%m%d_%H%M%S')

    # 옵션 태그 생성
    options = ["kfold", args.ensemble_method]
    if args.batch_size != 16:
        options.append(f"bs{args.batch_size}")
    if args.max_new_tokens != 100:
        options.append(f"maxnew{args.max_new_tokens}")
    if args.use_pretrained_correction:
        options.append("hf")
    if args.use_solar_api:
        options.append("solar")

    # 폴더명 생성
    folder_name = "_".join([timestamp, "inference_kobart"] + options)
    date_folder = datetime.now().strftime("%Y%m%d")
    output_dir = Path(f"experiments/{date_folder}/{folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------- Logger 초기화 -------------- #
    log_path = output_dir / "inference.log"
    logger = Logger(log_path, print_also=True)
    logger.start_redirect()

    # ✅ 실행 명령어 저장
    from src.utils.core.path_resolver import save_command_to_experiment
    save_command_to_experiment(output_dir, verbose=False)

    try:
        logger.write("=" * 60)
        logger.write(f"K-Fold 앙상블 추론 시작")
        logger.write("=" * 60)

        # -------------- GPU 정보 출력 -------------- #
        logger.write("\n[GPU 정보]")
        gpu_info = get_gpu_info()
        for key, value in gpu_info.items():
            logger.write(f"  {key}: {value}")

        gpu_tier = check_gpu_tier()
        logger.write(f"  GPU Tier: {gpu_tier}")

        # -------------- 1. Fold 모델 경로 탐색 -------------- #
        logger.write(f"\n[1/4] Fold 모델 탐색: {args.experiment_dir}")
        experiment_dir = Path(args.experiment_dir)

        fold_model_dirs = []
        for fold_dir in sorted(experiment_dir.glob("fold_*")):
            # kfold/final_model 또는 default/final_model 찾기
            for subdir in ['kfold', 'default']:
                model_path = fold_dir / subdir / 'final_model'
                if model_path.exists():
                    fold_model_dirs.append(str(model_path))
                    logger.write(f"  ✅ {fold_dir.name}: {model_path}")
                    break

        if not fold_model_dirs:
            raise ValueError(f"Fold 모델을 찾을 수 없습니다: {experiment_dir}")

        logger.write(f"\n  📊 발견된 Fold 수: {len(fold_model_dirs)}")

        # -------------- 2. 테스트 데이터 로드 -------------- #
        logger.write(f"\n[2/4] 테스트 데이터 로딩: {args.test_data}")
        test_df = pd.read_csv(args.test_data)
        logger.write(f"  ✅ 테스트 샘플: {len(test_df)}개")

        # 대화 추출
        dialogues = test_df['dialogue'].tolist()

        # 체크포인트 디렉토리 설정
        checkpoint_dir = output_dir / "checkpoints" if args.resume else None

        # -------------- 3. K-Fold 앙상블 예측 -------------- #
        summaries = None
        kfold_checkpoint = None

        # 외부 체크포인트 지정 시 로드
        if args.kfold_checkpoint:
            logger.write(f"\n[3/6] 📂 외부 K-Fold 체크포인트 로드: {args.kfold_checkpoint}")
            try:
                with open(args.kfold_checkpoint, 'rb') as f:
                    kfold_checkpoint = pickle.load(f)
                logger.write(f"  ✅ 로드 성공: {len(kfold_checkpoint['summaries'])}개 요약")
                summaries = kfold_checkpoint['summaries']
            except Exception as e:
                logger.write(f"  ❌ 로드 실패: {e}")
                raise

        # 기존 체크포인트 확인
        elif checkpoint_dir:
            kfold_checkpoint = load_inference_checkpoint(checkpoint_dir, 'kfold', logger)

        if kfold_checkpoint and not args.kfold_checkpoint:
            logger.write(f"\n[3/6] ✅ K-Fold 앙상블 체크포인트에서 복원")
            logger.write(f"  - 복원된 요약 수: {len(kfold_checkpoint['summaries'])}")
            summaries = kfold_checkpoint['summaries']
        elif args.skip_kfold and summaries is not None:
            logger.write(f"\n[3/6] ⏭️  K-Fold 앙상블 건너뛰기 (외부 체크포인트 사용)")
        elif args.skip_kfold:
            raise ValueError("--skip_kfold 사용 시 --kfold_checkpoint 또는 --resume이 필요합니다")
        elif summaries is None:
            logger.write(f"\n[3/6] K-Fold 앙상블 추론 실행...")
            logger.write(f"  - 앙상블 방법: {args.ensemble_method}")
            logger.write(f"  - 배치 크기: {args.batch_size}")

            # K-Fold Ensemble Predictor 생성
            ensemble_predictor = KFoldEnsemblePredictor(
                fold_model_dirs=fold_model_dirs,
                ensemble_method=args.ensemble_method,
                logger=logger
            )

            # 생성 파라미터 준비
            generation_kwargs = {'num_beams': args.num_beams}
            if args.max_new_tokens is not None:
                generation_kwargs['max_new_tokens'] = args.max_new_tokens
            if args.min_new_tokens is not None:
                generation_kwargs['min_new_tokens'] = args.min_new_tokens
            if args.repetition_penalty is not None:
                generation_kwargs['repetition_penalty'] = args.repetition_penalty
            if args.no_repeat_ngram_size is not None:
                generation_kwargs['no_repeat_ngram_size'] = args.no_repeat_ngram_size
            if args.length_penalty is not None:
                generation_kwargs['length_penalty'] = args.length_penalty

            # 앙상블 예측 수행
            summaries = ensemble_predictor.predict_batch(
                dialogues=dialogues,
                batch_size=args.batch_size,
                show_progress=True,
                **generation_kwargs
            )

            # 체크포인트 저장
            if checkpoint_dir:
                save_inference_checkpoint(
                    checkpoint_dir,
                    'kfold',
                    {
                        'summaries': summaries,
                        'ensemble_method': args.ensemble_method,
                        'generation_kwargs': generation_kwargs
                    },
                    logger
                )

                # CSV 체크포인트도 저장 (이어서 실행 가능하도록)
                kfold_csv_path = checkpoint_dir / "kfold_summaries.csv"
                pd.DataFrame({
                    'fname': test_df['fname'],
                    'summary': summaries
                }).to_csv(kfold_csv_path, index=False, encoding='utf-8')
                logger.write(f"💾 K-Fold CSV 체크포인트 저장: {kfold_csv_path}")

        # -------------- 4. HuggingFace 보정 (선택적) -------------- #
        hf_checkpoint = None
        if checkpoint_dir and args.use_pretrained_correction:
            hf_checkpoint = load_inference_checkpoint(checkpoint_dir, 'hf_correction', logger)

        if hf_checkpoint:
            logger.write(f"\n[4/6] ✅ HuggingFace 보정 체크포인트에서 복원")
            logger.write(f"  - 복원된 요약 수: {len(hf_checkpoint['summaries'])}")
            summaries = hf_checkpoint['summaries']
        elif args.use_pretrained_correction:
            logger.write("\n[4/6] 🔧 HuggingFace 사전학습 모델 보정 시작")
            logger.write(f"  - 보정 모델: {', '.join(args.correction_models)}")
            logger.write(f"  - 보정 전략: {args.correction_strategy}")
            logger.write(f"  - 품질 임계값: {args.correction_threshold}")

            try:
                from src.correction.pretrained_corrector import PretrainedCorrector

                # 보정기 생성
                corrector = PretrainedCorrector(
                    model_names=args.correction_models,
                    correction_strategy=args.correction_strategy,
                    quality_threshold=args.correction_threshold,
                    logger=logger
                )

                # 보정 수행
                summaries = corrector.correct_batch(
                    dialogues=dialogues,
                    candidate_summaries=summaries,
                    batch_size=args.batch_size,
                    **generation_kwargs
                )

                logger.write("✅ HuggingFace 보정 완료")

                # 체크포인트 저장
                if checkpoint_dir:
                    save_inference_checkpoint(
                        checkpoint_dir,
                        'hf_correction',
                        {
                            'summaries': summaries,
                            'correction_models': args.correction_models,
                            'correction_strategy': args.correction_strategy
                        },
                        logger
                    )

                    # CSV 체크포인트도 저장
                    hf_csv_path = checkpoint_dir / "hf_correction_summaries.csv"
                    pd.DataFrame({
                        'fname': test_df['fname'],
                        'summary': summaries
                    }).to_csv(hf_csv_path, index=False, encoding='utf-8')
                    logger.write(f"💾 HuggingFace CSV 체크포인트 저장: {hf_csv_path}")
            except Exception as e:
                logger.write(f"❌ HuggingFace 보정 실패: {e}")
                logger.write("  ⚠️  보정 없이 진행")
        else:
            if not args.use_pretrained_correction:
                logger.write("\n[4/6] ⏭️  HuggingFace 보정 건너뛰기 (비활성화)")

        # -------------- 5. Solar API 앙상블 (선택적) -------------- #
        solar_checkpoint = None
        if checkpoint_dir and args.use_solar_api:
            solar_checkpoint = load_inference_checkpoint(checkpoint_dir, 'solar_api', logger)

        if solar_checkpoint:
            logger.write(f"\n[5/6] ✅ Solar API 체크포인트에서 복원")
            logger.write(f"  - 복원된 요약 수: {len(solar_checkpoint['summaries'])}")
            summaries = solar_checkpoint['summaries']
        elif args.use_solar_api:
            logger.write("\n[5/6] 🌞 Solar API 앙상블 시작")
            logger.write(f"  - 모델: {args.solar_model}")
            logger.write(f"  - Temperature: {args.solar_temperature}")
            logger.write(f"  - 배치 크기: {args.solar_batch_size}")
            logger.write(f"  - 대기 시간: {args.solar_delay}초")

            try:
                from src.api.solar_api import create_solar_api

                # Solar API 클라이언트 생성
                solar_api = create_solar_api(
                    api_key=args.solar_api_key,
                    token_limit=512,
                    cache_dir=str(output_dir / "cache" / "solar"),
                    logger=logger
                )

                # Solar API로 요약 생성
                if args.solar_use_voting:
                    logger.write(f"\n  Solar API 배치 요약 생성 중 (🔄 K-Fold 방식 {args.solar_n_samples}회 샘플링)...")
                else:
                    logger.write(f"\n  Solar API 배치 요약 생성 중...")

                solar_summaries = solar_api.summarize_batch(
                    dialogues=dialogues,
                    batch_size=args.solar_batch_size,
                    delay=args.solar_delay,
                    use_voting=args.solar_use_voting,
                    n_samples=args.solar_n_samples
                )

                # KoBART 요약과 Solar 요약 앙상블 (가중 평균)
                logger.write(f"\n  KoBART와 Solar 앙상블 수행 중...")
                ensemble_summaries = []
                for kobart_summary, solar_summary in zip(summaries, solar_summaries):
                    # 간단한 앙상블 전략: Solar 요약 우선 사용, 실패 시 KoBART 사용
                    if solar_summary and len(solar_summary.strip()) > 10:
                        ensemble_summaries.append(solar_summary)
                    else:
                        ensemble_summaries.append(kobart_summary)

                summaries = ensemble_summaries
                logger.write("✅ Solar API 앙상블 완료")

                # 체크포인트 저장
                if checkpoint_dir:
                    save_inference_checkpoint(
                        checkpoint_dir,
                        'solar_api',
                        {
                            'summaries': summaries,
                            'solar_model': args.solar_model,
                            'solar_temperature': args.solar_temperature
                        },
                        logger
                    )

                    # CSV 체크포인트도 저장
                    solar_csv_path = checkpoint_dir / "solar_api_summaries.csv"
                    pd.DataFrame({
                        'fname': test_df['fname'],
                        'summary': summaries
                    }).to_csv(solar_csv_path, index=False, encoding='utf-8')
                    logger.write(f"💾 Solar API CSV 체크포인트 저장: {solar_csv_path}")

            except ImportError as e:
                logger.write(f"❌ Solar API 모듈 임포트 실패: {e}")
                logger.write("  ⚠️  Solar API 없이 진행")
            except Exception as e:
                logger.write(f"❌ Solar API 앙상블 실패: {e}")
                logger.write("  ⚠️  Solar API 없이 진행")
        else:
            if not args.use_solar_api:
                logger.write("\n[5/6] ⏭️  Solar API 건너뛰기 (비활성화)")

        # 제출 DataFrame 생성
        submission_df = test_df[['fname']].copy()
        submission_df['summary'] = summaries

        # -------------- 6. 파일 저장 -------------- #
        logger.write("\n[6/6] 제출 파일 저장 중...")

        # 출력 경로 자동 생성
        if args.output is None:
            submission_dir = output_dir / "submission"
            submission_dir.mkdir(parents=True, exist_ok=True)
            args.output = str(submission_dir / f"{folder_name}.csv")

        # 출력 경로 디렉토리 생성
        ensure_dir(Path(args.output).parent)

        # 1) 실험 폴더에 저장
        submission_df.to_csv(args.output, index=False, encoding='utf-8')
        logger.write(f"  ✅ 제출 파일 생성 (1): {args.output}")

        # 2) 전역 submissions 폴더에도 저장
        global_submission_dir = Path('submissions') / date_folder
        global_submission_dir.mkdir(parents=True, exist_ok=True)
        global_submission_path = global_submission_dir / f"{folder_name}.csv"
        submission_df.to_csv(global_submission_path, index=False, encoding='utf-8')
        logger.write(f"  ✅ 제출 파일 생성 (2): {global_submission_path}")

        # -------------- 7. 결과 출력 -------------- #
        logger.write("\n" + "=" * 60)
        logger.write("📊 추론 파이프라인 완료!")
        logger.write("=" * 60)

        logger.write(f"\n✅ 실행된 단계:")
        logger.write(f"  1. K-Fold 앙상블 ({len(fold_model_dirs)}개 모델, {args.ensemble_method})")
        if args.use_pretrained_correction:
            logger.write(f"  2. HuggingFace 보정 ({', '.join(args.correction_models)})")
        if args.use_solar_api:
            logger.write(f"  3. Solar API 앙상블 ({args.solar_model})")
        logger.write(f"  4. 후처리 (Predictor 내장)")
        logger.write(f"  5. 제출 파일 생성")

        logger.write(f"\n📈 결과 통계:")
        logger.write(f"  - 총 샘플 수: {len(submission_df)}")
        logger.write(f"  - 평균 요약 길이: {sum(len(s) for s in summaries) / len(summaries):.1f}자")

        # 샘플 출력
        logger.write("\n📝 샘플 예측 결과 (처음 3개):")
        for idx, row in submission_df.head(3).iterrows():
            logger.write(f"  [{row['fname']}]: {row['summary'][:80]}...")

        logger.write("\n" + "=" * 60)
        logger.write("🎉 K-Fold 앙상블 추론 완료!")
        logger.write(f"📁 제출 파일: {args.output}")
        logger.write("=" * 60)

    except Exception as e:
        logger.write(f"\n❌ 추론 오류 발생: {e}", print_error=True)
        logger.write_last_progress()
        raise

    finally:
        # Logger 정리
        logger.stop_redirect()
        logger.close()


# ==================== 실행부 ==================== #
if __name__ == "__main__":
    main()
