# ==================== 추론 실행 스크립트 ==================== #
"""
추론 및 제출 파일 생성 스크립트

사용법:
    python scripts/inference.py --model outputs/baseline_kobart/final_model --output submissions/submission.csv
    python scripts/inference.py --model outputs/baseline_kobart/checkpoint-1000 --output submissions/test.csv
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.config import load_config
from src.inference import create_predictor
from src.logging.logger import Logger
from src.utils.core.common import now
from src.utils.gpu_optimization.team_gpu_check import get_gpu_info, check_gpu_tier


# ==================== 메인 함수 ==================== #
def main():
    # -------------- 인자 파싱 -------------- #
    parser = argparse.ArgumentParser(description="추론 실행 스크립트")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="모델 체크포인트 경로 (예: outputs/baseline_kobart/final_model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="제출 파일 출력 경로 (미지정 시 자동 생성: {date}_{time}_{mode}_{models}_{options}.csv)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/raw/test.csv",
        help="테스트 데이터 경로"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        default=None,
        help="생성할 최대 토큰 수 (None: config 파일 값 사용, 권장: 200)"
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=None,
        help="생성할 최소 토큰 수 (None: config 파일 값 사용, 권장: 30)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="반복 억제 강도 (None: config 파일 값 사용, 권장: 1.5~2.0)"
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        help="반복 금지 n-gram 크기 (None: config 파일 값 사용, 권장: 3)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline_kobart",
        help="실험 Config 이름 (생성 파라미터 로드용)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="출력 디렉토리 (None: 자동 생성 experiments/날짜/추론폴더)"
    )

    # ==================== Solar API 옵션 (PRD 09) ====================
    parser.add_argument(
        "--use_solar_api",
        action="store_true",
        help="Solar API 앙상블 사용"
    )
    parser.add_argument(
        "--solar_weight",
        type=float,
        default=0.3,
        help="Solar API 가중치 (0.0~1.0, 기본값: 0.3)"
    )
    parser.add_argument(
        "--kobart_weight",
        type=float,
        default=0.7,
        help="KoBART 가중치 (0.0~1.0, 기본값: 0.7)"
    )
    parser.add_argument(
        "--ensemble_strategy",
        type=str,
        default="weighted_avg",
        choices=["weighted_avg", "quality_based", "voting"],
        help="앙상블 전략"
    )

    # ==================== HuggingFace 보정 옵션 (PRD 04, 12) ====================
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
        help="보정 전략 (quality_based 추천)"
    )
    parser.add_argument(
        "--correction_threshold",
        type=float,
        default=0.3,
        help="품질 임계값 (0.0~1.0)"
    )

    args = parser.parse_args()

    # -------------- 출력 디렉토리 설정 -------------- #
    timestamp = now('%Y%m%d_%H%M%S')

    # 모델명 추출
    if 'kobart' in args.model.lower():
        model_name_short = 'kobart'
    elif 'solar' in args.model.lower():
        model_name_short = 'solar'
    elif 'pegasus' in args.model.lower():
        model_name_short = 'pegasus'
    elif 'bart' in args.model.lower():
        model_name_short = 'bart'
    else:
        model_name_short = Path(args.model).name

    # 옵션 태그 생성
    options = []
    if args.batch_size != 32:
        options.append(f"bs{args.batch_size}")
    if args.num_beams != 4:
        options.append(f"beam{args.num_beams}")
    if args.max_new_tokens is not None:
        options.append(f"maxnew{args.max_new_tokens}")
    if args.min_new_tokens is not None:
        options.append(f"minnew{args.min_new_tokens}")
    if args.repetition_penalty is not None:
        options.append(f"rep{args.repetition_penalty}")
    if args.no_repeat_ngram_size is not None:
        options.append(f"ngram{args.no_repeat_ngram_size}")
    if args.use_solar_api:
        options.append("solar")
    if args.use_pretrained_correction:
        options.append("hf")

    # 출력 디렉토리 자동 생성 (지정되지 않은 경우)
    if args.output_dir is None:
        from datetime import datetime
        date_folder = datetime.now().strftime("%Y%m%d")

        # 폴더명 생성
        folder_parts = [timestamp, "inference", model_name_short]
        if options:
            folder_parts.extend(options)
        folder_name = "_".join(folder_parts)

        output_dir = Path(f"experiments/{date_folder}/{folder_name}")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------- Logger 초기화 -------------- #
    log_path = output_dir / "inference.log"
    logger = Logger(log_path, print_also=True)
    logger.start_redirect()

    try:
        logger.write("=" * 60)
        logger.write(f"추론 시작")
        logger.write("=" * 60)

        # -------------- GPU 정보 출력 -------------- #
        logger.write("\n[GPU 정보]")
        gpu_info = get_gpu_info()
        for key, value in gpu_info.items():
            logger.write(f"  {key}: {value}")

        gpu_tier = check_gpu_tier()
        logger.write(f"  GPU Tier: {gpu_tier}")

        # -------------- 1. Config 로드 (선택적) -------------- #
        logger.write("\n[1/5] Config 로딩...")
        try:
            config = load_config(args.experiment)
            logger.write(f"  ✅ Config 로드 완료: {args.experiment}")
        except:
            logger.write("  ⚠️ Config 로드 실패, 기본 설정 사용")
            config = None

        # -------------- 출력 파일명 자동 생성 -------------- #
        folder_name = output_dir.name  # 예: 20251013_101219_inference_kobart

        if args.output is None:
            # 1. experiments/{날짜}/{실행폴더}/submissions/{실행폴더명}.csv 저장
            submission_dir = output_dir / "submissions"
            submission_dir.mkdir(parents=True, exist_ok=True)
            args.output = str(submission_dir / f"{folder_name}.csv")

            logger.write(f"  📝 자동 생성된 제출 파일 경로: {args.output}")

        # 2. submissions/{날짜}/{실행폴더명}.csv에도 저장 (추가)
        from datetime import datetime
        date_str = datetime.now().strftime('%Y%m%d')
        global_submission_dir = Path('submissions') / date_str
        global_submission_dir.mkdir(parents=True, exist_ok=True)
        global_submission_path = global_submission_dir / f"{folder_name}.csv"

        # -------------- 2. 모델 및 토크나이저 로드 -------------- #
        logger.write(f"\n[2/5] 모델 로딩: {args.model}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        # GPU 사용 가능하면 모델을 GPU로 이동
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        logger.write(f"  ✅ 모델 로드 완료")
        logger.write(f"  디바이스: {device}")
        logger.write(f"  모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

        # -------------- 3. 테스트 데이터 로드 -------------- #
        logger.write(f"\n[3/5] 테스트 데이터 로딩: {args.test_data}")
        test_df = pd.read_csv(args.test_data)
        logger.write(f"  ✅ 테스트 샘플: {len(test_df)}개")

        # -------------- 4. Predictor 생성 및 추론 -------------- #
        logger.write("\n[4/5] 추론 실행...")
        predictor = create_predictor(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            logger=logger
        )

        # 제출 파일 생성
        # 생성 파라미터 준비 (None이 아닌 값만 전달)
        generation_kwargs = {'num_beams': args.num_beams}
        if args.max_new_tokens is not None:
            generation_kwargs['max_new_tokens'] = args.max_new_tokens
        if args.min_new_tokens is not None:
            generation_kwargs['min_new_tokens'] = args.min_new_tokens
        if args.repetition_penalty is not None:
            generation_kwargs['repetition_penalty'] = args.repetition_penalty
        if args.no_repeat_ngram_size is not None:
            generation_kwargs['no_repeat_ngram_size'] = args.no_repeat_ngram_size

        # HuggingFace 보정 옵션 추가
        if args.use_pretrained_correction:
            logger.write("\n🔧 HuggingFace 사전학습 모델 보정 활성화")
            logger.write(f"  - 보정 모델: {', '.join(args.correction_models)}")
            logger.write(f"  - 보정 전략: {args.correction_strategy}")
            logger.write(f"  - 품질 임계값: {args.correction_threshold}")

        # Solar API 옵션 확인 (현재 미구현 - config를 통해 설정 필요)
        if args.use_solar_api:
            logger.write("\n⚠️  Solar API 앙상블은 현재 config 파일을 통해서만 지원됩니다")
            logger.write(f"  - --use_solar_api 플래그는 무시됩니다")
            logger.write(f"  - config 파일에서 solar_api 섹션을 설정하세요")

        # 대화 추출
        dialogues = test_df['dialogue'].tolist()

        # 배치 예측 수행 (HF 보정 포함)
        summaries = predictor.predict_batch(
            dialogues=dialogues,
            batch_size=args.batch_size,
            show_progress=True,
            use_pretrained_correction=args.use_pretrained_correction,
            correction_models=args.correction_models if args.use_pretrained_correction else None,
            correction_strategy=args.correction_strategy,
            correction_threshold=args.correction_threshold,
            **generation_kwargs  # 생성 파라미터 오버라이드
        )

        # 제출 DataFrame 생성
        submission_df = test_df[['fname']].copy()
        submission_df['summary'] = summaries

        # -------------- 5. 파일 저장 -------------- #
        logger.write("\n[5/5] 제출 파일 저장 중...")

        # 출력 경로 디렉토리 생성
        from src.utils.core.common import ensure_dir
        ensure_dir(Path(args.output).parent)

        # 1) 실험 폴더에 저장
        submission_df.to_csv(args.output, index=False, encoding='utf-8')
        logger.write(f"  ✅ 제출 파일 생성 (1): {args.output}")

        # 2) 전역 submissions 폴더에도 저장
        submission_df.to_csv(global_submission_path, index=False, encoding='utf-8')
        logger.write(f"  ✅ 제출 파일 생성 (2): {global_submission_path}")

        # -------------- 6. 결과 출력 -------------- #
        logger.write("\n추론 완료!")
        logger.write(f"  샘플 수: {len(submission_df)}")

        # 샘플 출력
        logger.write("\n  샘플 예측 결과 (처음 3개):")
        for idx, row in submission_df.head(3).iterrows():
            logger.write(f"    [{row['fname']}]: {row['summary'][:50]}...")

        logger.write("\n" + "=" * 60)
        logger.write("🎉 추론 완료!")
        logger.write("=" * 60)

    except Exception as e:
        logger.write(f"\n❌ 추론 오류 발생: {e}", print_error=True)
        # 오류 발생 시 마지막 진행률 기록
        logger.write_last_progress()
        raise

    finally:
        # Logger 정리
        logger.stop_redirect()
        logger.close()


# ==================== 실행부 ==================== #
if __name__ == "__main__":
    main()
