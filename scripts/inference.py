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

        submission_df = predictor.create_submission(
            test_df=test_df,
            output_path=args.output,
            batch_size=args.batch_size,
            show_progress=True,
            **generation_kwargs  # 생성 파라미터 오버라이드
        )

        # 전역 submissions 폴더에도 저장
        submission_df.to_csv(global_submission_path, index=False, encoding='utf-8')
        logger.write(f"  ✅ 제출 파일 추가 저장: {global_submission_path}")

        # -------------- 5. 결과 출력 -------------- #
        logger.write("\n[5/5] 추론 완료!")
        logger.write(f"  ✅ 제출 파일 생성 (1): {args.output}")
        logger.write(f"  ✅ 제출 파일 생성 (2): {global_submission_path}")
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
