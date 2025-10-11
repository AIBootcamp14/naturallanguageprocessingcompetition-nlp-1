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
from src.utils.core.common import create_log_path
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
        required=True,
        help="제출 파일 출력 경로 (예: submissions/submission.csv)"
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
        "--experiment",
        type=str,
        default="baseline_kobart",
        help="실험 Config 이름 (생성 파라미터 로드용)"
    )
    args = parser.parse_args()

    # -------------- Logger 초기화 -------------- #
    model_name = Path(args.model).name
    log_path = create_log_path("outputs/logs", f"inference_{model_name}")
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

        # -------------- 2. 모델 및 토크나이저 로드 -------------- #
        logger.write(f"\n[2/5] 모델 로딩: {args.model}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        logger.write(f"  ✅ 모델 로드 완료")
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
            logger=logger
        )

        # 제출 파일 생성
        submission_df = predictor.create_submission(
            test_df=test_df,
            output_path=args.output,
            batch_size=args.batch_size,
            show_progress=True,
            num_beams=args.num_beams  # 오버라이드
        )

        # -------------- 5. 결과 출력 -------------- #
        logger.write("\n[5/5] 추론 완료!")
        logger.write(f"  ✅ 제출 파일 생성: {args.output}")
        logger.write(f"  샘플 수: {len(submission_df)}")

        # 샘플 출력
        logger.write("\n  샘플 예측 결과 (처음 3개):")
        for idx, row in submission_df.head(3).iterrows():
            logger.write(f"    [{row['fname']}]: {row['summary'][:50]}...")

        logger.write("\n" + "=" * 60)
        logger.write("🎉 추론 완료!")
        logger.write("=" * 60)

    finally:
        # Logger 정리
        logger.stop_redirect()
        logger.close()


# ==================== 실행부 ==================== #
if __name__ == "__main__":
    main()
