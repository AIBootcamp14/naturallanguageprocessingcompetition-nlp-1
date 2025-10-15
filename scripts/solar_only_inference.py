#!/usr/bin/env python3
# ==================== Solar API 독립 추론 스크립트 ==================== #
"""
Solar API만 사용하는 독립 추론 스크립트

K-Fold 앙상블이 이미 완료된 CSV 파일을 입력으로 받아서
Solar API만 적용하여 새로운 제출 파일 생성

사용법:
    # 기본 사용
    python scripts/solar_only_inference.py \
        --input submissions/20251015/kfold_ensemble.csv \
        --test_data data/raw/test.csv

    # K-Fold 방식 3회 샘플링
    python scripts/solar_only_inference.py \
        --input submissions/20251015/kfold_ensemble.csv \
        --test_data data/raw/test.csv \
        --use_voting \
        --n_samples 3

    # 5회 샘플링
    python scripts/solar_only_inference.py \
        --input submissions/20251015/kfold_ensemble.csv \
        --test_data data/raw/test.csv \
        --use_voting \
        --n_samples 5
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 서드파티 라이브러리 ---------------------- #
import pandas as pd

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.logging.logger import Logger
from src.utils.core.common import now, ensure_dir


# ==================== 메인 함수 ==================== #
def main():
    # -------------- 인자 파싱 -------------- #
    parser = argparse.ArgumentParser(description="Solar API 독립 추론 스크립트")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 CSV 파일 (K-Fold 앙상블 결과)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/raw/test.csv",
        help="원본 테스트 데이터 (대화 추출용)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 파일 경로 (미지정 시 자동 생성)"
    )
    parser.add_argument(
        "--solar_model",
        type=str,
        default="solar-1-mini-chat",
        help="Solar 모델 선택"
    )
    parser.add_argument(
        "--solar_api_key",
        type=str,
        default=None,
        help="Solar API 키 (환경변수 SOLAR_API_KEY 사용 가능)"
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
        "--use_voting",
        action="store_true",
        help="K-Fold 방식 다중 샘플링 사용"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="Voting 사용 시 샘플링 횟수"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Solar API 생성 온도"
    )

    args = parser.parse_args()

    # -------------- 출력 디렉토리 설정 -------------- #
    timestamp = now('%Y%m%d_%H%M%S')

    # 옵션 태그 생성
    options = ["solar_only"]
    if args.use_voting:
        options.append(f"voting{args.n_samples}")

    # 폴더명 생성
    folder_name = "_".join([timestamp, "inference"] + options)
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
        logger.write(f"Solar API 독립 추론 시작")
        logger.write("=" * 60)

        # -------------- 1. 입력 파일 로드 -------------- #
        logger.write(f"\n[1/3] 입력 파일 로딩: {args.input}")
        input_df = pd.read_csv(args.input)
        logger.write(f"  ✅ 입력 샘플: {len(input_df)}개")

        # -------------- 2. 테스트 데이터 로드 (대화 추출) -------------- #
        logger.write(f"\n[2/3] 테스트 데이터 로딩: {args.test_data}")
        test_df = pd.read_csv(args.test_data)
        logger.write(f"  ✅ 테스트 샘플: {len(test_df)}개")

        # fname 순서 매칭
        dialogues = []
        for fname in input_df['fname']:
            dialogue = test_df[test_df['fname'] == fname]['dialogue'].values[0]
            dialogues.append(dialogue)

        logger.write(f"  ✅ 대화 추출 완료: {len(dialogues)}개")

        # -------------- 3. Solar API 실행 -------------- #
        logger.write("\n[3/3] 🌞 Solar API 추론 시작")
        logger.write(f"  - 모델: {args.solar_model}")
        logger.write(f"  - Temperature: {args.temperature}")
        logger.write(f"  - 배치 크기: {args.solar_batch_size}")
        logger.write(f"  - 대기 시간: {args.solar_delay}초")
        if args.use_voting:
            logger.write(f"  - 🔄 K-Fold 방식 샘플링: {args.n_samples}회")

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
            logger.write(f"\n  Solar API 배치 요약 생성 중...")
            solar_summaries = solar_api.summarize_batch(
                dialogues=dialogues,
                batch_size=args.solar_batch_size,
                delay=args.solar_delay,
                use_voting=args.use_voting,
                n_samples=args.n_samples
            )

            logger.write("✅ Solar API 추론 완료")

            # 결과 DataFrame 생성
            submission_df = input_df[['fname']].copy()
            submission_df['summary'] = solar_summaries

            # -------------- 4. 파일 저장 -------------- #
            logger.write("\n제출 파일 저장 중...")

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

            # -------------- 5. 결과 출력 -------------- #
            logger.write("\n" + "=" * 60)
            logger.write("📊 Solar API 독립 추론 완료!")
            logger.write("=" * 60)

            logger.write(f"\n📈 결과 통계:")
            logger.write(f"  - 총 샘플 수: {len(submission_df)}")
            logger.write(f"  - 평균 요약 길이: {sum(len(s) for s in solar_summaries) / len(solar_summaries):.1f}자")

            # 샘플 출력
            logger.write("\n📝 샘플 예측 결과 (처음 3개):")
            for idx, row in submission_df.head(3).iterrows():
                logger.write(f"  [{row['fname']}]: {row['summary'][:80]}...")

            logger.write("\n" + "=" * 60)
            logger.write("🎉 Solar API 독립 추론 완료!")
            logger.write(f"📁 제출 파일: {args.output}")
            logger.write("=" * 60)

        except ImportError as e:
            logger.write(f"❌ Solar API 모듈 임포트 실패: {e}")
            raise
        except Exception as e:
            logger.write(f"❌ Solar API 추론 실패: {e}")
            raise

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
