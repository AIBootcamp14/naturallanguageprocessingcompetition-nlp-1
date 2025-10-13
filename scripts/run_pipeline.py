# ==================== 전체 파이프라인 실행 스크립트 ==================== #
"""
학습 + 추론 전체 파이프라인 실행

사용법:
    python scripts/run_pipeline.py --experiment baseline_kobart
    python scripts/run_pipeline.py --experiment baseline_kobart --skip_training
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
import argparse
import subprocess
from pathlib import Path

# 프로젝트 루트
project_root = Path(__file__).parent.parent


# ==================== 메인 함수 ==================== #
def main():
    # -------------- 인자 파싱 -------------- #
    parser = argparse.ArgumentParser(description="전체 파이프라인 실행")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="실험 이름"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="학습 건너뛰기 (기존 모델 사용)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="추론에 사용할 모델 경로 (skip_training 시 필수)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submissions/submission.csv",
        help="제출 파일 출력 경로"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"전체 파이프라인 실행: {args.experiment}")
    print("=" * 60)

    # -------------- 1. 학습 단계 -------------- #
    if not args.skip_training:
        print("\n[단계 1/2] 학습 시작...")
        print("-" * 60)

        train_cmd = [
            sys.executable,
            str(project_root / "scripts" / "train.py"),
            "--experiment", args.experiment
        ]

        result = subprocess.run(train_cmd, cwd=project_root)

        if result.returncode != 0:
            print("\n❌ 학습 실패!")
            sys.exit(1)

        print("\n✅ 학습 완료!")

        # 학습된 모델 경로 설정
        model_path = f"outputs/{args.experiment}/final_model"
    else:
        print("\n[단계 1/2] 학습 건너뛰기")
        if args.model_path is None:
            print("❌ --model_path가 필요합니다 (--skip_training 사용 시)")
            sys.exit(1)
        model_path = args.model_path

    # -------------- 2. 추론 단계 -------------- #
    print(f"\n[단계 2/2] 추론 시작...")
    print("-" * 60)

    inference_cmd = [
        sys.executable,
        str(project_root / "scripts" / "inference.py"),
        "--model", model_path,
        "--output", args.output,
        "--experiment", args.experiment
    ]

    result = subprocess.run(inference_cmd, cwd=project_root)

    if result.returncode != 0:
        print("\n❌ 추론 실패!")
        sys.exit(1)

    print("\n✅ 추론 완료!")

    # -------------- 완료 -------------- #
    print("\n" + "=" * 60)
    print("🎉 전체 파이프라인 완료!")
    print(f"  모델: {model_path}")
    print(f"  제출 파일: {args.output}")
    print("=" * 60)


# ==================== 실행부 ==================== #
if __name__ == "__main__":
    main()
