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

    print("=" * 60)
    print(f"추론 시작")
    print("=" * 60)

    # -------------- 1. Config 로드 (선택적) -------------- #
    print("\n[1/5] Config 로딩...")
    try:
        config = load_config(args.experiment)
        print(f"  ✅ Config 로드 완료: {args.experiment}")
    except:
        print("  ⚠️ Config 로드 실패, 기본 설정 사용")
        config = None

    # -------------- 2. 모델 및 토크나이저 로드 -------------- #
    print(f"\n[2/5] 모델 로딩: {args.model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"  ✅ 모델 로드 완료")
    print(f"  모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # -------------- 3. 테스트 데이터 로드 -------------- #
    print(f"\n[3/5] 테스트 데이터 로딩: {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    print(f"  ✅ 테스트 샘플: {len(test_df)}개")

    # -------------- 4. Predictor 생성 및 추론 -------------- #
    print("\n[4/5] 추론 실행...")
    predictor = create_predictor(
        model=model,
        tokenizer=tokenizer,
        config=config
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
    print("\n[5/5] 추론 완료!")
    print(f"  ✅ 제출 파일 생성: {args.output}")
    print(f"  샘플 수: {len(submission_df)}")

    # 샘플 출력
    print("\n  샘플 예측 결과 (처음 3개):")
    for idx, row in submission_df.head(3).iterrows():
        print(f"    [{row['fname']}]: {row['summary'][:50]}...")

    print("\n" + "=" * 60)
    print("🎉 추론 완료!")
    print("=" * 60)


# ==================== 실행부 ==================== #
if __name__ == "__main__":
    main()
