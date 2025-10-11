# ==================== Predictor 테스트 스크립트 ==================== #
"""
추론 시스템 테스트

테스트 항목:
1. Predictor 초기화
2. 단일 예측
3. 배치 예측
4. DataFrame 예측
5. 제출 파일 생성
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
from pathlib import Path
import tempfile

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 서드파티 라이브러리 ---------------------- #
import pandas as pd

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.config import load_config
from src.models import load_model_and_tokenizer
from src.inference import Predictor, create_predictor


# ==================== 테스트 함수들 ==================== #
# ---------------------- Predictor 초기화 테스트 ---------------------- #
def test_predictor_initialization():
    """Predictor 초기화 테스트"""
    print("\n" + "="*60)
    print("테스트 1: Predictor 초기화")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")             # 베이스라인 Config
    print(f"  Config 로드 완료")

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(config)  # 모델 로드
    print(f"  모델 로드 완료")

    # Predictor 초기화
    predictor = Predictor(                              # Predictor 생성
        model=model,
        tokenizer=tokenizer,
        config=config
    )

    # 검증
    assert predictor.model is not None                  # 모델 존재 확인
    assert predictor.tokenizer is not None              # 토크나이저 확인
    assert predictor.generation_config is not None      # 생성 설정 확인
    assert predictor.device is not None                 # 디바이스 확인

    print(f"\n  디바이스: {predictor.device}")
    print(f"  생성 설정: {predictor.generation_config}")

    print("\n✅ Predictor 초기화 테스트 성공!")


# ---------------------- 단일 예측 테스트 ---------------------- #
def test_single_prediction():
    """단일 예측 테스트"""
    print("\n" + "="*60)
    print("테스트 2: 단일 예측")
    print("="*60)

    # Config 및 모델 로드
    config = load_config("baseline_kobart")
    model, tokenizer = load_model_and_tokenizer(config)

    # Predictor 생성
    predictor = Predictor(model, tokenizer, config)

    # 테스트 대화
    dialogue = "#Person1#: 안녕하세요. 오늘 날씨가 참 좋네요.\\n#Person2#: 네, 정말 화창하네요. 산책하기 좋은 날씨예요."

    # 예측 실행
    print(f"\n  입력 대화: {dialogue[:50]}...")
    summary = predictor.predict_single(dialogue)        # 단일 예측

    # 결과 출력
    print(f"  예측 요약: {summary}")

    # 검증
    assert isinstance(summary, str)                     # 문자열 타입 확인
    assert len(summary) > 0                             # 비어있지 않음 확인

    print("\n✅ 단일 예측 테스트 성공!")


# ---------------------- 배치 예측 테스트 ---------------------- #
def test_batch_prediction():
    """배치 예측 테스트"""
    print("\n" + "="*60)
    print("테스트 3: 배치 예측")
    print("="*60)

    # Config 및 모델 로드
    config = load_config("baseline_kobart")
    model, tokenizer = load_model_and_tokenizer(config)

    # Predictor 생성
    predictor = Predictor(model, tokenizer, config)

    # 테스트 대화 리스트
    dialogues = [
        "#Person1#: 안녕하세요\\n#Person2#: 반갑습니다",
        "#Person1#: 오늘 날씨가 좋네요\\n#Person2#: 네, 산책하기 좋아요",
        "#Person1#: 점심 뭐 먹을까요?\\n#Person2#: 김치찌개 어때요?"
    ]

    # 배치 예측 실행
    print(f"\n  샘플 개수: {len(dialogues)}")
    summaries = predictor.predict_batch(                # 배치 예측
        dialogues,
        batch_size=2,
        show_progress=True
    )

    # 결과 출력
    print(f"\n  예측 결과:")
    for i, summary in enumerate(summaries, 1):
        print(f"    {i}. {summary}")

    # 검증
    assert len(summaries) == len(dialogues)             # 개수 일치 확인
    assert all(isinstance(s, str) for s in summaries)   # 모두 문자열 확인
    assert all(len(s) > 0 for s in summaries)           # 모두 비어있지 않음

    print("\n✅ 배치 예측 테스트 성공!")


# ---------------------- DataFrame 예측 테스트 ---------------------- #
def test_dataframe_prediction():
    """DataFrame 예측 테스트"""
    print("\n" + "="*60)
    print("테스트 4: DataFrame 예측")
    print("="*60)

    # Config 및 모델 로드
    config = load_config("baseline_kobart")
    model, tokenizer = load_model_and_tokenizer(config)

    # Predictor 생성
    predictor = Predictor(model, tokenizer, config)

    # 테스트 DataFrame 생성
    test_df = pd.DataFrame({
        'fname': ['test_001', 'test_002', 'test_003'],
        'dialogue': [
            "#Person1#: 안녕하세요\\n#Person2#: 반갑습니다",
            "#Person1#: 날씨가 좋네요\\n#Person2#: 네, 좋아요",
            "#Person1#: 점심 뭐 드실래요?\\n#Person2#: 김치찌개요"
        ]
    })

    # DataFrame 예측 실행
    print(f"\n  샘플 개수: {len(test_df)}")
    result_df = predictor.predict_dataframe(            # DataFrame 예측
        test_df,
        batch_size=2,
        show_progress=True
    )

    # 결과 출력
    print(f"\n  예측 결과:")
    print(result_df[['fname', 'summary']])

    # 검증
    assert 'summary' in result_df.columns               # summary 컬럼 존재 확인
    assert len(result_df) == len(test_df)               # 개수 일치 확인
    assert result_df['summary'].notna().all()           # 모두 값이 있음 확인

    print("\n✅ DataFrame 예측 테스트 성공!")


# ---------------------- 제출 파일 생성 테스트 ---------------------- #
def test_submission_creation():
    """제출 파일 생성 테스트"""
    print("\n" + "="*60)
    print("테스트 5: 제출 파일 생성")
    print("="*60)

    # Config 및 모델 로드
    config = load_config("baseline_kobart")
    model, tokenizer = load_model_and_tokenizer(config)

    # Predictor 생성
    predictor = Predictor(model, tokenizer, config)

    # 테스트 DataFrame 생성
    test_df = pd.DataFrame({
        'fname': ['test_001', 'test_002'],
        'dialogue': [
            "#Person1#: 안녕하세요\\n#Person2#: 반갑습니다",
            "#Person1#: 날씨가 좋네요\\n#Person2#: 네, 좋아요"
        ]
    })

    # 임시 출력 파일 경로
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_path = f.name

    # 제출 파일 생성
    print(f"\n  출력 경로: {output_path}")
    submission_df = predictor.create_submission(        # 제출 파일 생성
        test_df,
        output_path=output_path,
        batch_size=2,
        show_progress=True
    )

    # 결과 확인
    print(f"\n  제출 DataFrame:")
    print(submission_df)

    # 검증
    assert submission_df.columns.tolist() == ['fname', 'summary']  # 컬럼 확인
    assert len(submission_df) == len(test_df)           # 개수 일치 확인

    # 파일 존재 확인
    assert Path(output_path).exists()                   # 파일 생성 확인

    # 파일 읽어서 검증
    saved_df = pd.read_csv(output_path)                 # 저장된 파일 읽기
    assert len(saved_df) == len(test_df)                # 저장된 개수 확인

    # 임시 파일 삭제
    Path(output_path).unlink()                          # 파일 삭제

    print("\n✅ 제출 파일 생성 테스트 성공!")


# ---------------------- 편의 함수 테스트 ---------------------- #
def test_convenience_function():
    """편의 함수 테스트"""
    print("\n" + "="*60)
    print("테스트 6: 편의 함수")
    print("="*60)

    # Config 및 모델 로드
    config = load_config("baseline_kobart")
    model, tokenizer = load_model_and_tokenizer(config)

    # 편의 함수로 Predictor 생성
    predictor = create_predictor(                       # 편의 함수 사용
        model=model,
        tokenizer=tokenizer,
        config=config
    )

    # 검증
    assert isinstance(predictor, Predictor)             # 타입 확인
    assert predictor.model is not None                  # 모델 존재 확인

    print(f"\n  Predictor 생성 완료")
    print(f"  타입: {type(predictor).__name__}")

    print("\n✅ 편의 함수 테스트 성공!")


# ==================== 메인 실행부 ==================== #
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Predictor 테스트 시작")
    print("="*60)
    print("\n⚠️ 참고: 실제 추론을 수행하므로 시간이 걸릴 수 있습니다.")

    try:
        # 모든 테스트 실행
        test_predictor_initialization()                 # 테스트 1
        test_single_prediction()                        # 테스트 2
        test_batch_prediction()                         # 테스트 3
        test_dataframe_prediction()                     # 테스트 4
        test_submission_creation()                      # 테스트 5
        test_convenience_function()                     # 테스트 6

        # 최종 결과
        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 테스트 실패: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        raise
