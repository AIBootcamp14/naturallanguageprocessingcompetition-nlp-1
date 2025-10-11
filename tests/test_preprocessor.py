# ==================== Preprocessor 테스트 스크립트 ==================== #
"""
데이터 전처리 시스템 테스트

테스트 항목:
1. 노이즈 제거 (\\n, <br> 등)
2. 화자 추출
3. 턴 개수 계산
4. 대화 분할
5. DataFrame 전처리
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 서드파티 라이브러리 ---------------------- #
import pandas as pd

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.data import DialoguePreprocessor


# ==================== 테스트 함수들 ==================== #
# ---------------------- 노이즈 제거 테스트 ---------------------- #
def test_clean_dialogue():
    """노이즈 제거 테스트"""
    print("\n" + "="*60)
    print("테스트 1: 노이즈 제거")
    print("="*60)

    preprocessor = DialoguePreprocessor()                   # 전처리기 생성

    # 테스트 데이터
    test_cases = [
        # (입력, 기대 출력)
        ("안녕하세요\\n반갑습니다", "안녕하세요\n반갑습니다"),  # \\n → \n
        ("안녕<br>반갑습니다", "안녕\n반갑습니다"),  # <br> → \n
        ("안녕  하세요", "안녕 하세요"),  # 중복 공백 제거
        ("  안녕하세요  ", "안녕하세요"),  # 앞뒤 공백 제거
    ]

    # 각 테스트 케이스 실행
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = preprocessor.clean_dialogue(input_text)    # 전처리 실행
        print(f"  테스트 {i}: {'✅' if result == expected else '❌'}")
        print(f"    입력: {repr(input_text)}")
        print(f"    기대: {repr(expected)}")
        print(f"    결과: {repr(result)}")

        assert result == expected, f"테스트 {i} 실패"  # 검증

    print("✅ 노이즈 제거 테스트 성공!")


# ---------------------- 화자 추출 테스트 ---------------------- #
def test_extract_speakers():
    """화자 추출 테스트"""
    print("\n" + "="*60)
    print("테스트 2: 화자 추출")
    print("="*60)

    preprocessor = DialoguePreprocessor()                   # 전처리기 생성

    # 테스트 대화
    dialogue = "#Person1#: 안녕하세요\n#Person2#: 반갑습니다\n#Person1#: 잘 부탁드립니다"

    speakers = preprocessor.extract_speakers(dialogue)      # 화자 추출

    print(f"  대화: {dialogue[:50]}...")
    print(f"  화자: {speakers}")

    assert speakers == ['#Person1#', '#Person2#']           # 검증
    print("✅ 화자 추출 테스트 성공!")


# ---------------------- 턴 개수 계산 테스트 ---------------------- #
def test_count_turns():
    """턴 개수 계산 테스트"""
    print("\n" + "="*60)
    print("테스트 3: 턴 개수 계산")
    print("="*60)

    preprocessor = DialoguePreprocessor()                   # 전처리기 생성

    # 테스트 대화
    dialogue = "#Person1#: 안녕하세요\n#Person2#: 반갑습니다\n#Person1#: 잘 부탁드립니다"

    turns = preprocessor.count_turns(dialogue)              # 턴 개수 계산

    print(f"  대화: {dialogue[:50]}...")
    print(f"  턴 개수: {turns}")

    assert turns == 3                                       # 검증
    print("✅ 턴 개수 계산 테스트 성공!")


# ---------------------- 대화 분할 테스트 ---------------------- #
def test_split_dialogue():
    """대화 분할 테스트"""
    print("\n" + "="*60)
    print("테스트 4: 대화 분할")
    print("="*60)

    preprocessor = DialoguePreprocessor()                   # 전처리기 생성

    # 테스트 대화
    dialogue = "#Person1#: 안녕하세요\n#Person2#: 반갑습니다"

    turns = preprocessor.split_dialogue_by_speaker(dialogue)  # 대화 분할

    print(f"  대화: {dialogue}")
    print(f"  분할 결과:")
    for speaker, utterance in turns:
        print(f"    {speaker}: {utterance}")

    assert len(turns) == 2                                  # 턴 개수 검증
    assert turns[0][0] == '#Person1#'                       # 첫 번째 화자 검증
    assert turns[1][0] == '#Person2#'                       # 두 번째 화자 검증

    print("✅ 대화 분할 테스트 성공!")


# ---------------------- 실제 데이터 테스트 ---------------------- #
def test_real_data():
    """실제 데이터 테스트"""
    print("\n" + "="*60)
    print("테스트 5: 실제 데이터 전처리")
    print("="*60)

    # 실제 데이터 로드
    try:
        df = pd.read_csv('data/raw/train.csv')             # 학습 데이터 로드
        print(f"  원본 데이터 크기: {len(df)}")

        preprocessor = DialoguePreprocessor()               # 전처리기 생성

        # DataFrame 전처리
        df_processed = preprocessor.preprocess_dataframe(df)  # 전처리 실행

        print(f"  전처리 후 크기: {len(df_processed)}")
        print(f"  추가된 컬럼: {[col for col in df_processed.columns if col not in df.columns]}")

        # 샘플 출력
        if 'num_speakers' in df_processed.columns:
            print(f"\n  화자 수 통계:")
            print(df_processed['num_speakers'].value_counts().head())

        if 'num_turns' in df_processed.columns:
            print(f"\n  턴 수 통계:")
            print(df_processed['num_turns'].describe())

        print("\n✅ 실제 데이터 전처리 테스트 성공!")

    except FileNotFoundError:
        print("  ⚠️ 실제 데이터 파일을 찾을 수 없습니다 (선택적 테스트)")
        print("  → data/raw/train.csv 파일이 필요합니다")


# ==================== 메인 실행부 ==================== #
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Preprocessor 테스트 시작")
    print("="*60)

    try:
        # 모든 테스트 실행
        test_clean_dialogue()                               # 테스트 1
        test_extract_speakers()                             # 테스트 2
        test_count_turns()                                  # 테스트 3
        test_split_dialogue()                               # 테스트 4
        test_real_data()                                    # 테스트 5

        # 최종 결과
        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 테스트 실패: {e}")
        print("="*60)
        raise
