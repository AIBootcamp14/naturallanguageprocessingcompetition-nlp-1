"""
Solar API 테스트

PRD 09: Solar API 최적화 전략 구현
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ------------------------- 프로젝트 모듈 ------------------------- #
from src.api import SolarAPI, create_solar_api


# ==================== 테스트 함수 정의 ==================== #
# ---------------------- SolarAPI 초기화 테스트 ---------------------- #
def test_solar_api_init():
    """SolarAPI 초기화 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 1: SolarAPI 초기화")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # API 키 없이 초기화 (테스트용)
        api = SolarAPI(api_key=None, token_limit=512)

        # 초기화 정보 출력
        print("✅ SolarAPI 초기화 성공")
        print(f"  - 토큰 제한: {api.token_limit}")
        print(f"  - 캐시 디렉토리: {api.cache_dir}")
        print(f"  - 클라이언트 상태: {'설정됨' if api.client else '미설정'}")

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 대화 전처리 테스트 ---------------------- #
def test_preprocess_dialogue():
    """대화 전처리 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 2: 대화 전처리")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # SolarAPI 인스턴스 생성
        api = SolarAPI(api_key=None)

        # 테스트 대화 문자열
        dialogue = "#Person1#: 안녕하세요. 오늘 날씨가 좋네요. #Person2#: 네, 정말 좋아요."

        # 대화 전처리 실행
        processed = api.preprocess_dialogue(dialogue)

        # 결과 출력
        print(f"원본: {dialogue}")
        print(f"전처리: {processed}")

        # -------------- 검증 -------------- #
        # Person1이 A로 변환되었는지 확인
        assert 'A:' in processed, "Person1이 A로 변환되지 않음"
        # Person2가 B로 변환되었는지 확인
        assert 'B:' in processed, "Person2가 B로 변환되지 않음"
        # Person 태그가 제거되었는지 확인
        assert '#Person' not in processed, "Person 태그가 남아있음"

        print("✅ 대화 전처리 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 토큰 추정 테스트 ---------------------- #
def test_estimate_tokens():
    """토큰 추정 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 3: 토큰 추정")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # SolarAPI 인스턴스 생성
        api = SolarAPI(api_key=None)

        # -------------- 한글 텍스트 -------------- #
        # 한글 텍스트 토큰 추정
        korean_text = "안녕하세요" * 10
        korean_tokens = api.estimate_tokens(korean_text)

        # -------------- 영어 텍스트 -------------- #
        # 영어 텍스트 토큰 추정
        english_text = "Hello world " * 10
        english_tokens = api.estimate_tokens(english_text)

        # 추정 결과 출력
        print(f"한글 텍스트 ({len(korean_text)}자): {korean_tokens} 토큰")
        print(f"영어 텍스트 ({len(english_text)}자): {english_tokens} 토큰")

        # -------------- 검증 -------------- #
        # 토큰 수가 0보다 큰지 확인
        assert korean_tokens > 0, "토큰 수가 0"
        assert english_tokens > 0, "토큰 수가 0"

        # 한글 토큰 추정 정확도 검증 (2.5자당 1토큰 정도)
        expected_korean = len(korean_text) / 2.5
        assert abs(korean_tokens - expected_korean) < 10, "한글 토큰 추정이 부정확"

        print("✅ 토큰 추정 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 스마트 절단 테스트 ---------------------- #
def test_smart_truncate():
    """스마트 절단 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 4: 스마트 절단")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # SolarAPI 인스턴스 생성 (토큰 제한 50)
        api = SolarAPI(api_key=None, token_limit=50)

        # 긴 텍스트 생성
        long_text = "이것은 첫 번째 문장입니다. " * 20

        # 스마트 절단 실행
        truncated = api.smart_truncate(long_text, max_tokens=50)

        # 토큰 수 계산
        original_tokens = api.estimate_tokens(long_text)
        truncated_tokens = api.estimate_tokens(truncated)

        # 결과 출력
        print(f"원본: {original_tokens} 토큰")
        print(f"절단: {truncated_tokens} 토큰")
        print(f"절단된 텍스트 길이: {len(truncated)}자")

        # -------------- 검증 -------------- #
        # 토큰 제한을 초과하지 않는지 확인
        assert truncated_tokens <= 60, "토큰 제한을 초과"
        # 텍스트가 줄어들었는지 확인
        assert len(truncated) < len(long_text), "텍스트가 줄어들지 않음"

        print("✅ 스마트 절단 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- Few-shot 프롬프트 생성 테스트 ---------------------- #
def test_build_few_shot_prompt():
    """Few-shot 프롬프트 생성 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 5: Few-shot 프롬프트 생성")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # SolarAPI 인스턴스 생성
        api = SolarAPI(api_key=None)

        # 테스트 데이터
        dialogue = "A: 안녕 B: 안녕"
        example_dialogue = "A: 점심 뭐 먹을까? B: 김치찌개"
        example_summary = "점심 메뉴 상의"

        # Few-shot 프롬프트 생성
        messages = api.build_few_shot_prompt(
            dialogue,
            example_dialogue,
            example_summary
        )

        # 메시지 정보 출력
        print(f"메시지 수: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")

        # -------------- 검증 -------------- #
        # 메시지 수 검증 (system + user + assistant + user)
        assert len(messages) == 4, "메시지 수가 4개가 아님 (system + user + assistant + user)"
        # 각 메시지 역할 검증
        assert messages[0]['role'] == 'system', "첫 번째는 system"
        assert messages[1]['role'] == 'user', "두 번째는 user (예시)"
        assert messages[2]['role'] == 'assistant', "세 번째는 assistant (예시 답변)"
        assert messages[3]['role'] == 'user', "네 번째는 user (실제 입력)"

        print("✅ Few-shot 프롬프트 생성 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 캐시 동작 테스트 ---------------------- #
def test_cache_operations():
    """캐시 동작 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 6: 캐시 동작")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        import tempfile
        import shutil

        # 임시 캐시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()

        # SolarAPI 인스턴스 생성 (임시 캐시 디렉토리 사용)
        api = SolarAPI(api_key=None, cache_dir=temp_dir)

        # -------------- 캐시에 데이터 추가 -------------- #
        # 캐시 데이터 추가
        test_key = "test_dialogue_hash"
        test_value = "테스트 요약"

        api.cache[test_key] = test_value
        api._save_cache()

        # -------------- 새로운 인스턴스로 캐시 로드 -------------- #
        # 새로운 인스턴스로 캐시 로드
        api2 = SolarAPI(api_key=None, cache_dir=temp_dir)

        # 캐시 로드 결과 출력
        print(f"캐시 저장 및 로드 성공")
        print(f"  - 캐시 항목 수: {len(api2.cache)}")
        print(f"  - 테스트 키 존재: {test_key in api2.cache}")

        # -------------- 검증 -------------- #
        # 캐시가 로드되었는지 확인
        assert test_key in api2.cache, "캐시가 로드되지 않음"
        # 캐시 값이 일치하는지 확인
        assert api2.cache[test_key] == test_value, "캐시 값이 다름"

        # 임시 디렉토리 정리
        shutil.rmtree(temp_dir)

        print("✅ 캐시 동작 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- create_solar_api 편의 함수 테스트 ---------------------- #
def test_create_solar_api():
    """create_solar_api 편의 함수 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 7: create_solar_api 편의 함수")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # 편의 함수로 SolarAPI 생성
        api = create_solar_api(api_key=None, token_limit=256)

        # 생성 결과 출력
        print("✅ create_solar_api 성공")
        print(f"  - 토큰 제한: {api.token_limit}")

        # -------------- 검증 -------------- #
        # 토큰 제한이 설정되었는지 확인
        assert api.token_limit == 256, "토큰 제한이 설정되지 않음"

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 메인 실행부 ==================== #
# ---------------------- 전체 테스트 실행 함수 ---------------------- #
def main():
    """전체 테스트 실행"""
    # 테스트 시작 헤더 출력
    print("\n" + "=" * 70)
    print(" " * 25 + "Solar API 테스트 시작")
    print("=" * 70)

    # -------------- 테스트 실행 -------------- #
    # 테스트 결과 수집용 리스트
    results = []
    results.append(("SolarAPI 초기화", test_solar_api_init()))
    results.append(("대화 전처리", test_preprocess_dialogue()))
    results.append(("토큰 추정", test_estimate_tokens()))
    results.append(("스마트 절단", test_smart_truncate()))
    results.append(("Few-shot 프롬프트 생성", test_build_few_shot_prompt()))
    results.append(("캐시 동작", test_cache_operations()))
    results.append(("create_solar_api 함수", test_create_solar_api()))

    # -------------- 결과 요약 출력 -------------- #
    print("\n" + "=" * 70)
    print(" " * 25 + "테스트 결과 요약")
    print("=" * 70)

    # 통과한 테스트 개수 계산
    passed = sum(1 for _, result in results if result)
    total = len(results)

    # 각 테스트 결과 출력
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")

    # 최종 요약 출력
    print("=" * 70)
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    print("=" * 70)

    # 주의사항 출력
    print("\n⚠️  참고사항:")
    print("- 실제 API 호출 테스트는 API 키가 필요합니다")
    print("- SOLAR_API_KEY 환경 변수를 설정하여 실제 API 테스트 가능")

    # 전체 테스트 통과 여부 반환
    return passed == total


# ---------------------- 메인 진입점 ---------------------- #
if __name__ == "__main__":
    success = main()  # 전체 테스트 실행
    sys.exit(0 if success else 1)  # 성공 시 0, 실패 시 1 반환
