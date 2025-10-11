"""
프롬프트 엔지니어링 시스템 테스트

PRD 15: 프롬프트 엔지니어링 전략 구현
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.prompts import (
    PromptTemplate,
    PromptLibrary,
    create_prompt_library,
    PromptSelector,
    create_prompt_selector
)


def test_prompt_template():
    """PromptTemplate 클래스 테스트"""
    print("\n" + "="*60)
    print("테스트 1: PromptTemplate 클래스")
    print("="*60)

    try:
        template = PromptTemplate(
            name="test_template",
            template="대화: {dialogue}\n요약: {summary}",
            description="테스트 템플릿",
            category="test",
            variables=["dialogue", "summary"]
        )

        # 템플릿 포맷팅
        result = template.format(
            dialogue="안녕하세요",
            summary="인사"
        )

        print(f"템플릿 이름: {template.name}")
        print(f"카테고리: {template.category}")
        print(f"변수: {template.variables}")
        print(f"포맷 결과:\n{result}")

        # 검증
        assert "안녕하세요" in result, "대화 내용이 없음"
        assert "인사" in result, "요약 내용이 없음"

        # 누락된 변수 테스트
        try:
            template.format(dialogue="테스트")  # summary 누락
            print("❌ 누락된 변수 감지 실패")
            return False
        except ValueError as e:
            print(f"✓ 누락된 변수 감지: {str(e)}")

        print("✅ PromptTemplate 클래스 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_library_init():
    """PromptLibrary 초기화 테스트"""
    print("\n" + "="*60)
    print("테스트 2: PromptLibrary 초기화")
    print("="*60)

    try:
        library = PromptLibrary()

        # 기본 템플릿 로드 확인
        template_names = library.list_templates()

        print(f"로드된 템플릿 수: {len(template_names)}")
        print(f"템플릿 목록: {template_names[:5]}...")

        # 주요 템플릿 존재 확인
        required_templates = [
            'zero_shot_basic',
            'few_shot_1shot',
            'cot_step_by_step',
            'short_dialogue',
            'two_speakers',
            'compressed_minimal'
        ]

        for template_name in required_templates:
            assert template_name in template_names, f"{template_name} 템플릿 없음"
            print(f"  ✓ {template_name}")

        print("✅ PromptLibrary 초기화 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_library_operations():
    """PromptLibrary 기능 테스트"""
    print("\n" + "="*60)
    print("테스트 3: PromptLibrary 기능")
    print("="*60)

    try:
        library = PromptLibrary()

        # 1. 템플릿 조회
        template = library.get_template('zero_shot_basic')
        assert template is not None, "템플릿 조회 실패"
        print(f"✓ 템플릿 조회: {template.name}")

        # 2. 카테고리별 조회
        zero_shot_templates = library.list_templates(category='zero_shot')
        print(f"✓ Zero-shot 템플릿 수: {len(zero_shot_templates)}")
        assert len(zero_shot_templates) > 0, "Zero-shot 템플릿 없음"

        # 3. 템플릿 포맷팅
        dialogue = "A: 안녕하세요 B: 안녕하세요"
        formatted = template.format(dialogue=dialogue)
        assert dialogue in formatted, "대화 내용이 포맷되지 않음"
        print(f"✓ 템플릿 포맷팅 성공")

        # 4. 토큰 추정
        tokens = library.estimate_tokens('zero_shot_basic', dialogue=dialogue)
        print(f"✓ 토큰 추정: {tokens} 토큰")
        assert tokens > 0, "토큰 추정 실패"

        # 5. 새 템플릿 추가
        new_template = PromptTemplate(
            name="custom_template",
            template="Custom: {text}",
            description="커스텀",
            category="custom",
            variables=["text"]
        )
        library.add_template(new_template)
        assert library.get_template('custom_template') is not None, "템플릿 추가 실패"
        print(f"✓ 새 템플릿 추가 성공")

        print("✅ PromptLibrary 기능 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_selector_by_length():
    """PromptSelector 길이 기반 선택 테스트"""
    print("\n" + "="*60)
    print("테스트 4: 길이 기반 프롬프트 선택")
    print("="*60)

    try:
        selector = PromptSelector()

        # 짧은 대화
        short_dialogue = "A: 안녕 B: 안녕"
        template = selector.select_by_length(short_dialogue)
        print(f"짧은 대화 ({len(short_dialogue.split())}단어): {template.name}")
        assert template.name == 'short_dialogue', "짧은 대화 템플릿 선택 실패"

        # 중간 대화
        medium_dialogue = " ".join(["단어"] * 250)
        template = selector.select_by_length(medium_dialogue)
        print(f"중간 대화 ({len(medium_dialogue.split())}단어): {template.name}")
        assert template.name == 'medium_dialogue', "중간 대화 템플릿 선택 실패"

        # 긴 대화
        long_dialogue = " ".join(["단어"] * 600)
        template = selector.select_by_length(long_dialogue)
        print(f"긴 대화 ({len(long_dialogue.split())}단어): {template.name}")
        assert template.name == 'long_dialogue', "긴 대화 템플릿 선택 실패"

        print("✅ 길이 기반 선택 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_selector_by_speakers():
    """PromptSelector 참여자 수 기반 선택 테스트"""
    print("\n" + "="*60)
    print("테스트 5: 참여자 수 기반 프롬프트 선택")
    print("="*60)

    try:
        selector = PromptSelector()

        # 2인 대화
        two_person = "#Person1#: 안녕 #Person2#: 안녕"
        template = selector.select_by_speakers(two_person)
        num_speakers = selector._count_speakers(two_person)
        print(f"2인 대화 ({num_speakers}명): {template.name}")
        assert template.name == 'two_speakers', "2인 대화 템플릿 선택 실패"

        # 소그룹 (3명)
        small_group = "#Person1#: A #Person2#: B #Person3#: C"
        template = selector.select_by_speakers(small_group)
        num_speakers = selector._count_speakers(small_group)
        print(f"소그룹 대화 ({num_speakers}명): {template.name}")
        assert template.name == 'group_small', "소그룹 템플릿 선택 실패"

        # 대규모 (5명)
        large_group = "#Person1#: A #Person2#: B #Person3#: C #Person4#: D #Person5#: E"
        template = selector.select_by_speakers(large_group)
        num_speakers = selector._count_speakers(large_group)
        print(f"대규모 대화 ({num_speakers}명): {template.name}")
        assert template.name == 'group_large', "대규모 템플릿 선택 실패"

        print("✅ 참여자 수 기반 선택 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_selector_by_token_budget():
    """PromptSelector 토큰 예산 기반 선택 테스트"""
    print("\n" + "="*60)
    print("테스트 6: 토큰 예산 기반 프롬프트 선택")
    print("="*60)

    try:
        selector = PromptSelector()

        # 매우 긴 대화 (토큰 예산의 80% 이상)
        very_long_dialogue = "대화 내용 " * 200  # 약 400 토큰
        template = selector.select_by_token_budget(very_long_dialogue, token_budget=500)
        estimated_tokens = selector._estimate_tokens(very_long_dialogue)
        print(f"긴 대화 ({estimated_tokens} 토큰): {template.name}")
        assert template.name == 'compressed_minimal', "압축 최소 템플릿 선택 실패"

        # 중간 길이 대화
        medium_dialogue = "대화 내용 " * 150  # 약 340 토큰 (500의 68%)
        template = selector.select_by_token_budget(medium_dialogue, token_budget=500)
        estimated_tokens = selector._estimate_tokens(medium_dialogue)
        print(f"중간 대화 ({estimated_tokens} 토큰): {template.name}")
        assert template.name == 'compressed_concise', "압축 간결 템플릿 선택 실패"

        # 짧은 대화 (여유 있음)
        short_dialogue = "대화 내용 " * 20  # 약 40 토큰
        template = selector.select_by_token_budget(short_dialogue, token_budget=500)
        estimated_tokens = selector._estimate_tokens(short_dialogue)
        print(f"짧은 대화 ({estimated_tokens} 토큰): {template.name}")
        assert template.name == 'zero_shot_detailed', "상세 템플릿 선택 실패"

        print("✅ 토큰 예산 기반 선택 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_selector_adaptive():
    """PromptSelector 적응형 선택 테스트"""
    print("\n" + "="*60)
    print("테스트 7: 적응형 프롬프트 선택")
    print("="*60)

    try:
        selector = PromptSelector()

        dialogue = "#Person1#: 안녕하세요 #Person2#: 안녕하세요"

        # 1. 토큰 예산 우선
        template = selector.select_adaptive(dialogue, token_budget=50)
        print(f"✓ 토큰 예산 우선: {template.name}")

        # 2. 카테고리 선호
        template = selector.select_adaptive(dialogue, prefer_category="zero_shot")
        print(f"✓ Zero-shot 선호: {template.name}")
        assert template.category == "zero_shot", "카테고리 선택 실패"

        template = selector.select_adaptive(dialogue, prefer_category="cot")
        print(f"✓ CoT 선호: {template.name}")
        assert template.category == "cot", "CoT 선택 실패"

        # 3. 기본 전략 (길이 기반)
        template = selector.select_adaptive(dialogue)
        print(f"✓ 기본 전략: {template.name}")

        print("✅ 적응형 선택 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_selection_info():
    """대화 분석 정보 테스트"""
    print("\n" + "="*60)
    print("테스트 8: 대화 분석 정보")
    print("="*60)

    try:
        selector = PromptSelector()

        dialogue = "#Person1#: 안녕하세요. 오늘 날씨가 좋네요. #Person2#: 네, 정말 좋아요."

        info = selector.get_selection_info(dialogue)

        print(f"단어 수: {info['word_count']}")
        print(f"참여자 수: {info['num_speakers']}")
        print(f"추정 토큰: {info['estimated_tokens']}")
        print(f"추천 템플릿:")
        for criterion, template_name in info['recommendations'].items():
            print(f"  - {criterion}: {template_name}")
        print(f"특성:")
        for char, value in info['characteristics'].items():
            print(f"  - {char}: {value}")

        # 검증
        assert 'word_count' in info, "단어 수 없음"
        assert 'num_speakers' in info, "참여자 수 없음"
        assert 'estimated_tokens' in info, "토큰 추정 없음"
        assert 'recommendations' in info, "추천 정보 없음"
        assert 'characteristics' in info, "특성 정보 없음"

        print("✅ 대화 분석 정보 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_create_functions():
    """편의 생성 함수 테스트"""
    print("\n" + "="*60)
    print("테스트 9: 편의 생성 함수")
    print("="*60)

    try:
        # create_prompt_library
        library = create_prompt_library()
        assert len(library.templates) > 0, "라이브러리 생성 실패"
        print(f"✓ create_prompt_library: {len(library.templates)}개 템플릿")

        # create_prompt_selector
        selector = create_prompt_selector()
        assert selector.library is not None, "선택기 생성 실패"
        print(f"✓ create_prompt_selector: 라이브러리 연결 확인")

        # 커스텀 라이브러리로 선택기 생성
        selector_custom = create_prompt_selector(library)
        assert selector_custom.library == library, "커스텀 라이브러리 연결 실패"
        print(f"✓ 커스텀 라이브러리 연결 성공")

        print("✅ 편의 생성 함수 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """전체 테스트 실행"""
    print("\n" + "="*70)
    print(" "*20 + "프롬프트 엔지니어링 시스템 테스트 시작")
    print("="*70)

    results = []
    results.append(("PromptTemplate 클래스", test_prompt_template()))
    results.append(("PromptLibrary 초기화", test_prompt_library_init()))
    results.append(("PromptLibrary 기능", test_prompt_library_operations()))
    results.append(("길이 기반 선택", test_prompt_selector_by_length()))
    results.append(("참여자 수 기반 선택", test_prompt_selector_by_speakers()))
    results.append(("토큰 예산 기반 선택", test_prompt_selector_by_token_budget()))
    results.append(("적응형 선택", test_prompt_selector_adaptive()))
    results.append(("대화 분석 정보", test_selection_info()))
    results.append(("편의 생성 함수", test_create_functions()))

    # 결과 요약
    print("\n" + "="*70)
    print(" "*25 + "테스트 결과 요약")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")

    print("="*70)
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
