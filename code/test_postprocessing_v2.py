#!/usr/bin/env python3
"""
Experiment #2: 후처리 v2 호환성 테스트

scripts/inference_utils.py에 추가된 새로운 함수들이
올바르게 작동하는지 검증합니다.

테스트 항목:
1. normalize_whitespace() - 공백 정규화
2. remove_duplicate_sentences() - 중복 문장 제거
3. postprocess_summaries_v2() - 통합 후처리
"""

import sys
sys.path.append('../scripts')

from inference_utils import (
    normalize_whitespace,
    remove_duplicate_sentences,
    postprocess_summaries_v2
)


def test_normalize_whitespace():
    """공백 정규화 테스트"""
    print("=" * 80)
    print("Test 1: normalize_whitespace() 함수 테스트")
    print("=" * 80)

    test_cases = [
        ("안녕하세요    세상  ", "안녕하세요 세상"),
        ("  텍스트\n\n정규화  ", "텍스트 정규화"),
        ("여러  개의   공백", "여러 개의 공백"),
        ("탭\t문자\t테스트", "탭 문자 테스트"),
    ]

    passed = 0
    failed = 0

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = normalize_whitespace(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n  Case {i}: {status}")
        print(f"    Input:    '{input_text}'")
        print(f"    Expected: '{expected}'")
        print(f"    Got:      '{result}'")

    print(f"\n  Result: {passed} passed, {failed} failed")
    return failed == 0


def test_remove_duplicate_sentences():
    """중복 문장 제거 테스트"""
    print("\n" + "=" * 80)
    print("Test 2: remove_duplicate_sentences() 함수 테스트")
    print("=" * 80)

    test_cases = [
        (
            "오늘 날씨가 좋습니다. 오늘 날씨가 좋습니다. 내일도 좋을 것 같습니다.",
            "오늘 날씨가 좋습니다. 내일도 좋을 것 같습니다."
        ),
        (
            "안녕하세요! 안녕하세요! 반갑습니다.",
            "안녕하세요! 반갑습니다."
        ),
        (
            "첫 번째 문장. 두 번째 문장. 첫 번째 문장.",
            "첫 번째 문장. 두 번째 문장."
        ),
        (
            "중복 없는 문장입니다. 다른 문장입니다.",
            "중복 없는 문장입니다. 다른 문장입니다."
        ),
    ]

    passed = 0
    failed = 0

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = remove_duplicate_sentences(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n  Case {i}: {status}")
        print(f"    Input:    '{input_text}'")
        print(f"    Expected: '{expected}'")
        print(f"    Got:      '{result}'")

    print(f"\n  Result: {passed} passed, {failed} failed")
    return failed == 0


def test_postprocess_summaries_v2():
    """통합 후처리 v2 테스트"""
    print("\n" + "=" * 80)
    print("Test 3: postprocess_summaries_v2() 함수 테스트")
    print("=" * 80)

    test_cases = [
        {
            "input": ["<s>오늘  날씨가 좋습니다.  오늘 날씨가 좋습니다.</s>"],
            "tokens": ["<s>", "</s>"],
            "expected": ["오늘 날씨가 좋습니다."]
        },
        {
            "input": ["<s>안녕하세요!  안녕하세요!  반갑습니다.</s><pad><pad>"],
            "tokens": ["<s>", "</s>", "<pad>"],
            "expected": ["안녕하세요! 반갑습니다."]
        },
        {
            "input": ["<usr>  여행  갔다왔어요.  <usr>  여행  갔다왔어요.  "],
            "tokens": ["<usr>"],
            "expected": ["여행 갔다왔어요."]
        },
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        result = postprocess_summaries_v2(test_case["input"], test_case["tokens"])
        status = "✅ PASS" if result == test_case["expected"] else "❌ FAIL"

        if result == test_case["expected"]:
            passed += 1
        else:
            failed += 1

        print(f"\n  Case {i}: {status}")
        print(f"    Input:    {test_case['input']}")
        print(f"    Tokens:   {test_case['tokens']}")
        print(f"    Expected: {test_case['expected']}")
        print(f"    Got:      {result}")

    print(f"\n  Result: {passed} passed, {failed} failed")
    return failed == 0


def test_import():
    """Import 테스트"""
    print("=" * 80)
    print("Test 0: Import 테스트")
    print("=" * 80)

    try:
        print("  ✅ normalize_whitespace import 성공")
        print("  ✅ remove_duplicate_sentences import 성공")
        print("  ✅ postprocess_summaries_v2 import 성공")
        return True
    except Exception as e:
        print(f"  ❌ Import 실패: {e}")
        return False


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 80)
    print("🧪 Experiment #2: 후처리 v2 호환성 테스트 시작")
    print("=" * 80 + "\n")

    results = []

    # Import 테스트
    results.append(("Import", test_import()))

    # 개별 함수 테스트
    results.append(("normalize_whitespace", test_normalize_whitespace()))
    results.append(("remove_duplicate_sentences", test_remove_duplicate_sentences()))
    results.append(("postprocess_summaries_v2", test_postprocess_summaries_v2()))

    # 최종 결과
    print("\n" + "=" * 80)
    print("📊 최종 테스트 결과")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 모든 테스트 통과! 후처리 v2가 정상 작동합니다.")
        print("✅ exp2_postprocessing.ipynb 생성을 진행할 수 있습니다.")
    else:
        print("❌ 일부 테스트 실패. 코드를 수정해야 합니다.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)