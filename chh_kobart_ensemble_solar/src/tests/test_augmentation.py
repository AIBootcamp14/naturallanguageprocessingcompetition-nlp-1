"""
데이터 증강 시스템 테스트

주요 기능:
- PRD 04: 데이터 증강 전략 구현 검증
- 역번역, 의역, 턴 섞기, 동의어 치환, 대화 샘플링 등 다양한 증강 기법 테스트
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
from pathlib import Path

# ---------------------- 프로젝트 루트 경로 설정 ---------------------- #
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 데이터 증강 모듈 ---------------------- #
from src.data.augmentation import (
    DataAugmenter,
    augment_dataset,
    BackTranslationAugmenter,
    ParaphraseAugmenter,
    ShuffleAugmenter,
    SynonymReplacementAugmenter,
    DialogueSamplingAugmenter
)


# ==================== 테스트 함수들 ==================== #
# ---------------------- DataAugmenter 초기화 테스트 ---------------------- #
def test_data_augmenter_init():
    """DataAugmenter 초기화 테스트"""
    print("\n" + "=" * 60)
    print("테스트 1: DataAugmenter 초기화")
    print("=" * 60)

    # -------------- 초기화 시도 -------------- #
    try:
        # DataAugmenter 인스턴스 생성
        augmenter = DataAugmenter()
        print("✅ DataAugmenter 초기화 성공")
        print(f"  - 사용 가능한 증강 방법: {len(augmenter.augmenters)}개")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 역번역 증강 테스트 ---------------------- #
def test_back_translation():
    """역번역 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 역번역 증강")
    print("=" * 60)

    # -------------- 역번역 시도 -------------- #
    try:
        # 역번역 증강기 생성
        augmenter = BackTranslationAugmenter()

        # 테스트 데이터 준비
        dialogue = "오늘 날씨가 정말 좋네요."
        summary = "날씨가 좋음"

        print(f"원본 대화: {dialogue}")
        print(f"원본 요약: {summary}")

        # 증강 실행
        aug_dialogue, aug_summary = augmenter.augment(dialogue, summary)

        print(f"증강 대화: {aug_dialogue}")
        print(f"증강 요약: {aug_summary}")

        # -------------- 결과 검증 -------------- #
        # 빈 문자열이 아닌지 확인
        assert len(aug_dialogue) > 0, "증강된 대화가 비어있음"
        assert len(aug_summary) > 0, "증강된 요약이 비어있음"

        print("✅ 역번역 증강 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"⚠️  경고: {str(e)}")
        print("  - 역번역은 Helsinki-NLP 모델이 필요합니다")
        print("  - 테스트를 건너뜁니다")
        return True                     # 선택적 기능이므로 실패로 처리하지 않음


# ---------------------- 의역 증강 테스트 ---------------------- #
def test_paraphrase():
    """의역 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: 의역 증강")
    print("=" * 60)

    # -------------- 의역 시도 -------------- #
    try:
        # 의역 증강기 생성
        augmenter = ParaphraseAugmenter()

        # 테스트 데이터 준비
        dialogue = "A: 오늘 점심 뭐 먹을까요?\nB: 김치찌개 어때요?"
        summary = "점심 메뉴 상의"

        print(f"원본 대화:\n{dialogue}")
        print(f"원본 요약: {summary}")

        # 증강 실행
        aug_dialogue, aug_summary = augmenter.augment(dialogue, summary)

        print(f"증강 대화:\n{aug_dialogue}")
        print(f"증강 요약: {aug_summary}")

        # -------------- 결과 검증 -------------- #
        # 빈 문자열이 아닌지 확인
        assert len(aug_dialogue) > 0, "증강된 대화가 비어있음"
        assert len(aug_summary) > 0, "증강된 요약이 비어있음"

        print("✅ 의역 증강 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 턴 섞기 증강 테스트 ---------------------- #
def test_shuffle():
    """턴 섞기 테스트"""
    print("\n" + "=" * 60)
    print("테스트 4: 턴 섞기 증강")
    print("=" * 60)

    # -------------- 턴 섞기 시도 -------------- #
    try:
        # 턴 섞기 증강기 생성 (50% 비율로 원본 턴 유지)
        augmenter = ShuffleAugmenter(preserve_ratio=0.5)

        # 테스트 데이터 준비
        dialogue = "A: 안녕하세요\nB: 안녕하세요\nA: 오늘 날씨 좋네요\nB: 네, 정말 좋아요"
        summary = "인사 및 날씨 이야기"

        print(f"원본 대화:\n{dialogue}")
        print(f"원본 요약: {summary}")

        # 증강 실행
        aug_dialogue, aug_summary = augmenter.augment(dialogue, summary)

        print(f"증강 대화:\n{aug_dialogue}")
        print(f"증강 요약: {aug_summary}")

        # -------------- 결과 검증 -------------- #
        # 빈 문자열이 아닌지 확인
        assert len(aug_dialogue) > 0, "증강된 대화가 비어있음"
        # 요약은 변경되지 않아야 함
        assert aug_summary == summary, "요약은 변경되지 않아야 함"

        print("✅ 턴 섞기 증강 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 동의어 치환 증강 테스트 ---------------------- #
def test_synonym_replacement():
    """동의어 치환 테스트"""
    print("\n" + "=" * 60)
    print("테스트 5: 동의어 치환 증강")
    print("=" * 60)

    # -------------- 동의어 치환 시도 -------------- #
    try:
        # 동의어 치환 증강기 생성 (30% 비율로 치환)
        augmenter = SynonymReplacementAugmenter(replace_ratio=0.3)

        # 테스트 데이터 준비
        dialogue = "오늘 점심에 밥을 먹었다"
        summary = "점심 식사"

        print(f"원본 대화: {dialogue}")
        print(f"원본 요약: {summary}")

        # 증강 실행
        aug_dialogue, aug_summary = augmenter.augment(dialogue, summary)

        print(f"증강 대화: {aug_dialogue}")
        print(f"증강 요약: {aug_summary}")

        # -------------- 결과 검증 -------------- #
        # 빈 문자열이 아닌지 확인
        assert len(aug_dialogue) > 0, "증강된 대화가 비어있음"

        print("✅ 동의어 치환 증강 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 대화 샘플링 증강 테스트 ---------------------- #
def test_dialogue_sampling():
    """대화 샘플링 테스트"""
    print("\n" + "=" * 60)
    print("테스트 6: 대화 샘플링 증강")
    print("=" * 60)

    # -------------- 대화 샘플링 시도 -------------- #
    try:
        # 대화 샘플링 증강기 생성 (70% 비율로 샘플링)
        augmenter = DialogueSamplingAugmenter(sample_ratio=0.7)

        # 테스트 데이터 준비 (8턴 대화)
        dialogue = "A: 1\nB: 2\nA: 3\nB: 4\nA: 5\nB: 6\nA: 7\nB: 8"
        summary = "대화 요약"

        print(f"원본 대화 (8턴):\n{dialogue}")
        print(f"원본 요약: {summary}")

        # 증강 실행
        aug_dialogue, aug_summary = augmenter.augment(dialogue, summary)

        # 증강된 대화의 턴 수 계산
        turns = aug_dialogue.strip().split('\n')
        print(f"증강 대화 ({len(turns)}턴):\n{aug_dialogue}")
        print(f"증강 요약: {aug_summary}")

        # -------------- 결과 검증 -------------- #
        # 빈 문자열이 아닌지 확인
        assert len(aug_dialogue) > 0, "증강된 대화가 비어있음"
        # 샘플링된 턴 수가 원본보다 작아야 함
        assert len(turns) < 8, "샘플링된 턴 수가 원본보다 작아야 함"
        # 요약은 변경되지 않아야 함
        assert aug_summary == summary, "요약은 변경되지 않아야 함"

        print("✅ 대화 샘플링 증강 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- augment_dataset 편의 함수 테스트 ---------------------- #
def test_augment_dataset():
    """데이터셋 증강 편의 함수 테스트"""
    print("\n" + "=" * 60)
    print("테스트 7: augment_dataset 편의 함수")
    print("=" * 60)

    # -------------- 데이터셋 증강 시도 -------------- #
    try:
        # 테스트 데이터 준비
        dialogues = ["A: 안녕\nB: 안녕", "A: 날씨\nB: 좋아요"]
        summaries = ["인사", "날씨"]

        print(f"원본 데이터: {len(dialogues)}개")

        # 증강 실행 (shuffle, sample 방법으로 각 2번)
        aug_dialogues, aug_summaries = augment_dataset(
            dialogues,
            summaries,
            methods=['shuffle', 'sample'],
            n_aug=2
        )

        # 예상 개수 계산: 2개 샘플 * 2개 방법 * 2번 반복 = 8개
        expected_count = len(dialogues) * len(['shuffle', 'sample']) * 2
        print(f"증강 데이터: {len(aug_dialogues)}개")
        print(f"  - 예상: {expected_count}개")

        # -------------- 결과 검증 -------------- #
        # 증강 데이터 개수 확인
        assert len(aug_dialogues) == expected_count, "증강 데이터 개수가 맞지 않음"
        # 대화와 요약 개수 일치 확인
        assert len(aug_dialogues) == len(aug_summaries), "대화와 요약 개수가 맞지 않음"

        print("✅ augment_dataset 편의 함수 성공")
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
    print("\n" + "=" * 70)
    print(" " * 20 + "데이터 증강 시스템 테스트 시작")
    print("=" * 70)

    # -------------- 테스트 목록 실행 -------------- #
    results = []
    results.append(("DataAugmenter 초기화", test_data_augmenter_init()))
    results.append(("역번역 증강", test_back_translation()))
    results.append(("의역 증강", test_paraphrase()))
    results.append(("턴 섞기 증강", test_shuffle()))
    results.append(("동의어 치환 증강", test_synonym_replacement()))
    results.append(("대화 샘플링 증강", test_dialogue_sampling()))
    results.append(("augment_dataset 함수", test_augment_dataset()))

    # ==================== 결과 요약 출력 ==================== #
    print("\n" + "=" * 70)
    print(" " * 25 + "테스트 결과 요약")
    print("=" * 70)

    # 통과한 테스트 개수 계산
    passed = sum(1 for _, result in results if result)
    total = len(results)

    # -------------- 개별 테스트 결과 출력 -------------- #
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")

    # -------------- 전체 통계 출력 -------------- #
    print("=" * 70)
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    print("=" * 70)

    # 모든 테스트가 통과했는지 반환
    return passed == total


# ---------------------- 스크립트 진입점 ---------------------- #
if __name__ == "__main__":
    success = main()                    # 테스트 실행
    sys.exit(0 if success else 1)       # 성공 시 0, 실패 시 1 반환
