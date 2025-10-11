# ==================== Metrics 테스트 스크립트 ==================== #
"""
평가 지표 시스템 테스트

테스트 항목:
1. 단일 샘플 ROUGE 계산
2. Multi-reference ROUGE 계산
3. 배치 ROUGE 계산
4. ROUGE Sum 계산
5. 점수 포맷팅
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.evaluation import RougeCalculator, calculate_rouge_scores
from src.evaluation.metrics import format_rouge_scores


# ==================== 테스트 함수들 ==================== #
# ---------------------- 단일 샘플 ROUGE 테스트 ---------------------- #
def test_single_rouge():
    """단일 샘플 ROUGE 계산 테스트"""
    print("\n" + "="*60)
    print("테스트 1: 단일 샘플 ROUGE 계산")
    print("="*60)

    # ROUGE Calculator 생성
    calculator = RougeCalculator()                      # 계산기 초기화

    # 테스트 데이터
    prediction = "두 사람이 인사를 나누었다"
    reference = "두 사람이 서로 인사를 하였다"

    # ROUGE 계산
    scores = calculator.calculate_single(prediction, reference)  # ROUGE 계산

    # 결과 출력
    print(f"\n  예측: {prediction}")
    print(f"  정답: {reference}")
    print(f"\n  ROUGE-1 F1: {scores['rouge1']['fmeasure']:.4f}")
    print(f"  ROUGE-2 F1: {scores['rouge2']['fmeasure']:.4f}")
    print(f"  ROUGE-L F1: {scores['rougeL']['fmeasure']:.4f}")

    # 검증
    assert 'rouge1' in scores                           # ROUGE-1 존재 확인
    assert 'rouge2' in scores                           # ROUGE-2 존재 확인
    assert 'rougeL' in scores                           # ROUGE-L 존재 확인
    assert 0 <= scores['rouge1']['fmeasure'] <= 1      # F1 점수 범위 확인

    print("\n✅ 단일 샘플 ROUGE 계산 테스트 성공!")


# ---------------------- Multi-reference ROUGE 테스트 ---------------------- #
def test_multi_reference_rouge():
    """Multi-reference ROUGE 계산 테스트"""
    print("\n" + "="*60)
    print("테스트 2: Multi-reference ROUGE 계산")
    print("="*60)

    # ROUGE Calculator 생성
    calculator = RougeCalculator()                      # 계산기 초기화

    # 테스트 데이터 (다중 정답)
    prediction = "두 사람이 인사를 나누었다"
    references = [
        "두 사람이 서로 인사를 하였다",
        "두 사람이 만나 인사했다",
        "사람들이 서로 인사를 주고받았다"
    ]

    # ROUGE 계산 (최대값 선택)
    scores = calculator.calculate_single(prediction, references)  # Multi-ref ROUGE

    # 결과 출력
    print(f"\n  예측: {prediction}")
    print(f"  정답 개수: {len(references)}")
    print(f"\n  ROUGE-1 F1: {scores['rouge1']['fmeasure']:.4f}")
    print(f"  ROUGE-2 F1: {scores['rouge2']['fmeasure']:.4f}")
    print(f"  ROUGE-L F1: {scores['rougeL']['fmeasure']:.4f}")

    # 검증
    assert 'rouge1' in scores                           # ROUGE-1 존재 확인
    assert 0 <= scores['rouge1']['fmeasure'] <= 1      # F1 점수 범위 확인

    print("\n✅ Multi-reference ROUGE 계산 테스트 성공!")


# ---------------------- 배치 ROUGE 테스트 ---------------------- #
def test_batch_rouge():
    """배치 ROUGE 계산 테스트"""
    print("\n" + "="*60)
    print("테스트 3: 배치 ROUGE 계산")
    print("="*60)

    # ROUGE Calculator 생성
    calculator = RougeCalculator()                      # 계산기 초기화

    # 테스트 데이터
    predictions = [
        "두 사람이 인사를 나누었다",
        "날씨가 좋아서 산책을 했다",
        "저녁 식사로 피자를 먹었다"
    ]
    references = [
        "두 사람이 서로 인사를 하였다",
        "날씨가 좋아 밖에서 걸었다",
        "저녁에 피자를 주문해서 먹었다"
    ]

    # 배치 ROUGE 계산
    scores = calculator.calculate_batch(predictions, references)  # 배치 계산

    # 결과 출력
    print(f"\n  샘플 개수: {len(predictions)}")
    print(f"\n  ROUGE-1 F1: {scores['rouge1']['fmeasure']:.4f} (±{scores['rouge1']['std']:.4f})")
    print(f"  ROUGE-2 F1: {scores['rouge2']['fmeasure']:.4f} (±{scores['rouge2']['std']:.4f})")
    print(f"  ROUGE-L F1: {scores['rougeL']['fmeasure']:.4f} (±{scores['rougeL']['std']:.4f})")

    # 검증
    assert 'rouge1' in scores                           # ROUGE-1 존재 확인
    assert 'std' in scores['rouge1']                    # 표준편차 존재 확인
    assert 'min' in scores['rouge1']                    # 최소값 존재 확인
    assert 'max' in scores['rouge1']                    # 최대값 존재 확인

    print("\n✅ 배치 ROUGE 계산 테스트 성공!")


# ---------------------- ROUGE Sum 테스트 ---------------------- #
def test_rouge_sum():
    """ROUGE Sum 계산 테스트"""
    print("\n" + "="*60)
    print("테스트 4: ROUGE Sum 계산")
    print("="*60)

    # ROUGE Calculator 생성
    calculator = RougeCalculator()                      # 계산기 초기화

    # 테스트 데이터
    predictions = [
        "두 사람이 인사를 나누었다",
        "날씨가 좋아서 산책을 했다"
    ]
    references = [
        "두 사람이 서로 인사를 하였다",
        "날씨가 좋아 밖에서 걸었다"
    ]

    # 배치 ROUGE 계산
    scores = calculator.calculate_batch(predictions, references)  # 배치 계산

    # ROUGE Sum 확인
    rouge_sum = scores['rouge_sum']['fmeasure']         # ROUGE Sum 추출

    # 수동 계산으로 검증
    manual_sum = (
        scores['rouge1']['fmeasure'] +
        scores['rouge2']['fmeasure'] +
        scores['rougeL']['fmeasure']
    )

    # 결과 출력
    print(f"\n  ROUGE-1 F1: {scores['rouge1']['fmeasure']:.4f}")
    print(f"  ROUGE-2 F1: {scores['rouge2']['fmeasure']:.4f}")
    print(f"  ROUGE-L F1: {scores['rougeL']['fmeasure']:.4f}")
    print(f"  ROUGE Sum: {rouge_sum:.4f}")

    # 검증
    assert 'rouge_sum' in scores                        # ROUGE Sum 존재 확인
    assert abs(rouge_sum - manual_sum) < 1e-6           # 수동 계산과 일치 확인

    print("\n✅ ROUGE Sum 계산 테스트 성공!")


# ---------------------- 편의 함수 테스트 ---------------------- #
def test_convenience_function():
    """편의 함수 테스트"""
    print("\n" + "="*60)
    print("테스트 5: 편의 함수")
    print("="*60)

    # 단일 샘플 테스트
    prediction = "두 사람이 인사를 나누었다"
    reference = "두 사람이 서로 인사를 하였다"

    # 편의 함수로 ROUGE 계산
    scores = calculate_rouge_scores(prediction, reference)  # 편의 함수 사용

    # 결과 출력
    print(f"\n  편의 함수 사용:")
    print(f"  ROUGE-1 F1: {scores['rouge1']['fmeasure']:.4f}")

    # 검증
    assert 'rouge1' in scores                           # ROUGE-1 존재 확인

    print("\n✅ 편의 함수 테스트 성공!")


# ---------------------- 포맷팅 테스트 ---------------------- #
def test_formatting():
    """점수 포맷팅 테스트"""
    print("\n" + "="*60)
    print("테스트 6: 점수 포맷팅")
    print("="*60)

    # ROUGE 계산
    prediction = "두 사람이 인사를 나누었다"
    reference = "두 사람이 서로 인사를 하였다"

    scores = calculate_rouge_scores(prediction, reference)  # ROUGE 계산

    # 포맷팅
    formatted = format_rouge_scores(scores, decimal_places=4)  # 포맷팅

    # 결과 출력
    print("\n포맷팅된 결과:")
    print(formatted)

    # 검증
    assert isinstance(formatted, str)                   # 문자열 타입 확인
    assert 'ROUGE1' in formatted                        # ROUGE1 포함 확인

    print("✅ 점수 포맷팅 테스트 성공!")


# ==================== 메인 실행부 ==================== #
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Metrics 테스트 시작")
    print("="*60)

    try:
        # 모든 테스트 실행
        test_single_rouge()                             # 테스트 1
        test_multi_reference_rouge()                    # 테스트 2
        test_batch_rouge()                              # 테스트 3
        test_rouge_sum()                                # 테스트 4
        test_convenience_function()                     # 테스트 5
        test_formatting()                               # 테스트 6

        # 최종 결과
        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 테스트 실패: {e}")
        print("="*60)
        raise
