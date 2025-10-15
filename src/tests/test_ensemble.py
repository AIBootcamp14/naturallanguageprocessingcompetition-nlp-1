"""
앙상블 시스템 테스트

주요 기능:
- PRD 12: 다중 모델 앙상블 전략 구현 검증
- 가중치 앙상블, 투표 앙상블, 모델 관리자 기능 테스트
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
from pathlib import Path

# ---------------------- 프로젝트 루트 경로 설정 ---------------------- #
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 앙상블 모듈 ---------------------- #
from src.ensemble import (
    WeightedEnsemble,
    VotingEnsemble,
    ModelManager
)


# ==================== 테스트 함수들 ==================== #
# ---------------------- ModelManager 초기화 테스트 ---------------------- #
def test_model_manager_init():
    """ModelManager 초기화 테스트"""
    print("\n" + "=" * 60)
    print("테스트 1: ModelManager 초기화")
    print("=" * 60)

    # -------------- 초기화 시도 -------------- #
    try:
        # ModelManager 인스턴스 생성
        manager = ModelManager()
        print("✅ ModelManager 초기화 성공")
        print(f"  - 로드된 모델 수: {len(manager.models)}")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- ModelManager 정보 조회 테스트 ---------------------- #
def test_model_manager_info():
    """ModelManager 정보 조회 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: ModelManager 정보 조회")
    print("=" * 60)

    # -------------- 정보 조회 시도 -------------- #
    try:
        # ModelManager 인스턴스 생성
        manager = ModelManager()
        info = manager.get_info()       # 정보 조회

        print("✅ 정보 조회 성공")
        print(f"  - 모델 수: {info['num_models']}")
        print(f"  - 모델 이름: {info['model_names']}")

        # -------------- 결과 검증 -------------- #
        # 필수 키 존재 확인
        assert 'num_models' in info, "num_models 키가 없음"
        assert 'model_names' in info, "model_names 키가 없음"

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==================== Mock 클래스 정의 ==================== #
# ---------------------- Mock 모델 클래스 ---------------------- #
class MockModel:
    """테스트용 Mock 모델"""

    def parameters(self):
        """빈 파라미터 반환"""
        return iter([])


# ---------------------- Mock 토크나이저 클래스 ---------------------- #
class MockTokenizer:
    """테스트용 Mock 토크나이저"""
    pass


# ==================== WeightedEnsemble 테스트 ==================== #
# ---------------------- WeightedEnsemble 초기화 테스트 ---------------------- #
def test_weighted_ensemble_init():
    """WeightedEnsemble 초기화 테스트 (Mock)"""
    print("\n" + "=" * 60)
    print("테스트 3: WeightedEnsemble 초기화 (Mock)")
    print("=" * 60)

    # -------------- 초기화 시도 -------------- #
    try:
        # Mock 모델과 토크나이저 생성
        models = [MockModel(), MockModel()]
        tokenizers = [MockTokenizer(), MockTokenizer()]
        weights = [0.6, 0.4]            # 가중치 설정

        # WeightedEnsemble 인스턴스 생성
        ensemble = WeightedEnsemble(models, tokenizers, weights)

        print("✅ WeightedEnsemble 초기화 성공")
        print(f"  - 모델 수: {len(ensemble.models)}")
        print(f"  - 가중치: {ensemble.weights}")

        # -------------- 결과 검증 -------------- #
        # 모델 수 확인
        assert len(ensemble.models) == 2, "모델 수가 맞지 않음"
        # 가중치 합이 1인지 확인
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6, "가중치 합이 1이 아님"

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- WeightedEnsemble 균등 가중치 테스트 ---------------------- #
def test_weighted_ensemble_equal_weights():
    """WeightedEnsemble 균등 가중치 테스트"""
    print("\n" + "=" * 60)
    print("테스트 4: WeightedEnsemble 균등 가중치")
    print("=" * 60)

    # -------------- 균등 가중치 초기화 시도 -------------- #
    try:
        # Mock 모델과 토크나이저 생성 (3개)
        models = [MockModel(), MockModel(), MockModel()]
        tokenizers = [MockTokenizer(), MockTokenizer(), MockTokenizer()]

        # 가중치 없이 초기화 (자동으로 균등 가중치 설정)
        ensemble = WeightedEnsemble(models, tokenizers)

        print("✅ 균등 가중치 초기화 성공")
        print(f"  - 가중치: {ensemble.weights}")

        # -------------- 결과 검증 -------------- #
        # 각 가중치가 1/3인지 확인
        expected_weight = 1.0 / 3.0
        for weight in ensemble.weights:
            assert abs(weight - expected_weight) < 1e-6, "균등 가중치가 아님"

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==================== VotingEnsemble 테스트 ==================== #
# ---------------------- VotingEnsemble 초기화 테스트 ---------------------- #
def test_voting_ensemble_init():
    """VotingEnsemble 초기화 테스트 (Mock)"""
    print("\n" + "=" * 60)
    print("테스트 5: VotingEnsemble 초기화 (Mock)")
    print("=" * 60)

    # -------------- 초기화 시도 -------------- #
    try:
        # Mock 모델과 토크나이저 생성
        models = [MockModel(), MockModel()]
        tokenizers = [MockTokenizer(), MockTokenizer()]

        # VotingEnsemble 인스턴스 생성 (Hard Voting)
        ensemble = VotingEnsemble(models, tokenizers, voting="hard")

        print("✅ VotingEnsemble 초기화 성공")
        print(f"  - 모델 수: {len(ensemble.models)}")
        print(f"  - 투표 방식: {ensemble.voting}")

        # -------------- 결과 검증 -------------- #
        # 모델 수 확인
        assert len(ensemble.models) == 2, "모델 수가 맞지 않음"
        # 투표 방식 확인
        assert ensemble.voting == "hard", "투표 방식이 맞지 않음"

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- VotingEnsemble Hard Voting 로직 테스트 ---------------------- #
def test_voting_hard_voting():
    """VotingEnsemble Hard Voting 로직 테스트"""
    print("\n" + "=" * 60)
    print("테스트 6: VotingEnsemble Hard Voting 로직")
    print("=" * 60)

    # -------------- Hard Voting 로직 시도 -------------- #
    try:
        # Mock 모델과 토크나이저 생성 (3개)
        models = [MockModel(), MockModel(), MockModel()]
        tokenizers = [MockTokenizer(), MockTokenizer(), MockTokenizer()]

        # VotingEnsemble 인스턴스 생성
        ensemble = VotingEnsemble(models, tokenizers, voting="hard")

        # -------------- Mock 예측 결과 준비 -------------- #
        # 3개 모델의 3개 샘플에 대한 예측
        all_predictions = [
            ["요약A", "요약B", "요약C"],  # 모델 1 예측
            ["요약A", "요약B", "요약D"],  # 모델 2 예측
            ["요약A", "요약E", "요약C"],  # 모델 3 예측
        ]

        # -------------- Hard Voting 실행 -------------- #
        result = ensemble._hard_voting(all_predictions)

        print("✅ Hard Voting 로직 성공")
        print(f"  - 결과: {result}")

        # -------------- 결과 검증 -------------- #
        # 샘플 1: 모든 모델이 "요약A" 예측 → "요약A" 선택
        assert result[0] == "요약A", "샘플 1 투표 결과가 맞지 않음"

        # 샘플 2: 2개 모델이 "요약B" 예측 → "요약B" 선택
        assert result[1] == "요약B", "샘플 2 투표 결과가 맞지 않음"

        # 샘플 3: 2개 모델이 "요약C" 예측 → "요약C" 선택
        assert result[2] == "요약C", "샘플 3 투표 결과가 맞지 않음"

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
    print(" " * 25 + "앙상블 시스템 테스트 시작")
    print("=" * 70)

    # -------------- 테스트 목록 실행 -------------- #
    results = []
    results.append(("ModelManager 초기화", test_model_manager_init()))
    results.append(("ModelManager 정보 조회", test_model_manager_info()))
    results.append(("WeightedEnsemble 초기화", test_weighted_ensemble_init()))
    results.append(("WeightedEnsemble 균등 가중치", test_weighted_ensemble_equal_weights()))
    results.append(("VotingEnsemble 초기화", test_voting_ensemble_init()))
    results.append(("VotingEnsemble Hard Voting", test_voting_hard_voting()))

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
