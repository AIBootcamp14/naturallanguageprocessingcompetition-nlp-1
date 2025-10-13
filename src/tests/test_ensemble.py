"""
앙상블 시스템 테스트

PRD 12: 다중 모델 앙상블 전략 구현
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ensemble import (
    WeightedEnsemble,
    VotingEnsemble,
    ModelManager
)


def test_model_manager_init():
    """ModelManager 초기화 테스트"""
    print("\n" + "=" * 60)
    print("테스트 1: ModelManager 초기화")
    print("=" * 60)

    try:
        manager = ModelManager()
        print("✅ ModelManager 초기화 성공")
        print(f"  - 로드된 모델 수: {len(manager.models)}")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_manager_info():
    """ModelManager 정보 조회 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: ModelManager 정보 조회")
    print("=" * 60)

    try:
        manager = ModelManager()
        info = manager.get_info()

        print("✅ 정보 조회 성공")
        print(f"  - 모델 수: {info['num_models']}")
        print(f"  - 모델 이름: {info['model_names']}")

        assert 'num_models' in info, "num_models 키가 없음"
        assert 'model_names' in info, "model_names 키가 없음"

        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_weighted_ensemble_init():
    """WeightedEnsemble 초기화 테스트 (Mock)"""
    print("\n" + "=" * 60)
    print("테스트 3: WeightedEnsemble 초기화 (Mock)")
    print("=" * 60)

    try:
        # Mock 모델과 토크나이저
        class MockModel:
            def parameters(self):
                return iter([])

        class MockTokenizer:
            pass

        models = [MockModel(), MockModel()]
        tokenizers = [MockTokenizer(), MockTokenizer()]
        weights = [0.6, 0.4]

        ensemble = WeightedEnsemble(models, tokenizers, weights)

        print("✅ WeightedEnsemble 초기화 성공")
        print(f"  - 모델 수: {len(ensemble.models)}")
        print(f"  - 가중치: {ensemble.weights}")

        assert len(ensemble.models) == 2, "모델 수가 맞지 않음"
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6, "가중치 합이 1이 아님"

        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_weighted_ensemble_equal_weights():
    """WeightedEnsemble 균등 가중치 테스트"""
    print("\n" + "=" * 60)
    print("테스트 4: WeightedEnsemble 균등 가중치")
    print("=" * 60)

    try:
        class MockModel:
            def parameters(self):
                return iter([])

        class MockTokenizer:
            pass

        models = [MockModel(), MockModel(), MockModel()]
        tokenizers = [MockTokenizer(), MockTokenizer(), MockTokenizer()]

        # 가중치 없이 초기화 (균등 가중치)
        ensemble = WeightedEnsemble(models, tokenizers)

        print("✅ 균등 가중치 초기화 성공")
        print(f"  - 가중치: {ensemble.weights}")

        expected_weight = 1.0 / 3.0
        for weight in ensemble.weights:
            assert abs(weight - expected_weight) < 1e-6, "균등 가중치가 아님"

        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_voting_ensemble_init():
    """VotingEnsemble 초기화 테스트 (Mock)"""
    print("\n" + "=" * 60)
    print("테스트 5: VotingEnsemble 초기화 (Mock)")
    print("=" * 60)

    try:
        class MockModel:
            def parameters(self):
                return iter([])

        class MockTokenizer:
            pass

        models = [MockModel(), MockModel()]
        tokenizers = [MockTokenizer(), MockTokenizer()]

        ensemble = VotingEnsemble(models, tokenizers, voting="hard")

        print("✅ VotingEnsemble 초기화 성공")
        print(f"  - 모델 수: {len(ensemble.models)}")
        print(f"  - 투표 방식: {ensemble.voting}")

        assert len(ensemble.models) == 2, "모델 수가 맞지 않음"
        assert ensemble.voting == "hard", "투표 방식이 맞지 않음"

        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_voting_hard_voting():
    """VotingEnsemble Hard Voting 로직 테스트"""
    print("\n" + "=" * 60)
    print("테스트 6: VotingEnsemble Hard Voting 로직")
    print("=" * 60)

    try:
        class MockModel:
            def parameters(self):
                return iter([])

        class MockTokenizer:
            pass

        models = [MockModel(), MockModel(), MockModel()]
        tokenizers = [MockTokenizer(), MockTokenizer(), MockTokenizer()]

        ensemble = VotingEnsemble(models, tokenizers, voting="hard")

        # Mock 예측 결과
        all_predictions = [
            ["요약A", "요약B", "요약C"],  # 모델 1
            ["요약A", "요약B", "요약D"],  # 모델 2
            ["요약A", "요약E", "요약C"],  # 모델 3
        ]

        result = ensemble._hard_voting(all_predictions)

        print("✅ Hard Voting 로직 성공")
        print(f"  - 결과: {result}")

        # 샘플 1: 모든 모델이 "요약A" → "요약A"
        assert result[0] == "요약A", "샘플 1 투표 결과가 맞지 않음"

        # 샘플 2: 2개 모델이 "요약B" → "요약B"
        assert result[1] == "요약B", "샘플 2 투표 결과가 맞지 않음"

        # 샘플 3: 2개 모델이 "요약C" → "요약C"
        assert result[2] == "요약C", "샘플 3 투표 결과가 맞지 않음"

        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 70)
    print(" " * 25 + "앙상블 시스템 테스트 시작")
    print("=" * 70)

    results = []
    results.append(("ModelManager 초기화", test_model_manager_init()))
    results.append(("ModelManager 정보 조회", test_model_manager_info()))
    results.append(("WeightedEnsemble 초기화", test_weighted_ensemble_init()))
    results.append(("WeightedEnsemble 균등 가중치", test_weighted_ensemble_equal_weights()))
    results.append(("VotingEnsemble 초기화", test_voting_ensemble_init()))
    results.append(("VotingEnsemble Hard Voting", test_voting_hard_voting()))

    # 결과 요약
    print("\n" + "=" * 70)
    print(" " * 25 + "테스트 결과 요약")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")

    print("=" * 70)
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
