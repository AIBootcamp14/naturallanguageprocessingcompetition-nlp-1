"""
K-Fold 교차 검증 시스템 테스트

PRD 10: 교차 검증 시스템 구현
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.validation.kfold import (
    KFoldSplitter,
    create_kfold_splits,
    aggregate_fold_results
)


def test_kfold_splitter_init():
    """KFoldSplitter 초기화 테스트"""
    print("\n" + "=" * 60)
    print("테스트 1: KFoldSplitter 초기화")
    print("=" * 60)

    try:
        splitter = KFoldSplitter(n_splits=5)
        print("✅ KFoldSplitter 초기화 성공")
        print(f"  - Fold 수: {splitter.n_splits}")
        print(f"  - Shuffle: {splitter.shuffle}")
        print(f"  - Random state: {splitter.random_state}")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_kfold():
    """기본 K-Fold 분할 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 기본 K-Fold 분할")
    print("=" * 60)

    try:
        # 테스트 데이터 생성
        data = pd.DataFrame({
            'dialogue': [f'대화{i}' for i in range(100)],
            'summary': [f'요약{i}' for i in range(100)]
        })

        splitter = KFoldSplitter(n_splits=5, shuffle=True, random_state=42)
        folds = splitter.split(data)

        print(f"원본 데이터: {len(data)}개")
        print(f"생성된 Fold 수: {len(folds)}개")

        for i, (train_df, val_df) in enumerate(folds):
            print(f"  - Fold {i+1}: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        # 검증
        assert len(folds) == 5, "Fold 수가 5개가 아님"

        for train_df, val_df in folds:
            assert len(train_df) + len(val_df) == len(data), "데이터 개수가 맞지 않음"
            assert len(val_df) == 20, "검증 데이터 크기가 20이 아님"

        print("✅ 기본 K-Fold 분할 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_stratified_kfold():
    """Stratified K-Fold 분할 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: Stratified K-Fold 분할")
    print("=" * 60)

    try:
        # 테스트 데이터 생성 (길이가 다른 대화)
        data = pd.DataFrame({
            'dialogue': ['짧은 대화' * i for i in range(1, 101)],
            'summary': [f'요약{i}' for i in range(100)]
        })

        splitter = KFoldSplitter(
            n_splits=5,
            shuffle=True,
            random_state=42,
            stratified=True
        )

        folds = splitter.split(data, stratify_column='length')

        print(f"원본 데이터: {len(data)}개")
        print(f"생성된 Fold 수: {len(folds)}개")
        print(f"층화 기준: 대화 길이 (4분위)")

        for i, (train_df, val_df) in enumerate(folds):
            print(f"  - Fold {i+1}: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        # 검증
        assert len(folds) == 5, "Fold 수가 5개가 아님"

        print("✅ Stratified K-Fold 분할 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_create_kfold_splits():
    """create_kfold_splits 편의 함수 테스트"""
    print("\n" + "=" * 60)
    print("테스트 4: create_kfold_splits 편의 함수")
    print("=" * 60)

    try:
        data = pd.DataFrame({
            'dialogue': [f'대화{i}' for i in range(50)],
            'summary': [f'요약{i}' for i in range(50)]
        })

        folds = create_kfold_splits(
            data,
            n_splits=3,
            stratified=False
        )

        print(f"원본 데이터: {len(data)}개")
        print(f"생성된 Fold 수: {len(folds)}개")

        for i, (train_df, val_df) in enumerate(folds):
            print(f"  - Fold {i+1}: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        assert len(folds) == 3, "Fold 수가 3개가 아님"

        print("✅ create_kfold_splits 편의 함수 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregate_fold_results():
    """aggregate_fold_results 함수 테스트"""
    print("\n" + "=" * 60)
    print("테스트 5: aggregate_fold_results 함수")
    print("=" * 60)

    try:
        # Fold 결과 시뮬레이션
        fold_results = [
            {'rouge1': 0.85, 'rouge2': 0.75, 'rougeL': 0.80},
            {'rouge1': 0.87, 'rouge2': 0.77, 'rougeL': 0.82},
            {'rouge1': 0.86, 'rouge2': 0.76, 'rougeL': 0.81},
            {'rouge1': 0.88, 'rouge2': 0.78, 'rougeL': 0.83},
            {'rouge1': 0.84, 'rouge2': 0.74, 'rougeL': 0.79},
        ]

        aggregated = aggregate_fold_results(fold_results)

        print("Fold 결과 집계:")
        print(f"  - ROUGE-1 평균: {aggregated['rouge1_mean']:.4f} (±{aggregated['rouge1_std']:.4f})")
        print(f"  - ROUGE-2 평균: {aggregated['rouge2_mean']:.4f} (±{aggregated['rouge2_std']:.4f})")
        print(f"  - ROUGE-L 평균: {aggregated['rougeL_mean']:.4f} (±{aggregated['rougeL_std']:.4f})")

        # 검증
        assert 'rouge1_mean' in aggregated, "rouge1_mean이 없음"
        assert 'rouge1_std' in aggregated, "rouge1_std가 없음"
        assert 'rouge1_min' in aggregated, "rouge1_min이 없음"
        assert 'rouge1_max' in aggregated, "rouge1_max가 없음"

        assert 0.85 <= aggregated['rouge1_mean'] <= 0.87, "ROUGE-1 평균이 범위를 벗어남"

        print("✅ aggregate_fold_results 함수 성공")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_fold_data_integrity():
    """Fold 데이터 무결성 테스트"""
    print("\n" + "=" * 60)
    print("테스트 6: Fold 데이터 무결성")
    print("=" * 60)

    try:
        data = pd.DataFrame({
            'dialogue': [f'대화{i}' for i in range(100)],
            'summary': [f'요약{i}' for i in range(100)]
        })

        splitter = KFoldSplitter(n_splits=5, shuffle=False, random_state=None)
        folds = splitter.split(data)

        # 모든 fold의 검증 데이터를 합치면 원본 데이터와 같아야 함
        all_val_data = []

        for fold_idx, (train_df, val_df) in enumerate(folds):
            # 학습/검증 데이터가 겹치지 않아야 함 (dialogue 기준으로 체크)
            train_dialogues = set(train_df['dialogue'].tolist())
            val_dialogues = set(val_df['dialogue'].tolist())

            overlap = train_dialogues & val_dialogues
            assert len(overlap) == 0, f"Fold {fold_idx+1}: 학습/검증 데이터가 겹침: {len(overlap)}개"

            all_val_data.extend(val_df['dialogue'].tolist())

        # 모든 검증 데이터를 합치면 전체 데이터와 같아야 함
        assert len(all_val_data) == len(data), "검증 데이터 합이 전체와 다름"
        assert len(set(all_val_data)) == len(data), "중복된 검증 데이터 존재"

        print("✅ Fold 데이터 무결성 검증 성공")
        print(f"  - 학습/검증 데이터 중복 없음")
        print(f"  - 모든 데이터가 검증에 한 번씩 사용됨")
        return True

    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 70)
    print(" " * 20 + "K-Fold 교차 검증 시스템 테스트 시작")
    print("=" * 70)

    results = []
    results.append(("KFoldSplitter 초기화", test_kfold_splitter_init()))
    results.append(("기본 K-Fold 분할", test_basic_kfold()))
    results.append(("Stratified K-Fold 분할", test_stratified_kfold()))
    results.append(("create_kfold_splits 함수", test_create_kfold_splits()))
    results.append(("aggregate_fold_results 함수", test_aggregate_fold_results()))
    results.append(("Fold 데이터 무결성", test_fold_data_integrity()))

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
