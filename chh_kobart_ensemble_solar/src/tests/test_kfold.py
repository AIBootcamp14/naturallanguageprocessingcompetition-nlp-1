"""
K-Fold 교차 검증 시스템 테스트

PRD 10: 교차 검증 시스템 구현
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ------------------------- 서드파티 라이브러리 ------------------------- #
import pandas as pd

# ------------------------- 프로젝트 모듈 ------------------------- #
from src.validation.kfold import (
    KFoldSplitter,
    create_kfold_splits,
    aggregate_fold_results
)


# ==================== 테스트 함수 정의 ==================== #
# ---------------------- KFoldSplitter 초기화 테스트 ---------------------- #
def test_kfold_splitter_init():
    """KFoldSplitter 초기화 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 1: KFoldSplitter 초기화")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # KFoldSplitter 인스턴스 생성
        splitter = KFoldSplitter(n_splits=5)

        # 성공 메시지 출력
        print("✅ KFoldSplitter 초기화 성공")
        print(f"  - Fold 수: {splitter.n_splits}")
        print(f"  - Shuffle: {splitter.shuffle}")
        print(f"  - Random state: {splitter.random_state}")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 기본 K-Fold 분할 테스트 ---------------------- #
def test_basic_kfold():
    """기본 K-Fold 분할 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 2: 기본 K-Fold 분할")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # 테스트 데이터 생성
        data = pd.DataFrame({
            'dialogue': [f'대화{i}' for i in range(100)],
            'summary': [f'요약{i}' for i in range(100)]
        })

        # KFoldSplitter 생성 및 데이터 분할
        splitter = KFoldSplitter(n_splits=5, shuffle=True, random_state=42)
        folds = splitter.split(data)

        # 분할 결과 출력
        print(f"원본 데이터: {len(data)}개")
        print(f"생성된 Fold 수: {len(folds)}개")

        # -------------- Fold 정보 출력 -------------- #
        for i, (train_df, val_df) in enumerate(folds):
            print(f"  - Fold {i+1}: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        # -------------- 검증 -------------- #
        # Fold 수 검증
        assert len(folds) == 5, "Fold 수가 5개가 아님"

        # 각 Fold 데이터 크기 검증
        for train_df, val_df in folds:
            assert len(train_df) + len(val_df) == len(data), "데이터 개수가 맞지 않음"
            assert len(val_df) == 20, "검증 데이터 크기가 20이 아님"

        print("✅ 기본 K-Fold 분할 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- Stratified K-Fold 분할 테스트 ---------------------- #
def test_stratified_kfold():
    """Stratified K-Fold 분할 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 3: Stratified K-Fold 분할")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # 테스트 데이터 생성 (길이가 다른 대화)
        data = pd.DataFrame({
            'dialogue': ['짧은 대화' * i for i in range(1, 101)],
            'summary': [f'요약{i}' for i in range(100)]
        })

        # Stratified KFoldSplitter 생성
        splitter = KFoldSplitter(
            n_splits=5,
            shuffle=True,
            random_state=42,
            stratified=True
        )

        # 층화 분할 실행
        folds = splitter.split(data, stratify_column='length')

        # 분할 결과 출력
        print(f"원본 데이터: {len(data)}개")
        print(f"생성된 Fold 수: {len(folds)}개")
        print(f"층화 기준: 대화 길이 (4분위)")

        # -------------- Fold 정보 출력 -------------- #
        for i, (train_df, val_df) in enumerate(folds):
            print(f"  - Fold {i+1}: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        # -------------- 검증 -------------- #
        # Fold 수 검증
        assert len(folds) == 5, "Fold 수가 5개가 아님"

        print("✅ Stratified K-Fold 분할 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- create_kfold_splits 편의 함수 테스트 ---------------------- #
def test_create_kfold_splits():
    """create_kfold_splits 편의 함수 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 4: create_kfold_splits 편의 함수")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # 테스트 데이터 생성
        data = pd.DataFrame({
            'dialogue': [f'대화{i}' for i in range(50)],
            'summary': [f'요약{i}' for i in range(50)]
        })

        # 편의 함수로 Fold 생성
        folds = create_kfold_splits(
            data,
            n_splits=3,
            stratified=False
        )

        # 분할 결과 출력
        print(f"원본 데이터: {len(data)}개")
        print(f"생성된 Fold 수: {len(folds)}개")

        # -------------- Fold 정보 출력 -------------- #
        for i, (train_df, val_df) in enumerate(folds):
            print(f"  - Fold {i+1}: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        # -------------- 검증 -------------- #
        # Fold 수 검증
        assert len(folds) == 3, "Fold 수가 3개가 아님"

        print("✅ create_kfold_splits 편의 함수 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- aggregate_fold_results 함수 테스트 ---------------------- #
def test_aggregate_fold_results():
    """aggregate_fold_results 함수 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 5: aggregate_fold_results 함수")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Fold 결과 시뮬레이션 데이터 생성
        fold_results = [
            {'rouge1': 0.85, 'rouge2': 0.75, 'rougeL': 0.80},
            {'rouge1': 0.87, 'rouge2': 0.77, 'rougeL': 0.82},
            {'rouge1': 0.86, 'rouge2': 0.76, 'rougeL': 0.81},
            {'rouge1': 0.88, 'rouge2': 0.78, 'rougeL': 0.83},
            {'rouge1': 0.84, 'rouge2': 0.74, 'rougeL': 0.79},
        ]

        # 결과 집계 실행
        aggregated = aggregate_fold_results(fold_results)

        # -------------- 집계 결과 출력 -------------- #
        print("Fold 결과 집계:")
        print(f"  - ROUGE-1 평균: {aggregated['rouge1_mean']:.4f} (±{aggregated['rouge1_std']:.4f})")
        print(f"  - ROUGE-2 평균: {aggregated['rouge2_mean']:.4f} (±{aggregated['rouge2_std']:.4f})")
        print(f"  - ROUGE-L 평균: {aggregated['rougeL_mean']:.4f} (±{aggregated['rougeL_std']:.4f})")

        # -------------- 검증 -------------- #
        # 필수 키 존재 확인
        assert 'rouge1_mean' in aggregated, "rouge1_mean이 없음"
        assert 'rouge1_std' in aggregated, "rouge1_std가 없음"
        assert 'rouge1_min' in aggregated, "rouge1_min이 없음"
        assert 'rouge1_max' in aggregated, "rouge1_max가 없음"

        # 평균값 범위 검증
        assert 0.85 <= aggregated['rouge1_mean'] <= 0.87, "ROUGE-1 평균이 범위를 벗어남"

        print("✅ aggregate_fold_results 함수 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- Fold 데이터 무결성 테스트 ---------------------- #
def test_fold_data_integrity():
    """Fold 데이터 무결성 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 6: Fold 데이터 무결성")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # 테스트 데이터 생성
        data = pd.DataFrame({
            'dialogue': [f'대화{i}' for i in range(100)],
            'summary': [f'요약{i}' for i in range(100)]
        })

        # KFoldSplitter 생성 및 데이터 분할
        splitter = KFoldSplitter(n_splits=5, shuffle=False, random_state=None)
        folds = splitter.split(data)

        # 모든 검증 데이터 수집용 리스트
        all_val_data = []

        # -------------- 각 Fold 무결성 검증 -------------- #
        for fold_idx, (train_df, val_df) in enumerate(folds):
            # 학습/검증 데이터 대화 텍스트 추출
            train_dialogues = set(train_df['dialogue'].tolist())
            val_dialogues = set(val_df['dialogue'].tolist())

            # 학습/검증 데이터 중복 확인
            overlap = train_dialogues & val_dialogues
            assert len(overlap) == 0, f"Fold {fold_idx+1}: 학습/검증 데이터가 겹침: {len(overlap)}개"

            # 검증 데이터 수집
            all_val_data.extend(val_df['dialogue'].tolist())

        # -------------- 전체 데이터 무결성 검증 -------------- #
        # 모든 검증 데이터 합이 전체 데이터와 같은지 확인
        assert len(all_val_data) == len(data), "검증 데이터 합이 전체와 다름"
        # 중복 데이터가 없는지 확인
        assert len(set(all_val_data)) == len(data), "중복된 검증 데이터 존재"

        print("✅ Fold 데이터 무결성 검증 성공")
        print(f"  - 학습/검증 데이터 중복 없음")
        print(f"  - 모든 데이터가 검증에 한 번씩 사용됨")
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
    print(" " * 20 + "K-Fold 교차 검증 시스템 테스트 시작")
    print("=" * 70)

    # -------------- 테스트 실행 -------------- #
    # 테스트 결과 수집용 리스트
    results = []
    results.append(("KFoldSplitter 초기화", test_kfold_splitter_init()))
    results.append(("기본 K-Fold 분할", test_basic_kfold()))
    results.append(("Stratified K-Fold 분할", test_stratified_kfold()))
    results.append(("create_kfold_splits 함수", test_create_kfold_splits()))
    results.append(("aggregate_fold_results 함수", test_aggregate_fold_results()))
    results.append(("Fold 데이터 무결성", test_fold_data_integrity()))

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

    # 전체 테스트 통과 여부 반환
    return passed == total


# ---------------------- 메인 진입점 ---------------------- #
if __name__ == "__main__":
    success = main()  # 전체 테스트 실행
    sys.exit(0 if success else 1)  # 성공 시 0, 실패 시 1 반환
