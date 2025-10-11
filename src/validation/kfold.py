"""
K-Fold 교차 검증 시스템

PRD 10: 교차 검증 시스템 구현
- K-Fold 분할
- Stratified K-Fold 지원
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List, Tuple, Dict, Any


class KFoldSplitter:
    """K-Fold 교차 검증 분할기"""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        stratified: bool = False,
        logger=None
    ):
        """
        Args:
            n_splits: Fold 개수
            shuffle: 데이터 셔플 여부
            random_state: 랜덤 시드
            stratified: 층화 추출 여부
            logger: Logger 인스턴스
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified
        self.logger = logger

        # KFold 객체 생성
        if stratified:
            self.kfold = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
            self._log(f"StratifiedKFold 초기화 (n_splits={n_splits})")
        else:
            self.kfold = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
            self._log(f"KFold 초기화 (n_splits={n_splits})")

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def split(
        self,
        data: pd.DataFrame,
        stratify_column: str = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        데이터를 K개 fold로 분할

        Args:
            data: 원본 데이터프레임
            stratify_column: 층화 추출 기준 컬럼 (stratified=True일 때만)

        Returns:
            [(train_df, val_df), ...] 리스트 (K개)
        """
        self._log(f"\n데이터 분할 시작")
        self._log(f"  - 전체 데이터: {len(data)}개")
        self._log(f"  - Fold 수: {self.n_splits}")

        X = data.index.values
        y = None

        # Stratified 설정
        if self.stratified and stratify_column:
            # 길이 기반 stratification (예: 대화 길이)
            if stratify_column == 'length':
                lengths = data['dialogue'].str.len()
                # 4분위로 나눔
                y = pd.qcut(lengths, q=4, labels=False, duplicates='drop')
                self._log(f"  - 층화 기준: 대화 길이 (4분위)")
            # 토픽 기반 stratification
            elif stratify_column in data.columns:
                y = data[stratify_column]
                self._log(f"  - 층화 기준: {stratify_column}")
            else:
                self._log(f"  - 경고: {stratify_column} 컬럼 없음, 일반 KFold 사용")

        # Fold 분할
        folds = []
        for fold_idx, (train_indices, val_indices) in enumerate(self.kfold.split(X, y)):
            train_df = data.iloc[train_indices].reset_index(drop=True)
            val_df = data.iloc[val_indices].reset_index(drop=True)

            folds.append((train_df, val_df))

            self._log(f"  - Fold {fold_idx + 1}: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

        self._log(f"데이터 분할 완료")
        return folds


def create_kfold_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    stratified: bool = False,
    stratify_column: str = None,
    logger=None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    편의 함수: K-Fold 분할

    Args:
        data: 원본 데이터프레임
        n_splits: Fold 개수
        stratified: 층화 추출 여부
        stratify_column: 층화 기준 컬럼
        logger: Logger 인스턴스

    Returns:
        [(train_df, val_df), ...] 리스트
    """
    splitter = KFoldSplitter(
        n_splits=n_splits,
        stratified=stratified,
        logger=logger
    )
    return splitter.split(data, stratify_column=stratify_column)


def aggregate_fold_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fold 결과 집계

    Args:
        fold_results: Fold별 결과 리스트
                      [{'rouge1': 0.5, 'rouge2': 0.3, ...}, ...]

    Returns:
        집계 결과 (평균, 표준편차, 최소, 최대)
    """
    if not fold_results:
        return {}

    # 모든 메트릭 키 추출
    metric_keys = list(fold_results[0].keys())

    aggregated = {}
    for key in metric_keys:
        values = [fold[key] for fold in fold_results if key in fold]

        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)

    return aggregated
