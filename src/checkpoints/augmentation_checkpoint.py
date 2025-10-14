"""
데이터 증강 체크포인트 관리자

데이터 증강 진행 상황 저장 및 복원
"""

from typing import Optional, Dict, List, Any
from pathlib import Path
import pandas as pd

from src.checkpoints.base_checkpoint import BaseCheckpointManager


class AugmentationCheckpointManager(BaseCheckpointManager):
    """
    데이터 증강 체크포인트 관리자

    기능:
    1. 증강 진행 중 주기적으로 저장 (배치 단위)
    2. 중단 후 Resume 시 완료된 증강 데이터 복원
    3. 진행률 추적

    Usage:
        checkpoint_mgr = AugmentationCheckpointManager(
            checkpoint_dir="experiments/.../checkpoints"
        )

        # 체크포인트 확인
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            augmented_data = checkpoint['augmented_data']
            progress = checkpoint['progress']

        # 증강 진행 중 저장 (배치마다)
        checkpoint_mgr.save_checkpoint(
            augmented_data=current_data,
            progress={'completed': 100, 'total': 1000}
        )
    """

    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        super().__init__(checkpoint_dir, "augmentation_checkpoint")

    def get_checkpoint_path(self) -> Path:
        """체크포인트 파일 경로"""
        return self.checkpoint_dir / f"{self.checkpoint_name}.pkl"

    def save_checkpoint(
        self,
        augmented_data: pd.DataFrame,
        progress: Dict[str, Any],
        methods: Optional[List[str]] = None
    ):
        """
        증강 진행 상황 저장

        Args:
            augmented_data: 증강된 데이터 (원본 + 증강)
            progress: 진행 상황
                - completed: 완료된 증강 샘플 수
                - total: 목표 증강 샘플 수
                - ratio: 진행률 (0.0 ~ 1.0)
            methods: 사용된 증강 방법 리스트
        """
        checkpoint = {
            'augmented_data': augmented_data,
            'progress': progress,
            'methods': methods,
            'original_size': progress.get('original_size', 0),
            'timestamp': self.get_timestamp()
        }

        self._atomic_save_pickle(self.get_checkpoint_path(), checkpoint)

    def load_checkpoint(self) -> Optional[Dict]:
        """
        체크포인트 로드

        Returns:
            Optional[Dict]: 체크포인트 데이터 (없으면 None)
                - augmented_data: 증강된 DataFrame
                - progress: 진행 상황
                - methods: 증강 방법
                - timestamp: 저장 시간
        """
        return self._load_pickle(self.get_checkpoint_path())

    def is_complete(self, target_size: Optional[int] = None) -> bool:
        """
        증강 완료 여부 확인

        Args:
            target_size: 목표 데이터 크기 (None이면 progress의 total 사용)

        Returns:
            bool: 증강이 완료되었으면 True
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return False

        progress = checkpoint.get('progress', {})
        completed = progress.get('completed', 0)
        total = target_size if target_size is not None else progress.get('total', 0)

        return completed >= total

    def get_progress(self) -> Optional[Dict[str, Any]]:
        """
        현재 진행 상황 조회

        Returns:
            Optional[Dict]: 진행 상황 정보 (없으면 None)
                - completed: 완료된 증강 샘플 수
                - total: 목표 증강 샘플 수
                - ratio: 진행률
                - data_size: 현재 데이터 크기
                - methods: 증강 방법
                - timestamp: 마지막 저장 시간
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None

        progress = checkpoint.get('progress', {})
        data = checkpoint.get('augmented_data')
        data_size = len(data) if data is not None else 0

        return {
            'completed': progress.get('completed', 0),
            'total': progress.get('total', 0),
            'ratio': progress.get('ratio', 0.0),
            'data_size': data_size,
            'original_size': checkpoint.get('original_size', 0),
            'methods': checkpoint.get('methods', []),
            'timestamp': checkpoint.get('timestamp')
        }

    def should_save_checkpoint(
        self,
        current_idx: int,
        save_interval: int = 100
    ) -> bool:
        """
        체크포인트 저장 여부 판단 (주기적 저장)

        Args:
            current_idx: 현재 증강 인덱스
            save_interval: 저장 주기 (기본값: 100개마다)

        Returns:
            bool: 저장해야 하면 True
        """
        return (current_idx + 1) % save_interval == 0
