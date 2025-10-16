"""
HuggingFace 보정 체크포인트 관리자

HuggingFace 사전학습 모델 보정 진행 상황 저장 및 복원
"""

from typing import Optional, Dict, List, Any
from pathlib import Path

from src.checkpoints.base_checkpoint import BaseCheckpointManager


class CorrectionCheckpointManager(BaseCheckpointManager):
    """
    HuggingFace 보정 체크포인트 관리자

    기능:
    1. 보정 진행 중 배치 단위로 저장
    2. 중단 후 Resume 시 완료된 보정 데이터 복원
    3. 진행률 추적

    Usage:
        checkpoint_mgr = CorrectionCheckpointManager(
            checkpoint_dir="experiments/.../checkpoints"
        )

        # 체크포인트 확인
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            corrected_summaries = checkpoint['corrected_summaries']
            progress = checkpoint['progress']

        # 보정 진행 중 저장 (배치마다)
        checkpoint_mgr.save_checkpoint(
            corrected_summaries=current_summaries,
            progress={'completed': 100, 'total': 1000}
        )
    """

    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        super().__init__(checkpoint_dir, "correction_checkpoint")

    def get_checkpoint_path(self) -> Path:
        """체크포인트 파일 경로"""
        return self.checkpoint_dir / f"{self.checkpoint_name}.pkl"

    def save_checkpoint(
        self,
        corrected_summaries: List[str],
        progress: Dict[str, Any],
        correction_strategy: Optional[str] = None,
        models: Optional[List[str]] = None
    ):
        """
        보정 진행 상황 저장

        Args:
            corrected_summaries: 보정된 요약 리스트
            progress: 진행 상황
                - completed: 완료된 보정 수
                - total: 전체 보정 대상 수
                - ratio: 진행률 (0.0 ~ 1.0)
            correction_strategy: 보정 전략
            models: 사용된 보정 모델 리스트
        """
        checkpoint = {
            'corrected_summaries': corrected_summaries,
            'progress': progress,
            'correction_strategy': correction_strategy,
            'models': models,
            'timestamp': self.get_timestamp()
        }

        self._atomic_save_pickle(self.get_checkpoint_path(), checkpoint)

    def load_checkpoint(self) -> Optional[Dict]:
        """
        체크포인트 로드

        Returns:
            Optional[Dict]: 체크포인트 데이터 (없으면 None)
                - corrected_summaries: 보정된 요약 리스트
                - progress: 진행 상황
                - correction_strategy: 보정 전략
                - models: 보정 모델 리스트
                - timestamp: 저장 시간
        """
        return self._load_pickle(self.get_checkpoint_path())

    def is_complete(self, target_size: Optional[int] = None) -> bool:
        """
        보정 완료 여부 확인

        Args:
            target_size: 목표 보정 수 (None이면 progress의 total 사용)

        Returns:
            bool: 보정이 완료되었으면 True
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
                - completed: 완료된 보정 수
                - total: 전체 보정 대상 수
                - ratio: 진행률
                - corrected_count: 보정된 요약 수
                - correction_strategy: 보정 전략
                - models: 보정 모델
                - timestamp: 마지막 저장 시간
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None

        progress = checkpoint.get('progress', {})
        summaries = checkpoint.get('corrected_summaries', [])

        return {
            'completed': progress.get('completed', 0),
            'total': progress.get('total', 0),
            'ratio': progress.get('ratio', 0.0),
            'corrected_count': len(summaries),
            'correction_strategy': checkpoint.get('correction_strategy'),
            'models': checkpoint.get('models', []),
            'timestamp': checkpoint.get('timestamp')
        }
