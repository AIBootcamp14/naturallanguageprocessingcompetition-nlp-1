"""
검증 체크포인트 관리자

모델 검증 진행 상황 저장 및 복원
"""

from typing import Optional, Dict, List, Any
from pathlib import Path

from src.checkpoints.base_checkpoint import BaseCheckpointManager


class ValidationCheckpointManager(BaseCheckpointManager):
    """
    검증 체크포인트 관리자

    기능:
    1. 대규모 검증 세트 평가 진행 상황 저장
    2. 중단 후 Resume 시 완료된 평가 결과 복원
    3. 메트릭 및 진행률 추적

    Usage:
        checkpoint_mgr = ValidationCheckpointManager(
            checkpoint_dir="experiments/.../checkpoints"
        )

        # 체크포인트 확인
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            metrics = checkpoint['metrics']
            predictions = checkpoint['predictions']

        # 검증 진행 중 저장 (배치마다)
        checkpoint_mgr.save_checkpoint(
            predictions=current_predictions,
            metrics=current_metrics,
            progress={'completed': 100, 'total': 1000}
        )
    """

    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        super().__init__(checkpoint_dir, "validation_checkpoint")

    def get_checkpoint_path(self) -> Path:
        """체크포인트 파일 경로"""
        return self.checkpoint_dir / f"{self.checkpoint_name}.pkl"

    def save_checkpoint(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Dict[str, float],
        progress: Dict[str, Any]
    ):
        """
        검증 진행 상황 저장

        Args:
            predictions: 예측 결과 리스트
            references: 정답 리스트
            metrics: 평가 메트릭
                - rouge-1, rouge-2, rouge-l 등
            progress: 진행 상황
                - completed: 완료된 평가 수
                - total: 전체 평가 대상 수
                - ratio: 진행률 (0.0 ~ 1.0)
        """
        checkpoint = {
            'predictions': predictions,
            'references': references,
            'metrics': metrics,
            'progress': progress,
            'timestamp': self.get_timestamp()
        }

        self._atomic_save_pickle(self.get_checkpoint_path(), checkpoint)

    def load_checkpoint(self) -> Optional[Dict]:
        """
        체크포인트 로드

        Returns:
            Optional[Dict]: 체크포인트 데이터 (없으면 None)
                - predictions: 예측 결과
                - references: 정답
                - metrics: 평가 메트릭
                - progress: 진행 상황
                - timestamp: 저장 시간
        """
        return self._load_pickle(self.get_checkpoint_path())

    def is_complete(self, target_size: Optional[int] = None) -> bool:
        """
        검증 완료 여부 확인

        Args:
            target_size: 목표 평가 수 (None이면 progress의 total 사용)

        Returns:
            bool: 모든 평가가 완료되었으면 True
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
                - completed: 완료된 평가 수
                - total: 전체 평가 대상 수
                - ratio: 진행률
                - predictions_count: 예측 결과 수
                - current_metrics: 현재 메트릭
                - timestamp: 마지막 저장 시간
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None

        progress = checkpoint.get('progress', {})
        predictions = checkpoint.get('predictions', [])

        return {
            'completed': progress.get('completed', 0),
            'total': progress.get('total', 0),
            'ratio': progress.get('ratio', 0.0),
            'predictions_count': len(predictions),
            'current_metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp')
        }

    def append_batch_results(
        self,
        batch_predictions: List[str],
        batch_references: List[str],
        updated_metrics: Dict[str, float],
        completed_count: int,
        total_count: int
    ):
        """
        배치 평가 결과 추가

        Args:
            batch_predictions: 배치 예측 결과
            batch_references: 배치 정답
            updated_metrics: 업데이트된 메트릭
            completed_count: 현재까지 완료된 수
            total_count: 전체 수
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            predictions = []
            references = []
        else:
            predictions = checkpoint.get('predictions', [])
            references = checkpoint.get('references', [])

        predictions.extend(batch_predictions)
        references.extend(batch_references)

        progress = {
            'completed': completed_count,
            'total': total_count,
            'ratio': completed_count / total_count if total_count > 0 else 0.0
        }

        self.save_checkpoint(predictions, references, updated_metrics, progress)
