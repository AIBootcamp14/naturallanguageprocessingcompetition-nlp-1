"""
K-Fold 체크포인트 관리자

K-Fold Cross-Validation 진행 상황 저장 및 복원
"""

from typing import Optional, Dict, List, Any
from pathlib import Path

from src.checkpoints.base_checkpoint import BaseCheckpointManager


class KFoldCheckpointManager(BaseCheckpointManager):
    """
    K-Fold 체크포인트 관리자

    기능:
    1. Fold 완료마다 결과 저장
    2. 중단 후 Resume 시 완료된 Fold 건너뛰기
    3. 각 Fold별 메트릭 및 모델 경로 보존

    Usage:
        checkpoint_mgr = KFoldCheckpointManager(
            checkpoint_dir="experiments/.../checkpoints",
            n_folds=5
        )

        # 완료된 Fold 확인
        completed_folds = checkpoint_mgr.get_completed_folds()

        # 각 Fold 완료 후 저장
        checkpoint_mgr.save_fold_result(fold=0, metrics={...}, model_path="...")

        # 모든 Fold 완료 확인
        if checkpoint_mgr.is_complete():
            print("All folds completed!")
    """

    def __init__(self, checkpoint_dir: str, n_folds: int = 5):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            n_folds: 전체 Fold 수
        """
        super().__init__(checkpoint_dir, "kfold_checkpoint")
        self.n_folds = n_folds

    def get_checkpoint_path(self) -> Path:
        """체크포인트 파일 경로"""
        return self.checkpoint_dir / f"{self.checkpoint_name}.json"

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """
        체크포인트 저장 (내부 사용)

        Args:
            checkpoint_data: 저장할 체크포인트 데이터
        """
        self._atomic_save_json(self.get_checkpoint_path(), checkpoint_data)

    def load_checkpoint(self) -> Optional[Dict]:
        """
        체크포인트 로드

        Returns:
            Optional[Dict]: 체크포인트 데이터 (없으면 None)
        """
        return self._load_json(self.get_checkpoint_path())

    def save_fold_result(
        self,
        fold: int,
        metrics: Dict[str, float],
        model_path: Optional[str] = None
    ):
        """
        Fold 완료 후 결과 저장

        Args:
            fold: Fold 번호 (0-indexed)
            metrics: 평가 메트릭 (rouge1, rouge2, rougeL, rougeLsum 등)
            model_path: 저장된 모델 경로 (선택)
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            checkpoint = {
                'n_folds': self.n_folds,
                'completed_folds': [],
                'fold_results': {},
                'timestamp': self.get_timestamp()
            }

        # Fold 결과 추가
        fold_key = f"fold_{fold}"
        checkpoint['fold_results'][fold_key] = {
            'fold': fold,
            'metrics': metrics,
            'model_path': model_path,
            'timestamp': self.get_timestamp()
        }

        # 완료된 Fold 목록 업데이트
        if fold not in checkpoint['completed_folds']:
            checkpoint['completed_folds'].append(fold)
            checkpoint['completed_folds'].sort()

        # 마지막 업데이트 시간
        checkpoint['timestamp'] = self.get_timestamp()

        self.save_checkpoint(checkpoint)

    def get_completed_folds(self) -> List[int]:
        """
        완료된 Fold 목록 반환

        Returns:
            List[int]: 완료된 Fold 번호 리스트 (0-indexed)
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return []
        return checkpoint.get('completed_folds', [])

    def get_fold_result(self, fold: int) -> Optional[Dict[str, Any]]:
        """
        특정 Fold 결과 조회

        Args:
            fold: Fold 번호 (0-indexed)

        Returns:
            Optional[Dict]: Fold 결과 (없으면 None)
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None

        fold_key = f"fold_{fold}"
        return checkpoint.get('fold_results', {}).get(fold_key)

    def get_all_fold_results(self) -> Dict[int, Dict[str, Any]]:
        """
        모든 Fold 결과 조회

        Returns:
            Dict[int, Dict]: {fold_number: fold_result} 형태의 딕셔너리
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return {}

        results = {}
        for fold_key, fold_data in checkpoint.get('fold_results', {}).items():
            fold_num = fold_data['fold']
            results[fold_num] = fold_data

        return results

    def is_complete(self) -> bool:
        """
        모든 Fold 완료 여부 확인

        Returns:
            bool: 모든 Fold가 완료되었으면 True
        """
        completed_folds = self.get_completed_folds()
        return len(completed_folds) == self.n_folds

    def get_remaining_folds(self) -> List[int]:
        """
        남은 Fold 목록 반환

        Returns:
            List[int]: 아직 완료되지 않은 Fold 번호 리스트
        """
        completed_folds = self.get_completed_folds()
        all_folds = list(range(self.n_folds))
        return [f for f in all_folds if f not in completed_folds]

    def get_average_metrics(self) -> Optional[Dict[str, float]]:
        """
        완료된 Fold들의 평균 메트릭 계산

        Returns:
            Optional[Dict]: 평균 메트릭 딕셔너리 (완료된 Fold가 없으면 None)
        """
        all_results = self.get_all_fold_results()
        if not all_results:
            return None

        # 메트릭 키 수집
        metric_keys = set()
        for fold_data in all_results.values():
            metric_keys.update(fold_data['metrics'].keys())

        # 평균 계산
        avg_metrics = {}
        for key in metric_keys:
            values = [
                fold_data['metrics'][key]
                for fold_data in all_results.values()
                if key in fold_data['metrics']
            ]
            if values:
                avg_metrics[key] = sum(values) / len(values)

        return avg_metrics

    def get_progress(self) -> Dict[str, Any]:
        """
        현재 진행 상황 조회

        Returns:
            Dict: 진행 상황 정보
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return {
                'completed_folds': 0,
                'total_folds': self.n_folds,
                'progress_percent': 0.0,
                'is_complete': False
            }

        completed = len(checkpoint.get('completed_folds', []))
        return {
            'completed_folds': completed,
            'total_folds': self.n_folds,
            'progress_percent': (completed / self.n_folds) * 100,
            'is_complete': completed == self.n_folds,
            'average_metrics': self.get_average_metrics(),
            'timestamp': checkpoint.get('timestamp')
        }
