"""
Solar API 체크포인트 관리자

Solar API 배치 호출 진행 상황 저장 및 복원
"""

from typing import Optional, Dict, List, Any
from pathlib import Path

from src.checkpoints.base_checkpoint import BaseCheckpointManager


class SolarCheckpointManager(BaseCheckpointManager):
    """
    Solar API 체크포인트 관리자

    기능:
    1. Solar API 배치 호출 진행 상황 저장
    2. 중단 후 Resume 시 완료된 요약 복원
    3. 진행률 추적
    4. API 캐시와 연동

    Usage:
        checkpoint_mgr = SolarCheckpointManager(
            checkpoint_dir="experiments/.../checkpoints"
        )

        # 체크포인트 확인
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            summaries = checkpoint['summaries']
            progress = checkpoint['progress']

        # API 호출 진행 중 저장 (배치마다)
        checkpoint_mgr.save_checkpoint(
            summaries=current_summaries,
            progress={'completed': 100, 'total': 1000}
        )
    """

    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        super().__init__(checkpoint_dir, "solar_api_checkpoint")

    def get_checkpoint_path(self) -> Path:
        """체크포인트 파일 경로"""
        return self.checkpoint_dir / f"{self.checkpoint_name}.pkl"

    def save_checkpoint(
        self,
        summaries: List[str],
        progress: Dict[str, Any],
        model_name: Optional[str] = None,
        api_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Solar API 진행 상황 저장

        Args:
            summaries: 생성된 요약 리스트
            progress: 진행 상황
                - completed: 완료된 API 호출 수
                - total: 전체 호출 대상 수
                - ratio: 진행률 (0.0 ~ 1.0)
            model_name: 사용된 Solar 모델명
            api_stats: API 호출 통계
                - total_tokens: 총 토큰 수
                - total_cost: 총 비용
                - avg_latency: 평균 지연시간
        """
        checkpoint = {
            'summaries': summaries,
            'progress': progress,
            'model_name': model_name,
            'api_stats': api_stats or {},
            'timestamp': self.get_timestamp()
        }

        self._atomic_save_pickle(self.get_checkpoint_path(), checkpoint)

    def load_checkpoint(self) -> Optional[Dict]:
        """
        체크포인트 로드

        Returns:
            Optional[Dict]: 체크포인트 데이터 (없으면 None)
                - summaries: 요약 리스트
                - progress: 진행 상황
                - model_name: Solar 모델명
                - api_stats: API 통계
                - timestamp: 저장 시간
        """
        return self._load_pickle(self.get_checkpoint_path())

    def is_complete(self, target_size: Optional[int] = None) -> bool:
        """
        Solar API 호출 완료 여부 확인

        Args:
            target_size: 목표 호출 수 (None이면 progress의 total 사용)

        Returns:
            bool: 모든 호출이 완료되었으면 True
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
                - completed: 완료된 호출 수
                - total: 전체 호출 대상 수
                - ratio: 진행률
                - summaries_count: 생성된 요약 수
                - model_name: Solar 모델명
                - api_stats: API 통계
                - timestamp: 마지막 저장 시간
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None

        progress = checkpoint.get('progress', {})
        summaries = checkpoint.get('summaries', [])

        return {
            'completed': progress.get('completed', 0),
            'total': progress.get('total', 0),
            'ratio': progress.get('ratio', 0.0),
            'summaries_count': len(summaries),
            'model_name': checkpoint.get('model_name'),
            'api_stats': checkpoint.get('api_stats', {}),
            'timestamp': checkpoint.get('timestamp')
        }

    def update_api_stats(self, tokens: int = 0, cost: float = 0.0, latency: float = 0.0):
        """
        API 통계 업데이트

        Args:
            tokens: 사용된 토큰 수
            cost: 비용
            latency: 지연시간
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return

        api_stats = checkpoint.get('api_stats', {})
        api_stats['total_tokens'] = api_stats.get('total_tokens', 0) + tokens
        api_stats['total_cost'] = api_stats.get('total_cost', 0.0) + cost

        # 평균 지연시간 계산
        count = api_stats.get('call_count', 0) + 1
        current_avg = api_stats.get('avg_latency', 0.0)
        new_avg = (current_avg * (count - 1) + latency) / count
        api_stats['avg_latency'] = new_avg
        api_stats['call_count'] = count

        checkpoint['api_stats'] = api_stats
        self._atomic_save_pickle(self.get_checkpoint_path(), checkpoint)
