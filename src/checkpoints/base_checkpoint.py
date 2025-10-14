"""
베이스 체크포인트 관리자

모든 체크포인트 관리자의 공통 기능 제공
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from pathlib import Path
import pickle
import json
import gzip
import shutil
from datetime import datetime, timedelta


class BaseCheckpointManager(ABC):
    """
    체크포인트 관리자 베이스 클래스

    모든 체크포인트 관리자는 이 클래스를 상속받아 구현

    주요 기능:
    1. 원자적 저장 (저장 중 실패 시 이전 상태 유지)
    2. Pickle/JSON 저장 지원
    3. 체크포인트 존재 여부 확인
    4. 체크포인트 삭제

    서브클래스는 다음 메서드를 반드시 구현해야 함:
    - save_checkpoint(): 체크포인트 저장
    - load_checkpoint(): 체크포인트 로드
    - get_checkpoint_path(): 체크포인트 파일 경로 반환
    """

    def __init__(self, checkpoint_dir: str, checkpoint_name: str):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            checkpoint_name: 체크포인트 파일 이름 (확장자 제외)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = checkpoint_name

    @abstractmethod
    def save_checkpoint(self, data: Any, **kwargs):
        """
        체크포인트 저장 (서브클래스에서 구현)

        Args:
            data: 저장할 데이터
            **kwargs: 추가 파라미터
        """
        pass

    @abstractmethod
    def load_checkpoint(self) -> Optional[Any]:
        """
        체크포인트 로드 (서브클래스에서 구현)

        Returns:
            Optional[Any]: 로드된 데이터 (없으면 None)
        """
        pass

    @abstractmethod
    def get_checkpoint_path(self) -> Path:
        """
        체크포인트 파일 경로 반환 (서브클래스에서 구현)

        Returns:
            Path: 체크포인트 파일 경로
        """
        pass

    def _atomic_save_pickle(self, file_path: Path, data: Any):
        """
        원자적 Pickle 저장

        저장 중 실패하더라도 기존 파일은 유지됨

        Args:
            file_path: 저장할 파일 경로
            data: 저장할 데이터
        """
        tmp_file = file_path.with_suffix('.tmp')
        try:
            with open(tmp_file, 'wb') as f:
                pickle.dump(data, f)
            # 저장 성공 시에만 기존 파일 대체
            tmp_file.replace(file_path)
        except Exception as e:
            # 실패 시 임시 파일 삭제
            if tmp_file.exists():
                tmp_file.unlink()
            raise e

    def _atomic_save_json(self, file_path: Path, data: dict):
        """
        원자적 JSON 저장

        저장 중 실패하더라도 기존 파일은 유지됨

        Args:
            file_path: 저장할 파일 경로
            data: 저장할 딕셔너리 데이터
        """
        tmp_file = file_path.with_suffix('.tmp')
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # 저장 성공 시에만 기존 파일 대체
            tmp_file.replace(file_path)
        except Exception as e:
            # 실패 시 임시 파일 삭제
            if tmp_file.exists():
                tmp_file.unlink()
            raise e

    def _load_pickle(self, file_path: Path) -> Optional[Any]:
        """
        Pickle 파일 로드

        Args:
            file_path: 로드할 파일 경로

        Returns:
            Optional[Any]: 로드된 데이터 (실패 시 None)
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _load_json(self, file_path: Path) -> Optional[dict]:
        """
        JSON 파일 로드

        Args:
            file_path: 로드할 파일 경로

        Returns:
            Optional[dict]: 로드된 딕셔너리 (실패 시 None)
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def exists(self) -> bool:
        """
        체크포인트 존재 여부 확인

        Returns:
            bool: 체크포인트가 존재하면 True
        """
        return self.get_checkpoint_path().exists()

    def delete_checkpoint(self):
        """체크포인트 파일 삭제"""
        checkpoint_path = self.get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def get_timestamp(self) -> str:
        """
        현재 타임스탬프 반환 (ISO 형식)

        Returns:
            str: 타임스탬프 문자열
        """
        return datetime.now().isoformat()

    # ==================== Phase 3: 개선 및 최적화 기능 ====================

    def compress_checkpoint(self, compression_level: int = 6) -> bool:
        """
        체크포인트 파일 압축 (디스크 공간 절약)

        Args:
            compression_level: 압축 레벨 (1-9, 기본값: 6)

        Returns:
            bool: 압축 성공 여부
        """
        checkpoint_path = self.get_checkpoint_path()
        if not checkpoint_path.exists():
            return False

        compressed_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.gz')

        try:
            with open(checkpoint_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # 압축 성공 시 원본 파일 삭제
            checkpoint_path.unlink()
            return True
        except Exception:
            # 압축 실패 시 압축 파일 삭제
            if compressed_path.exists():
                compressed_path.unlink()
            return False

    def decompress_checkpoint(self) -> bool:
        """
        체크포인트 파일 압축 해제

        Returns:
            bool: 압축 해제 성공 여부
        """
        checkpoint_path = self.get_checkpoint_path()
        compressed_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.gz')

        if not compressed_path.exists():
            return False

        try:
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(checkpoint_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # 압축 해제 성공 시 압축 파일 삭제
            compressed_path.unlink()
            return True
        except Exception:
            # 압축 해제 실패 시 압축 해제 파일 삭제
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            return False

    def cleanup_old_checkpoints(self, keep_last_n: int = 3, max_age_days: Optional[int] = None):
        """
        오래된 체크포인트 자동 정리

        Args:
            keep_last_n: 최근 N개 체크포인트 유지 (기본값: 3)
            max_age_days: 최대 보관 기간 (일 단위, None이면 무제한)
        """
        # 체크포인트 디렉토리의 모든 파일 찾기
        pattern = f"{self.checkpoint_name}*"
        checkpoint_files = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # 최신 파일부터
        )

        # 최근 N개 유지
        files_to_delete = checkpoint_files[keep_last_n:]

        # 날짜 기준 필터링
        if max_age_days is not None:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            files_to_delete = [
                f for f in files_to_delete
                if datetime.fromtimestamp(f.stat().st_mtime) < cutoff_date
            ]

        # 파일 삭제
        for file_path in files_to_delete:
            try:
                file_path.unlink()
            except Exception:
                pass

    def get_checkpoint_size(self) -> Optional[int]:
        """
        체크포인트 파일 크기 조회 (바이트)

        Returns:
            Optional[int]: 파일 크기 (파일이 없으면 None)
        """
        checkpoint_path = self.get_checkpoint_path()
        if not checkpoint_path.exists():
            # 압축 파일 확인
            compressed_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.gz')
            if compressed_path.exists():
                return compressed_path.stat().st_size
            return None

        return checkpoint_path.stat().st_size

    def get_checkpoint_age(self) -> Optional[timedelta]:
        """
        체크포인트 파일 생성 후 경과 시간

        Returns:
            Optional[timedelta]: 경과 시간 (파일이 없으면 None)
        """
        checkpoint_path = self.get_checkpoint_path()
        if not checkpoint_path.exists():
            return None

        mtime = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
        return datetime.now() - mtime

    def list_all_checkpoints(self) -> List[Path]:
        """
        현재 디렉토리의 모든 체크포인트 파일 목록

        Returns:
            List[Path]: 체크포인트 파일 경로 리스트 (최신순)
        """
        pattern = f"{self.checkpoint_name}*"
        checkpoint_files = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return checkpoint_files

    def get_progress_percentage(self) -> Optional[float]:
        """
        진행률 반환 (서브클래스에서 get_progress() 구현 필요)

        Returns:
            Optional[float]: 진행률 (0.0 ~ 100.0), 없으면 None
        """
        if hasattr(self, 'get_progress'):
            progress = self.get_progress()
            if progress and 'ratio' in progress:
                return progress['ratio'] * 100.0
        return None

    def format_progress_bar(self, width: int = 50) -> str:
        """
        진행률 표시줄 생성

        Args:
            width: 표시줄 너비 (문자 수)

        Returns:
            str: 진행률 표시줄 문자열
        """
        percentage = self.get_progress_percentage()
        if percentage is None:
            return "[" + " " * width + "] 0.0%"

        filled = int(width * percentage / 100.0)
        bar = "=" * filled + " " * (width - filled)
        return f"[{bar}] {percentage:.1f}%"
