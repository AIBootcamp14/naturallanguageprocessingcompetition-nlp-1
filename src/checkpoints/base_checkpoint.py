"""
베이스 체크포인트 관리자

모든 체크포인트 관리자의 공통 기능 제공
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path
import pickle
import json
from datetime import datetime


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
