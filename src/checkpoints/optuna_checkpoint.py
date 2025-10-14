"""
Optuna 체크포인트 관리자

Optuna 하이퍼파라미터 최적화 진행 상황 저장 및 복원
"""

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import optuna

from src.checkpoints.base_checkpoint import BaseCheckpointManager


class OptunaCheckpointManager(BaseCheckpointManager):
    """
    Optuna 체크포인트 관리자

    기능:
    1. Trial 완료마다 Study 상태 저장
    2. 중단 후 Resume 시 완료된 Trial 복원
    3. 최적 파라미터 및 Trial 결과 보존

    Usage:
        checkpoint_mgr = OptunaCheckpointManager(
            checkpoint_dir="experiments/.../checkpoints",
            study_name="optuna_kobart_ultimate"
        )

        # Study 생성 또는 복원
        study, completed_trials = checkpoint_mgr.resume_study(sampler, pruner, direction)

        # Trial 완료마다 저장
        checkpoint_mgr.save_checkpoint(study, trial.number)
    """

    def __init__(self, checkpoint_dir: str, study_name: str):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            study_name: Optuna Study 이름
        """
        super().__init__(checkpoint_dir, f"{study_name}_checkpoint")
        self.study_name = study_name

    def get_checkpoint_path(self) -> Path:
        """체크포인트 파일 경로"""
        return self.checkpoint_dir / f"{self.checkpoint_name}.pkl"

    def save_checkpoint(self, study: optuna.Study, trial_number: int):
        """
        Trial 완료 후 체크포인트 저장

        Args:
            study: Optuna Study 객체
            trial_number: 완료된 Trial 번호
        """
        checkpoint = {
            'study_name': study.study_name,
            'direction': str(study.direction),
            'best_params': study.best_params if study.best_trial else None,
            'best_value': study.best_value if study.best_trial else None,
            'best_trial_number': study.best_trial.number if study.best_trial else None,
            'completed_trials': trial_number + 1,
            'all_trials': self._trials_to_dict_list(study.trials),
            'timestamp': self.get_timestamp()
        }

        self._atomic_save_pickle(self.get_checkpoint_path(), checkpoint)

    def load_checkpoint(self) -> Optional[Dict]:
        """
        체크포인트 로드

        Returns:
            Optional[Dict]: 체크포인트 데이터 (없으면 None)
        """
        return self._load_pickle(self.get_checkpoint_path())

    def resume_study(
        self,
        sampler: optuna.samplers.BaseSampler,
        pruner: optuna.pruners.BasePruner,
        direction: str = "maximize"
    ) -> Tuple[optuna.Study, int]:
        """
        체크포인트에서 Study 복원 또는 새로 생성

        Args:
            sampler: Optuna Sampler
            pruner: Optuna Pruner
            direction: 최적화 방향 ("maximize" 또는 "minimize")

        Returns:
            Tuple[optuna.Study, int]: (Study 객체, 완료된 Trial 수)
        """
        checkpoint = self.load_checkpoint()

        # 체크포인트가 없으면 새로 시작
        if checkpoint is None:
            study = optuna.create_study(
                study_name=self.study_name,
                sampler=sampler,
                pruner=pruner,
                direction=direction
            )
            return study, 0

        # 체크포인트에서 복원
        study = optuna.create_study(
            study_name=self.study_name,
            sampler=sampler,
            pruner=pruner,
            direction=checkpoint['direction']
        )

        # 완료된 Trial들 재등록
        for trial_dict in checkpoint['all_trials']:
            try:
                trial = self._dict_to_frozen_trial(trial_dict)
                study.add_trial(trial)
            except Exception:
                # Trial 복원 실패 시 건너뜀
                pass

        return study, checkpoint['completed_trials']

    def _trials_to_dict_list(self, trials: List[optuna.trial.FrozenTrial]) -> List[Dict]:
        """
        Trial 리스트를 딕셔너리 리스트로 변환

        Args:
            trials: FrozenTrial 리스트

        Returns:
            List[Dict]: Trial 정보 딕셔너리 리스트
        """
        trial_list = []
        for trial in trials:
            trial_dict = {
                'number': trial.number,
                'state': str(trial.state),
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
                'system_attrs': trial.system_attrs,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
            trial_list.append(trial_dict)
        return trial_list

    def _dict_to_frozen_trial(self, trial_dict: Dict) -> optuna.trial.FrozenTrial:
        """
        딕셔너리를 FrozenTrial로 변환

        Args:
            trial_dict: Trial 정보 딕셔너리

        Returns:
            optuna.trial.FrozenTrial: 복원된 Trial
        """
        from datetime import datetime as dt

        # TrialState 변환
        state_map = {
            'TrialState.COMPLETE': optuna.trial.TrialState.COMPLETE,
            'TrialState.PRUNED': optuna.trial.TrialState.PRUNED,
            'TrialState.FAIL': optuna.trial.TrialState.FAIL,
            'TrialState.RUNNING': optuna.trial.TrialState.RUNNING,
            'TrialState.WAITING': optuna.trial.TrialState.WAITING,
        }
        state = state_map.get(trial_dict['state'], optuna.trial.TrialState.COMPLETE)

        # 타임스탬프 변환
        datetime_start = None
        if trial_dict['datetime_start']:
            datetime_start = dt.fromisoformat(trial_dict['datetime_start'])

        datetime_complete = None
        if trial_dict['datetime_complete']:
            datetime_complete = dt.fromisoformat(trial_dict['datetime_complete'])

        # FrozenTrial 생성
        frozen_trial = optuna.trial.FrozenTrial(
            number=trial_dict['number'],
            state=state,
            value=trial_dict['value'],
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            params=trial_dict['params'],
            distributions={},  # 분포 정보는 복원 불필요
            user_attrs=trial_dict.get('user_attrs', {}),
            system_attrs=trial_dict.get('system_attrs', {}),
            intermediate_values={},
            trial_id=trial_dict['number'],
        )

        return frozen_trial

    def get_progress(self) -> Optional[Dict[str, Any]]:
        """
        현재 진행 상황 조회

        Returns:
            Optional[Dict]: 진행 상황 정보 (없으면 None)
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None

        return {
            'completed_trials': checkpoint['completed_trials'],
            'best_value': checkpoint['best_value'],
            'best_params': checkpoint['best_params'],
            'timestamp': checkpoint['timestamp']
        }
