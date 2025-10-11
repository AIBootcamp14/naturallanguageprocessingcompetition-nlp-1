# ==================== Optuna 하이퍼파라미터 튜닝 모듈 ==================== #
"""
Optuna 기반 하이퍼파라미터 튜닝

PRD 13: Optuna 탐색 공간 확장
- 하이퍼파라미터 자동 탐색
- 조기 종료 (Pruning) 지원
- 멀티-트라이얼 최적화
"""

# ---------------------- 라이브러리 임포트 ---------------------- #
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import json

# ---------------------- 외부 라이브러리 ---------------------- #
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import torch
from transformers import TrainingArguments


# ==================== OptunaHyperparameterTuner 클래스 ==================== #
class OptunaHyperparameterTuner:
    """Optuna 기반 하이퍼파라미터 튜너"""

    def __init__(
        self,
        study_name: str = "dialogue_summarization",
        storage: Optional[str] = None,
        direction: str = "maximize",
        pruner_type: str = "median",
        n_startup_trials: int = 5,
        n_warmup_steps: int = 100,
        logger=None
    ):
        """
        Args:
            study_name: Optuna 스터디 이름
            storage: 스터디 저장 경로 (None이면 메모리)
            direction: 최적화 방향 ("maximize" 또는 "minimize")
            pruner_type: Pruner 종류 ("median" 또는 "halving")
            n_startup_trials: Pruning 시작 전 최소 트라이얼 수
            n_warmup_steps: Pruning을 위한 warmup 스텝 수
            logger: Logger 인스턴스
        """
        self.study_name = study_name
        self.direction = direction
        self.logger = logger

        # Pruner 설정
        if pruner_type == "median":
            self.pruner = MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps
            )
        elif pruner_type == "halving":
            self.pruner = SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4
            )
        else:
            raise ValueError(f"지원하지 않는 pruner 종류: {pruner_type}")

        # Sampler 설정 (TPE: Tree-structured Parzen Estimator)
        self.sampler = TPESampler(seed=42)

        # Study 생성
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True
        )

        self._log(f"OptunaHyperparameterTuner 초기화 완료")
        self._log(f"  - Study: {study_name}")
        self._log(f"  - Direction: {direction}")
        self._log(f"  - Pruner: {pruner_type}")

    def _log(self, msg: str):
        """로그 출력"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        search_space: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        하이퍼파라미터 제안

        Args:
            trial: Optuna Trial 객체
            search_space: 탐색 공간 정의 (None이면 기본 공간 사용)

        Returns:
            제안된 하이퍼파라미터 딕셔너리
        """
        if search_space is None:
            search_space = self.get_default_search_space()

        params = {}

        for param_name, param_config in search_space.items():
            param_type = param_config["type"]

            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            else:
                raise ValueError(f"지원하지 않는 parameter type: {param_type}")

        return params

    def get_default_search_space(self) -> Dict[str, Any]:
        """
        기본 탐색 공간 정의 (PRD 13)

        기존 7개 파라미터:
        - learning_rate, per_device_train_batch_size, gradient_accumulation_steps
        - warmup_ratio, weight_decay, max_grad_norm, label_smoothing_factor

        추가 8개 생성 파라미터:
        - num_beams, temperature, top_p, top_k
        - repetition_penalty, length_penalty, no_repeat_ngram_size, early_stopping_patience

        Returns:
            기본 탐색 공간 딕셔너리
        """
        return {
            # ========== 기존 7개 학습 파라미터 ========== #
            # Learning rate (학습률)
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 5e-4,
                "log": True
            },
            # Batch size
            "per_device_train_batch_size": {
                "type": "categorical",
                "choices": [4, 8, 16, 32]
            },
            # Gradient accumulation steps
            "gradient_accumulation_steps": {
                "type": "categorical",
                "choices": [1, 2, 4, 8]
            },
            # Warmup ratio
            "warmup_ratio": {
                "type": "float",
                "low": 0.0,
                "high": 0.2,
                "log": False
            },
            # Weight decay
            "weight_decay": {
                "type": "float",
                "low": 0.0,
                "high": 0.1,
                "log": False
            },
            # Max gradient norm
            "max_grad_norm": {
                "type": "float",
                "low": 0.5,
                "high": 2.0,
                "log": False
            },
            # Label smoothing
            "label_smoothing_factor": {
                "type": "float",
                "low": 0.0,
                "high": 0.2,
                "log": False
            },

            # ========== 추가 8개 생성 파라미터 ========== #
            # Num beams (빔 서치 개수)
            "num_beams": {
                "type": "categorical",
                "choices": [1, 2, 4, 5, 8]
            },
            # Temperature (생성 다양성 조절)
            "temperature": {
                "type": "float",
                "low": 0.5,
                "high": 2.0,
                "log": False
            },
            # Top-p (nucleus sampling)
            "top_p": {
                "type": "float",
                "low": 0.7,
                "high": 1.0,
                "log": False
            },
            # Top-k (상위 k개 토큰만 고려)
            "top_k": {
                "type": "int",
                "low": 10,
                "high": 100,
                "log": False
            },
            # Repetition penalty (반복 억제)
            "repetition_penalty": {
                "type": "float",
                "low": 1.0,
                "high": 2.0,
                "log": False
            },
            # Length penalty (길이 페널티)
            "length_penalty": {
                "type": "float",
                "low": 0.5,
                "high": 2.0,
                "log": False
            },
            # No repeat ngram size (n-gram 반복 방지)
            "no_repeat_ngram_size": {
                "type": "int",
                "low": 0,
                "high": 5,
                "log": False
            },
            # Early stopping patience (조기 종료 인내)
            "early_stopping_patience": {
                "type": "int",
                "low": 1,
                "high": 5,
                "log": False
            }
        }

    def create_training_args(
        self,
        params: Dict[str, Any],
        output_dir: str,
        num_train_epochs: int = 3,
        **kwargs
    ) -> TrainingArguments:
        """
        Optuna 파라미터로 TrainingArguments 생성

        Args:
            params: Optuna가 제안한 파라미터
            output_dir: 출력 디렉토리
            num_train_epochs: 학습 에포크 수
            **kwargs: 추가 TrainingArguments 인자

        Returns:
            TrainingArguments 인스턴스
        """
        # 기본 설정
        default_args = {
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 100,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_rouge_l",
            "greater_is_better": True,
            "fp16": torch.cuda.is_available(),
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "push_to_hub": False,
        }

        # Optuna 파라미터 병합
        default_args.update(params)

        # 추가 인자 병합
        default_args.update(kwargs)

        return TrainingArguments(**default_args)

    def optimize(
        self,
        objective_fn: Callable[[optuna.Trial], float],
        n_trials: int = 20,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        """
        하이퍼파라미터 최적화 실행

        Args:
            objective_fn: 목적 함수 (trial을 받아 metric을 반환)
            n_trials: 시도할 트라이얼 수
            timeout: 타임아웃 (초)
            n_jobs: 병렬 작업 수
            show_progress_bar: 진행바 표시 여부

        Returns:
            (최적 파라미터, 최적 성능) 튜플
        """
        self._log(f"\n하이퍼파라미터 최적화 시작")
        self._log(f"  - Trials: {n_trials}")
        self._log(f"  - Direction: {self.direction}")

        # 최적화 실행
        self.study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar
        )

        # 최적 결과
        best_params = self.study.best_params
        best_value = self.study.best_value

        self._log(f"\n최적화 완료")
        self._log(f"  - 최적 성능: {best_value:.4f}")
        self._log(f"  - 최적 파라미터:")
        for key, value in best_params.items():
            self._log(f"    - {key}: {value}")

        return best_params, best_value

    def report_intermediate_value(
        self,
        trial: optuna.Trial,
        step: int,
        value: float
    ):
        """
        중간 결과 보고 (Pruning에 사용)

        Args:
            trial: Optuna Trial 객체
            step: 현재 스텝
            value: 중간 성능
        """
        trial.report(value, step)

        # Pruning 판단
        if trial.should_prune():
            self._log(f"Trial {trial.number} pruned at step {step}")
            raise optuna.TrialPruned()

    def get_best_params(self) -> Dict[str, Any]:
        """최적 파라미터 반환"""
        return self.study.best_params

    def get_best_value(self) -> float:
        """최적 성능 반환"""
        return self.study.best_value

    def get_study_summary(self) -> Dict[str, Any]:
        """
        스터디 요약 정보 반환

        Returns:
            스터디 요약 딕셔너리
        """
        return {
            "study_name": self.study_name,
            "direction": self.direction,
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "n_trials": len(self.study.trials),
            "best_trial_number": self.study.best_trial.number,
        }

    def save_study_summary(self, output_path: str):
        """
        스터디 요약을 JSON 파일로 저장

        Args:
            output_path: 출력 파일 경로
        """
        summary = self.get_study_summary()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self._log(f"Study summary saved: {output_path}")

    def plot_optimization_history(self, output_path: Optional[str] = None):
        """
        최적화 히스토리 시각화

        Args:
            output_path: 출력 파일 경로 (None이면 표시만)
        """
        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(self.study)

            if output_path:
                fig.write_html(output_path)
                self._log(f"최적화 히스토리 저장: {output_path}")
            else:
                fig.show()

        except ImportError:
            self._log("Warning: plotly가 설치되지 않아 시각화할 수 없습니다.")

    def plot_param_importances(self, output_path: Optional[str] = None):
        """
        파라미터 중요도 시각화

        Args:
            output_path: 출력 파일 경로 (None이면 표시만)
        """
        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(self.study)

            if output_path:
                fig.write_html(output_path)
                self._log(f"파라미터 중요도 저장: {output_path}")
            else:
                fig.show()

        except ImportError:
            self._log("Warning: plotly가 설치되지 않아 시각화할 수 없습니다.")


# ==================== 팩토리 함수 ==================== #
def create_optuna_tuner(
    study_name: str = "dialogue_summarization",
    storage: Optional[str] = None,
    direction: str = "maximize",
    pruner_type: str = "median",
    logger=None
) -> OptunaHyperparameterTuner:
    """
    Optuna 튜너 생성 팩토리 함수

    Args:
        study_name: 스터디 이름
        storage: 스터디 저장 경로
        direction: 최적화 방향
        pruner_type: Pruner 종류
        logger: Logger 인스턴스

    Returns:
        OptunaHyperparameterTuner 인스턴스
    """
    return OptunaHyperparameterTuner(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner_type=pruner_type,
        logger=logger
    )


# ==================== 사용 예시 ==================== #
if __name__ == "__main__":
    # Optuna 튜너 생성
    tuner = create_optuna_tuner(
        study_name="test_study",
        direction="maximize",
        pruner_type="median"
    )

    # 탐색 공간 확인
    search_space = tuner.get_default_search_space()
    print(f"탐색 공간: {len(search_space)}개 파라미터")
    print("\n학습 파라미터 (7개):")
    for i, key in enumerate(list(search_space.keys())[:7], 1):
        print(f"  {i}. {key}")

    print("\n생성 파라미터 (8개):")
    for i, key in enumerate(list(search_space.keys())[7:], 1):
        print(f"  {i}. {key}")
