"""
Optuna 하이퍼파라미터 최적화 시스템

PRD 13: Optuna 하이퍼파라미터 최적화 전략
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import json
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..models import load_model_and_tokenizer
from ..data import DialogueSummarizationDataset
from ..training import create_trainer


class OptunaOptimizer:
    """
    Optuna 하이퍼파라미터 최적화 클래스

    기능:
    - NLP 특화 하이퍼파라미터 탐색 공간 정의
    - Bayesian Optimization (TPE Sampler)
    - Median Pruner를 통한 조기 종료
    - ROUGE 점수 기반 최적화
    - 결과 저장 및 분석
    """

    def __init__(
        self,
        config: DictConfig,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = "maximize",
        logger=None
    ):
        """
        초기화

        Args:
            config: 기본 Config (탐색 공간 정의에 사용)
            train_df: 학습 데이터프레임
            eval_df: 검증 데이터프레임
            n_trials: 총 Trial 횟수
            timeout: 최대 실행 시간 (초)
            study_name: Study 이름
            storage: Study 저장소 (SQLite/PostgreSQL)
            direction: 최적화 방향 ("maximize" or "minimize")
            logger: Logger 인스턴스
        """
        self.config = config
        self.train_df = train_df
        self.eval_df = eval_df
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"optuna_study_{config.model.name}"
        self.storage = storage
        self.direction = direction
        self.logger = logger

        # Optuna Study
        self.study: Optional[optuna.Study] = None

        # Best params 저장
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None

        self._log(f"OptunaOptimizer 초기화 완료")
        self._log(f"  - Study 이름: {self.study_name}")
        self._log(f"  - Trial 횟수: {self.n_trials}")
        self._log(f"  - 방향: {self.direction}")

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            if hasattr(self.logger, 'write'):
                self.logger.write(msg)
            else:
                self.logger.info(msg)
        else:
            print(msg)

    def create_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        하이퍼파라미터 탐색 공간 생성

        Args:
            trial: Optuna Trial 객체

        Returns:
            샘플링된 하이퍼파라미터 딕셔너리
        """
        params = {}

        # 학습 파라미터 (핵심)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        params['num_epochs'] = trial.suggest_int('num_epochs', 3, 10)
        params['warmup_ratio'] = trial.suggest_float('warmup_ratio', 0.0, 0.2)
        params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)

        # Scheduler
        params['scheduler_type'] = trial.suggest_categorical(
            'scheduler_type',
            ['linear', 'cosine', 'cosine_with_restarts', 'polynomial']
        )

        # Generation 파라미터 (KoBART용)
        params['num_beams'] = trial.suggest_categorical('num_beams', [2, 4, 6, 8])
        params['length_penalty'] = trial.suggest_float('length_penalty', 0.5, 2.0)

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """
        최적화 목적 함수

        Args:
            trial: Optuna Trial 객체

        Returns:
            ROUGE 점수 (maximize)
        """
        # 1. 하이퍼파라미터 샘플링
        params = self.create_search_space(trial)

        self._log(f"\n{'='*60}")
        self._log(f"Trial {trial.number} 시작")
        self._log(f"파라미터: {params}")
        self._log(f"{'='*60}")

        try:
            # 2. Config 복사 및 업데이트
            config = OmegaConf.to_container(self.config, resolve=True)
            config = OmegaConf.create(config)

            # Training 파라미터 업데이트
            if not hasattr(config, 'training'):
                config.training = {}

            config.training.learning_rate = params['learning_rate']
            config.training.epochs = params['num_epochs']
            config.training.warmup_ratio = params['warmup_ratio']
            config.training.weight_decay = params['weight_decay']
            config.training.lr_scheduler_type = params['scheduler_type']

            # Inference 파라미터 업데이트 (KoBART는 inference 섹션 사용)
            if hasattr(config, 'inference'):
                config.inference.num_beams = params['num_beams']
                config.inference.length_penalty = params['length_penalty']

            # WandB 비활성화
            if not hasattr(config, 'logging'):
                config.logging = {}
            config.logging.use_wandb = False

            # 3. 모델 및 토크나이저 로드
            model, tokenizer = load_model_and_tokenizer(config, logger=self.logger)

            # 4. Dataset 생성
            model_type = config.model.get('type', 'encoder_decoder')

            train_dataset = DialogueSummarizationDataset(
                dialogues=self.train_df['dialogue'].tolist(),
                summaries=self.train_df['summary'].tolist(),
                tokenizer=tokenizer,
                encoder_max_len=config.tokenizer.encoder_max_len,
                decoder_max_len=config.tokenizer.decoder_max_len,
                preprocess=True,
                model_type=model_type
            )

            eval_dataset = DialogueSummarizationDataset(
                dialogues=self.eval_df['dialogue'].tolist(),
                summaries=self.eval_df['summary'].tolist(),
                tokenizer=tokenizer,
                encoder_max_len=config.tokenizer.encoder_max_len,
                decoder_max_len=config.tokenizer.decoder_max_len,
                preprocess=True,
                model_type=model_type
            )

            # 5. Trainer 생성
            trainer = create_trainer(
                config=config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                use_wandb=False,
                logger=self.logger
            )

            # 6. 학습
            trainer.train()

            # 7. 평가
            metrics = trainer.evaluate()

            # 8. ROUGE-L F1 추출
            rouge_l_f1 = 0.0
            # 다양한 키 형식 시도 (대소문자 구분)
            possible_keys = [
                'eval_rougeL',      # HuggingFace 기본 형식
                'eval_rouge_l',     # 소문자 형식
                'eval_rouge_l_f1',  # F1 명시 형식
                'rougeL',           # prefix 없는 형식
                'rouge_l',
                'rouge_l_f1'
            ]

            for key in possible_keys:
                if key in metrics:
                    rouge_l_f1 = metrics[key]
                    self._log(f"  → 메트릭 '{key}' 사용: {rouge_l_f1:.4f}")
                    break

            if rouge_l_f1 == 0.0:
                # 메트릭을 찾지 못한 경우 디버깅 정보 출력
                self._log(f"  ⚠️  ROUGE-L 메트릭을 찾을 수 없습니다")
                self._log(f"  사용 가능한 메트릭: {list(metrics.keys())}")

            self._log(f"Trial {trial.number} 완료")
            self._log(f"  - ROUGE-L F1: {rouge_l_f1:.4f}")

            # 9. 중간 결과 보고
            trial.report(rouge_l_f1, step=params['num_epochs'])

            # 10. Pruning 체크
            if trial.should_prune():
                self._log(f"Trial {trial.number} Pruned!")
                raise optuna.TrialPruned()

            return rouge_l_f1

        except Exception as e:
            self._log(f"Trial {trial.number} 실패: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise optuna.TrialPruned()

    def optimize(self) -> optuna.Study:
        """
        하이퍼파라미터 최적화 실행

        Returns:
            완료된 Optuna Study
        """
        self._log(f"\n{'='*70}")
        self._log(f"Optuna 최적화 시작")
        self._log(f"{'='*70}")

        # 1. Sampler 및 Pruner 설정
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )

        # 2. Study 생성
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=True
        )

        # 3. 최적화 실행
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # 4. 최적 파라미터 저장
        try:
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
        except ValueError as e:
            self._log(f"완료된 trial이 없습니다: {e}")
            self.best_params = {}
            self.best_value = 0.0

        self._log(f"\n{'='*70}")
        self._log(f"Optuna 최적화 완료")
        self._log(f"{'='*70}")
        self._log(f"최적 ROUGE-L F1: {self.best_value:.4f}")
        self._log(f"최적 파라미터:")
        for key, value in self.best_params.items():
            self._log(f"  - {key}: {value}")

        return self.study

    def get_best_params(self) -> Dict[str, Any]:
        """최적 하이퍼파라미터 반환"""
        if self.best_params is None:
            raise ValueError("optimize()를 먼저 실행해야 합니다")
        return self.best_params

    def get_best_value(self) -> float:
        """최적 ROUGE 점수 반환"""
        if self.best_value is None:
            raise ValueError("optimize()를 먼저 실행해야 합니다")
        return self.best_value

    def save_results(self, output_path: str):
        """최적화 결과 저장"""
        if self.study is None:
            raise ValueError("optimize()를 먼저 실행해야 합니다")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Best params 저장
        best_params_path = output_path / "best_params.json"
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': self.best_params,
                'best_value': self.best_value,
                'n_trials': len(self.study.trials)
            }, f, indent=2, ensure_ascii=False)

        self._log(f"최적 파라미터 저장: {best_params_path}")

        # 2. All trials 저장
        trials_df = self.study.trials_dataframe()
        trials_csv_path = output_path / "all_trials.csv"
        trials_df.to_csv(trials_csv_path, index=False, encoding='utf-8')

        self._log(f"전체 Trial 저장: {trials_csv_path}")

        # 3. Study 통계
        stats = {
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'n_completed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'best_value': self.best_value
        }

        if self.best_params:
            stats['best_trial_number'] = self.study.best_trial.number

        stats_path = output_path / "study_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self._log(f"Study 통계 저장: {stats_path}")
        self._log(f"  - 완료: {stats['n_completed']}")
        self._log(f"  - Pruned: {stats['n_pruned']}")
        self._log(f"  - 실패: {stats['n_failed']}")

    def plot_optimization_history(self, output_path: str):
        """최적화 히스토리 시각화"""
        if self.study is None:
            raise ValueError("optimize()를 먼저 실행해야 합니다")

        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate
            )

            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # 1. Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(str(output_path / "optimization_history.html"))

            # 2. Parameter importances
            fig = plot_param_importances(self.study)
            fig.write_html(str(output_path / "param_importances.html"))

            # 3. Parallel coordinate
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(str(output_path / "parallel_coordinate.html"))

            self._log(f"시각화 저장 완료: {output_path}")

        except ImportError:
            self._log("plotly가 설치되지 않아 시각화를 건너뜁니다")


def create_optuna_optimizer(
    config: DictConfig,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    n_trials: int = 50,
    **kwargs
) -> OptunaOptimizer:
    """
    OptunaOptimizer 편의 생성 함수

    Args:
        config: Config
        train_df: 학습 데이터프레임
        eval_df: 검증 데이터프레임
        n_trials: Trial 횟수
        **kwargs: 추가 파라미터

    Returns:
        OptunaOptimizer 인스턴스
    """
    return OptunaOptimizer(
        config=config,
        train_df=train_df,
        eval_df=eval_df,
        n_trials=n_trials,
        **kwargs
    )
