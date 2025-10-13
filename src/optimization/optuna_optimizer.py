"""
Optuna 하이퍼파라미터 최적화 시스템

PRD 13: Optuna 하이퍼파라미터 최적화 전략
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, Any, Optional, List, Callable
import logging
from pathlib import Path
import json
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..training import ModelTrainer
from ..data import DialogueSummarizationDataset
from ..models import ModelLoader


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
        train_dataset: DialogueSummarizationDataset,
        val_dataset: DialogueSummarizationDataset,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = "maximize",
        logger: Optional[logging.Logger] = None
    ):
        """
        초기화

        Args:
            config: 기본 Config (탐색 공간 정의에 사용)
            train_dataset: 학습 데이터셋
            val_dataset: 검증 데이터셋
            n_trials: 총 Trial 횟수
            timeout: 최대 실행 시간 (초)
            study_name: Study 이름
            storage: Study 저장소 (SQLite/PostgreSQL)
            direction: 최적화 방향 ("maximize" or "minimize")
            logger: 로거
        """
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"optuna_study_{config.model.name}"
        self.storage = storage
        self.direction = direction
        self.logger = logger or logging.getLogger(__name__)

        # Optuna Study
        self.study: Optional[optuna.Study] = None

        # Best params 저장
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None

        self.logger.info(f"OptunaOptimizer 초기화 완료")
        self.logger.info(f"  - Study 이름: {self.study_name}")
        self.logger.info(f"  - Trial 횟수: {self.n_trials}")
        self.logger.info(f"  - 방향: {self.direction}")

    def create_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        하이퍼파라미터 탐색 공간 생성

        Args:
            trial: Optuna Trial 객체

        Returns:
            샘플링된 하이퍼파라미터 딕셔너리
        """
        params = {}

        # 1. 모델 파라미터 (LoRA)
        if self.config.get('lora'):
            params['lora_r'] = trial.suggest_categorical('lora_r', [8, 16, 32, 64])
            params['lora_alpha'] = trial.suggest_categorical('lora_alpha', [16, 32, 64, 128])
            params['lora_dropout'] = trial.suggest_float('lora_dropout', 0.0, 0.2)

        # 2. 학습 파라미터
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        params['num_epochs'] = trial.suggest_int('num_epochs', 3, 10)
        params['warmup_ratio'] = trial.suggest_float('warmup_ratio', 0.0, 0.2)
        params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)

        # 3. Scheduler
        params['scheduler_type'] = trial.suggest_categorical(
            'scheduler_type',
            ['linear', 'cosine', 'cosine_with_restarts', 'polynomial']
        )

        # 4. Generation 파라미터
        params['temperature'] = trial.suggest_float('temperature', 0.1, 1.0)
        params['top_p'] = trial.suggest_float('top_p', 0.5, 1.0)
        params['num_beams'] = trial.suggest_categorical('num_beams', [2, 4, 6, 8])
        params['length_penalty'] = trial.suggest_float('length_penalty', 0.5, 2.0)

        # 5. Dropout (모델에 따라)
        if self.config.model.get('hidden_dropout_prob') is not None:
            params['hidden_dropout'] = trial.suggest_float('hidden_dropout', 0.0, 0.3)
            params['attention_dropout'] = trial.suggest_float('attention_dropout', 0.0, 0.3)

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

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Trial {trial.number} 시작")
        self.logger.info(f"파라미터: {params}")
        self.logger.info(f"{'='*60}")

        # 2. Config 업데이트
        config = OmegaConf.to_container(self.config, resolve=True)
        config = OmegaConf.create(config)

        # Training 파라미터 업데이트
        config.training.learning_rate = params['learning_rate']
        config.training.batch_size = params['batch_size']
        config.training.num_epochs = params['num_epochs']
        config.training.warmup_ratio = params['warmup_ratio']
        config.training.weight_decay = params['weight_decay']
        config.training.scheduler_type = params['scheduler_type']

        # Generation 파라미터 업데이트 (inference 섹션에 있음)
        if not hasattr(config, 'generation'):
            # generation 섹션이 없으면 inference 섹션 사용
            if hasattr(config, 'inference'):
                config.inference.num_beams = params['num_beams']
                config.inference.length_penalty = params['length_penalty']
                # temperature와 top_p는 KoBART에서 사용 안 함 (beam search 모델)
        else:
            config.generation.temperature = params['temperature']
            config.generation.top_p = params['top_p']
            config.generation.num_beams = params['num_beams']
            config.generation.length_penalty = params['length_penalty']

        # LoRA 파라미터 업데이트
        if 'lora_r' in params:
            config.lora.r = params['lora_r']
            config.lora.alpha = params['lora_alpha']
            config.lora.dropout = params['lora_dropout']

        # Dropout 업데이트
        if 'hidden_dropout' in params:
            config.model.hidden_dropout_prob = params['hidden_dropout']
            config.model.attention_dropout_prob = params['attention_dropout']

        try:
            # 3. 모델 로더 초기화
            model_loader = ModelLoader(config)
            model, tokenizer = model_loader.load_model_and_tokenizer()

            # 4. Trainer 초기화 (WandB 비활성화)
            config.logging.use_wandb = False

            trainer = ModelTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                config=config,
                output_dir=f"outputs/optuna_trial_{trial.number}"
            )

            # 5. 학습
            trainer.train()

            # 6. 검증 평가
            metrics = trainer.evaluate()

            # 7. ROUGE-L F1 반환 (maximize)
            rouge_l_f1 = metrics.get('rouge_l_f1', 0.0)

            self.logger.info(f"Trial {trial.number} 완료")
            self.logger.info(f"  - ROUGE-L F1: {rouge_l_f1:.4f}")
            self.logger.info(f"  - ROUGE-1 F1: {metrics.get('rouge_1_f1', 0.0):.4f}")
            self.logger.info(f"  - ROUGE-2 F1: {metrics.get('rouge_2_f1', 0.0):.4f}")

            # 8. 중간 결과 보고 (Pruning에 사용)
            trial.report(rouge_l_f1, step=config.training.num_epochs)

            # 9. Pruning 체크
            if trial.should_prune():
                self.logger.info(f"Trial {trial.number} Pruned!")
                raise optuna.TrialPruned()

            return rouge_l_f1

        except Exception as e:
            self.logger.error(f"Trial {trial.number} 실패: {str(e)}")
            raise optuna.TrialPruned()

    def optimize(self) -> optuna.Study:
        """
        하이퍼파라미터 최적화 실행

        Returns:
            완료된 Optuna Study
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Optuna 최적화 시작")
        self.logger.info(f"{'='*70}")

        # 1. Sampler 및 Pruner 설정
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(
            n_startup_trials=5,  # 처음 5개 trial은 pruning 안함
            n_warmup_steps=3,     # 3 에포크 후부터 pruning
            interval_steps=1      # 매 에포크마다 체크
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
            # 완료된 trial이 없는 경우
            self.logger.error(f"완료된 trial이 없습니다: {e}")
            self.best_params = {}
            self.best_value = 0.0

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Optuna 최적화 완료")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"최적 ROUGE-L F1: {self.best_value:.4f}")
        self.logger.info(f"최적 파라미터:")
        for key, value in self.best_params.items():
            self.logger.info(f"  - {key}: {value}")

        return self.study

    def get_best_params(self) -> Dict[str, Any]:
        """
        최적 하이퍼파라미터 반환

        Returns:
            최적 파라미터 딕셔너리
        """
        if self.best_params is None:
            raise ValueError("optimize()를 먼저 실행해야 합니다")

        return self.best_params

    def get_best_value(self) -> float:
        """
        최적 ROUGE 점수 반환

        Returns:
            최적 ROUGE 점수
        """
        if self.best_value is None:
            raise ValueError("optimize()를 먼저 실행해야 합니다")

        return self.best_value

    def save_results(self, output_path: str):
        """
        최적화 결과 저장

        Args:
            output_path: 저장 경로
        """
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

        self.logger.info(f"최적 파라미터 저장: {best_params_path}")

        # 2. All trials 저장
        trials_df = self.study.trials_dataframe()
        trials_csv_path = output_path / "all_trials.csv"
        trials_df.to_csv(trials_csv_path, index=False, encoding='utf-8')

        self.logger.info(f"전체 Trial 저장: {trials_csv_path}")

        # 3. Study 통계
        stats = {
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'n_completed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'best_value': self.best_value,
            'best_trial_number': self.study.best_trial.number
        }

        stats_path = output_path / "study_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Study 통계 저장: {stats_path}")
        self.logger.info(f"  - 완료: {stats['n_completed']}")
        self.logger.info(f"  - Pruned: {stats['n_pruned']}")
        self.logger.info(f"  - 실패: {stats['n_failed']}")

    def plot_optimization_history(self, output_path: str):
        """
        최적화 히스토리 시각화

        Args:
            output_path: 저장 경로
        """
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

            self.logger.info(f"시각화 저장 완료: {output_path}")

        except ImportError:
            self.logger.warning("plotly가 설치되지 않아 시각화를 건너뜁니다")


def create_optuna_optimizer(
    config: DictConfig,
    train_dataset: DialogueSummarizationDataset,
    val_dataset: DialogueSummarizationDataset,
    n_trials: int = 50,
    **kwargs
) -> OptunaOptimizer:
    """
    OptunaOptimizer 편의 생성 함수

    Args:
        config: Config
        train_dataset: 학습 데이터셋
        val_dataset: 검증 데이터셋
        n_trials: Trial 횟수
        **kwargs: 추가 파라미터

    Returns:
        OptunaOptimizer 인스턴스
    """
    return OptunaOptimizer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_trials=n_trials,
        **kwargs
    )
