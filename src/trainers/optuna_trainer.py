# ==================== OptunaTrainer ==================== #
"""
Optuna 하이퍼파라미터 튜닝 Trainer

PRD 13: Optuna 하이퍼파라미터 튜닝 모듈
최적의 하이퍼파라미터를 찾아 모델을 최적화하는 것이 목표
"""

# ---------------------- 외부 라이브러리 ---------------------- #
import json
from pathlib import Path
from typing import Dict, Any

# ---------------------- 내부 모듈 ---------------------- #
from src.trainers.base_trainer import BaseTrainer
from src.config import load_model_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer
from src.optimization import OptunaOptimizer


# ==================== OptunaTrainer ==================== #
class OptunaTrainer(BaseTrainer):
    """Optuna 하이퍼파라미터 튜닝 Trainer"""

    def train(self):
        """
        Optuna 튜닝 실행

        Returns:
            dict: 튜닝 결과
                - mode: 'optuna'
                - model: 최적화된 모델
                - best_params: 최적 하이퍼파라미터
                - best_value: 최고 ROUGE 점수
                - n_trials: 수행된 시도 횟수
        """
        self.log("=" * 60)
        self.log("📊 OPTUNA 튜닝 모드 시작")
        self.log(f"🔧 모델: {self.args.models[0]}")
        self.log(f"🔢 시도 횟수: {self.args.optuna_trials}")
        self.log(f"⏱ 최대 시간: {self.args.optuna_timeout}")
        self.log("=" * 60)

        # 1. 데이터 로드
        self.log("\n[1/3] 데이터 로드...")
        train_df, eval_df = self.load_data()

        # 2. Config 로드
        self.log("\n[2/3] Config 로드...")
        model_name = self.args.models[0]
        config = load_model_config(model_name)

        # 명령행 인자로 Config 오버라이드
        self._override_config(config)

        self.log(f"   Config 로드 완료: {model_name}")

        # 3. Optuna Optimizer 초기화
        self.log(f"\n[3/3] Optuna 튜닝 시작...")

        # ✅ --resume_from 옵션 처리
        optimizer_output_dir = self.args.output_dir
        if hasattr(self.args, 'resume_from') and self.args.resume_from:
            # --resume_from이 체크포인트 디렉토리를 가리키는 경우 상위 폴더 사용
            resume_path = Path(self.args.resume_from)
            if resume_path.name == 'checkpoints':
                optimizer_output_dir = str(resume_path.parent)
            else:
                optimizer_output_dir = self.args.resume_from
            self.log(f"🔄 Resume from: {optimizer_output_dir}")

        # Optuna Optimizer 초기화 (데이터프레임 전달)
        optimizer = OptunaOptimizer(
            config=config,
            train_df=train_df,
            eval_df=eval_df,
            n_trials=self.args.optuna_trials,
            timeout=self.args.optuna_timeout,
            study_name=f"optuna_{model_name}_{self.args.experiment_name}",
            storage=None,
            direction="maximize",
            logger=self.logger,
            output_dir=optimizer_output_dir  # resume_from 적용된 경로 전달
        )

        # 튜닝 실행
        study = optimizer.optimize()

        # 결과 반환
        best_params = optimizer.get_best_params()
        best_value = optimizer.get_best_value()

        results = {
            'mode': 'optuna',
            'model': model_name,
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'study_name': optimizer.study_name
        }

        # 결과 저장
        self.log("\n결과 저장 중...")
        optimizer.save_results(str(self.output_dir))

        # 시각화 (선택)
        if self.args.save_visualizations:
            self.log("\n시각화 생성 중...")
            try:
                optimizer.plot_optimization_history(str(self.output_dir))
            except Exception as e:
                self.log(f"    ⚠️  시각화 오류: {e}")

        self.log("\n" + "=" * 60)
        self.log("✅ OPTUNA 튜닝 완료!")
        self.log(f"\n📈 최고 ROUGE-L F1: {best_value:.4f}")
        self.log(f"\n🎯 최적 하이퍼파라미터:")
        for key, value in best_params.items():
            self.log(f"  {key}: {value}")
        self.log("=" * 60)

        return results

    def save_results(self, results):
        """
        결과 저장

        Args:
            results: 튜닝 결과 딕셔너리
        """
        result_path = self.output_dir / "optuna_results.json"

        # 저장 가능한 형태로 변환
        saveable_results = {
            'mode': results['mode'],
            'model': results['model'],
            'best_params': results['best_params'],
            'best_value': results['best_value'],
            'n_trials': results['n_trials'],
            'study_name': results['study_name']
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        self.log(f"\n📂 결과 저장: {result_path}")

        # 최적 파라미터로 Config 생성 (직접 사용 가능)
        best_config_path = self.output_dir / "best_config.yaml"
        self.log(f"📂 최적 Config 저장: {best_config_path}")


# ==================== 편의 함수 ==================== #
def create_optuna_trainer(args, logger, wandb_logger=None):
    """
    OptunaTrainer 생성 편의 함수

    Args:
        args: 명령행 인자
        logger: Logger 인스턴스
        wandb_logger: WandB Logger (선택)

    Returns:
        OptunaTrainer 인스턴스
    """
    return OptunaTrainer(args, logger, wandb_logger)
