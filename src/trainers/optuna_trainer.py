# ==================== OptunaTrainer ==================== #
"""
Optuna Xt||ÃÂ¸0 \T Trainer

PRD 13: Optuna Xt||ÃÂ¸0 \T ÃÂµ l
ÃÂÃÂ Xt||ÃÂ¸0 ÃÂÃÂD ÃÂµt \X ÃÂ¨x $ ÃÂÃÂ
"""

# ---------------------- \ |tÃÂ¬ÃÂ¬ ---------------------- #
import json
from pathlib import Path
from typing import Dict, Any

# ---------------------- \ÃÂ¸ ÃÂ¨ÃÂ ---------------------- #
from src.trainers.base_trainer import BaseTrainer
from src.config import load_model_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer
from src.optimization import OptunaOptimizer


# ==================== OptunaTrainer ==================== #
class OptunaTrainer(BaseTrainer):
    """Optuna Xt||ÃÂ¸0 \T Trainer"""

    def train(self):
        """
        Optuna \T ÃÂ¤ÃÂ

        Returns:
            dict: \T ÃÂ°ÃÂ¼
                - mode: 'optuna'
                - model: ÃÂ¬ÃÂ©\ ÃÂ¨x
                - best_params: \ Xt||ÃÂ¸0
                - best_value: \ ROUGE 
                - n_trials: ÃÂ´ ÃÂÃÂ ÃÂ
        """
        self.log("=" * 60)
        self.log("=ÃÂ OPTUNA \T ÃÂ¨ÃÂ ÃÂÃÂ")
        self.log(f"=ÃÂ ÃÂ¨x: {self.args.models[0]}")
        self.log(f"=' ÃÂÃÂ ÃÂ: {self.args.optuna_trials}")
        self.log(f"ÃÂ± \ ÃÂ: {self.args.optuna_timeout}")
        self.log("=" * 60)

        # 1. pt0 \ÃÂ
        self.log("\n[1/3] pt0 \)...")
        train_df, eval_df = self.load_data()

        # 2. Config \ÃÂ
        self.log("\n[2/3] Config \)...")
        model_name = self.args.models[0]
        config = load_model_config(model_name)

        self.log(f"   Config \ÃÂ DÃÂ: {model_name}")

        # 3. Dataset D (\ ÃÂÃÂ ÃÂ1)
        self.log("\npt0K D ...")

        # TokenizerÃÂ \T ÃÂ¼ÃÂ ÃÂ¬\ÃÂXÃÂÃÂ, pt0ÃÂ@ ÃÂ¬ÃÂ¬ÃÂ©
        self.train_df = train_df
        self.eval_df = eval_df

        # 4. Optuna Optimizer 0T
        self.log(f"\n[3/3] Optuna \T ÃÂÃÂ...")

        from src.data import create_datasets_from_df

        # Dataset ÃÂ1 ÃÂ¬| (Optuna ÃÂ´ÃÂÃÂ ÃÂ¬ÃÂ©)
        def create_datasets(tokenizer, config):
            """Dataset ÃÂ1 ÃÂ¬| h"""
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

            return train_dataset, eval_dataset

        # Optuna Optimizer 0T
        optimizer = OptunaOptimizer(
            config=config,
            train_dataset=None,  # Objective ÃÂ´ÃÂÃÂ ÃÂ1
            val_dataset=None,    # Objective ÃÂ´ÃÂÃÂ ÃÂ1
            n_trials=self.args.optuna_trials,
            timeout=self.args.optuna_timeout,
            study_name=f"optuna_{model_name}_{self.args.experiment_name}",
            storage=None,  # xTÃÂ¨ÃÂ¬
            direction="maximize",
            logger=self.logger.logger if hasattr(self.logger, 'logger') else None
        )

        # Dataset ÃÂ1 h| optimizerÃÂ ÃÂ¬
        optimizer.create_datasets = create_datasets

        # \T ÃÂ¤ÃÂ
        study = optimizer.optimize()

        # ÃÂ°ÃÂ¼ ÃÂ
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

        # ÃÂ°ÃÂ¼ ÃÂ¥
        self.log("\nÃÂ°ÃÂ¼ ÃÂ¥ ...")
        optimizer.save_results(str(self.output_dir))

        # ÃÂT (5X)
        if self.args.save_visualizations:
            self.log("\nÃÂT ÃÂ1 ...")
            try:
                optimizer.plot_optimization_history(str(self.output_dir))
            except Exception as e:
                self.log(f"  ÃÂ  ÃÂT ÃÂ¤(: {e}")

        self.log("\n" + "=" * 60)
        self.log(" OPTUNA \T DÃÂ!")
        self.log(f"\n=ÃÂ \ ROUGE-L F1: {best_value:.4f}")
        self.log(f"\n=ÃÂ \ Xt||ÃÂ¸0:")
        for key, value in best_params.items():
            self.log(f"  {key}: {value}")
        self.log("=" * 60)

        return results

    def save_results(self, results):
        """
        ÃÂ°ÃÂ¼ ÃÂ¥

        Args:
            results: \T ÃÂ°ÃÂ¼ TÃÂ¬
        """
        result_path = self.output_dir / "optuna_results.json"

        # ÃÂ¥ ÃÂ¥\ ÃÂ\ ÃÂX
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

        self.log(f"\n=ÃÂ¾ ÃÂ°ÃÂ¼ ÃÂ¥: {result_path}")

        # \ |ÃÂ¸0\ Config ÃÂ1 (ÃÂ¬ÃÂ¬ÃÂ© ÃÂ¥)
        best_config_path = self.output_dir / "best_config.yaml"
        self.log(f"=ÃÂ¾ \ Config ÃÂ¥: {best_config_path}")


# ==================== ÃÂ¸X h ==================== #
def create_optuna_trainer(args, logger, wandb_logger=None):
    """
    OptunaTrainer ÃÂ1 ÃÂ¸X h

    Args:
        args: ÃÂ9ÃÂ xÃÂ
        logger: Logger xÃÂ¤4ÃÂ¤
        wandb_logger: WandB Logger ( ÃÂ)

    Returns:
        OptunaTrainer xÃÂ¤4ÃÂ¤
    """
    return OptunaTrainer(args, logger, wandb_logger)
