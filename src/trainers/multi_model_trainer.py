# ==================== MultiModelEnsembleTrainer ==================== #
"""
ÃÂ¤ ÃÂ¨x YÃÂ Trainer

PRD 12: ÃÂ¤ ÃÂ¨x YÃÂ ÃÂµ l
ÃÂ¬ÃÂ¬ ÃÂ¨xD YÃÂµXÃÂ  YÃÂ\ ÃÂ°iXÃÂ¬ \ÃÂ ! ÃÂ
"""

# ---------------------- \ |tÃÂ¬ÃÂ¬ ---------------------- #
import json
from pathlib import Path
from typing import List, Dict, Any

# ---------------------- \ÃÂ¸ ÃÂ¨ÃÂ ---------------------- #
from src.trainers.base_trainer import BaseTrainer
from src.config import load_model_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer
from src.ensemble import ModelManager


# ==================== MultiModelEnsembleTrainer ==================== #
class MultiModelEnsembleTrainer(BaseTrainer):
    """ÃÂ¤ ÃÂ¨x YÃÂ Trainer"""

    def train(self):
        """
        ÃÂ¤ ÃÂ¨x YÃÂµ  YÃÂ ÃÂ¤ÃÂ

        Returns:
            dict: YÃÂµ ÃÂ°ÃÂ¼
                - mode: 'multi_model'
                - models: ÃÂ¨x ÃÂ¬ÃÂ¤ÃÂ¸
                - results:  ÃÂ¨xÃÂ YÃÂµ ÃÂ°ÃÂ¼
                - ensemble_strategy: YÃÂ ÃÂµ
                - eval_metrics: YÃÂ ÃÂ ÃÂ°ÃÂ¼
        """
        self.log("=" * 60)
        self.log("=ÃÂ MULTI MODEL ENSEMBLE ÃÂ¨ÃÂ YÃÂµ ÃÂÃÂ")
        self.log(f"=ÃÂ ÃÂ¨x: {', '.join(self.args.models)}")
        self.log(f"=' YÃÂ ÃÂµ: {self.args.ensemble_strategy}")
        self.log("=" * 60)

        # 1. pt0 \ÃÂ
        self.log("\n[1/4] pt0 \)...")
        train_df, eval_df = self.load_data()

        # 2.  ÃÂ¨x YÃÂµ
        self.log(f"\n[2/4] ÃÂ¨x YÃÂµ ({len(self.args.models)} ÃÂ¨x)...")
        model_results = []
        model_paths = []

        for idx, model_name in enumerate(self.args.models):
            self.log(f"\n{'='*50}")
            self.log(f"ÃÂ¨x {idx+1}/{len(self.args.models)}: {model_name}")
            self.log(f"{'='*50}")

            # Config \ÃÂ
            config = load_model_config(model_name)
            self._override_config(config)

            # ÃÂ¨x  ÃÂ lÃÂt \ÃÂ
            model, tokenizer = load_model_and_tokenizer(config, logger=self.logger)

            # Dataset ÃÂ1
            model_type = config.model.get('type', 'encoder_decoder')

            train_dataset = DialogueSummarizationDataset(
                dialogues=train_df['dialogue'].tolist(),
                summaries=train_df['summary'].tolist(),
                tokenizer=tokenizer,
                encoder_max_len=config.tokenizer.encoder_max_len,
                decoder_max_len=config.tokenizer.decoder_max_len,
                preprocess=True,
                model_type=model_type
            )

            eval_dataset = DialogueSummarizationDataset(
                dialogues=eval_df['dialogue'].tolist(),
                summaries=eval_df['summary'].tolist(),
                tokenizer=tokenizer,
                encoder_max_len=config.tokenizer.encoder_max_len,
                decoder_max_len=config.tokenizer.decoder_max_len,
                preprocess=True,
                model_type=model_type
            )

            # Trainer ÃÂ1  YÃÂµ
            model_output_dir = self.output_dir / f"model_{idx}_{model_name.replace('-', '_')}"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # ConfigÃÂ output_dir $
            config.training.output_dir = str(model_output_dir)

            trainer = create_trainer(
                config=config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                use_wandb=getattr(self.args, 'use_wandb', True),
                logger=self.logger
            )

            # YÃÂµ ÃÂ¤ÃÂ
            train_result = trainer.train()

            # ÃÂ¨x ÃÂ¥
            final_model_path = model_output_dir / 'final_model'
            trainer.save_model(str(final_model_path))
            model_paths.append(str(final_model_path))

            # ÃÂ°ÃÂ¼ ÃÂ¥
            eval_metrics = self._extract_eval_metrics(trainer.state.log_history)
            model_results.append({
                'model_name': model_name,
                'model_path': str(final_model_path),
                'eval_metrics': eval_metrics
            })

            # FIXME: Corrupted log message
            if eval_metrics:
                for key, value in eval_metrics.items():
                    if 'rouge' in key.lower():
                        self.log(f"  {key}: {value:.4f}")

        # 3. YÃÂ ÃÂ
        self.log(f"\n[3/4] YÃÂ ÃÂ ...")
        ensemble_metrics = self._evaluate_ensemble(
            model_paths=model_paths,
            eval_df=eval_df,
            strategy=self.args.ensemble_strategy
        )

        # 4. ÃÂ°ÃÂ¼ ÃÂ
        results = {
            'mode': 'multi_model',
            'models': self.args.models,
            'ensemble_strategy': self.args.ensemble_strategy,
            'model_results': model_results,
            'ensemble_metrics': ensemble_metrics
        }

        self.log("\n" + "=" * 60)
        self.log(" MULTI MODEL ENSEMBLE YÃÂµ DÃÂ!")
        self.log("\n=ÃÂ ÃÂ ÃÂ¨x 1ÃÂ¥:")
        for result in model_results:
            self.log(f"\n{result['model_name']}:")
            if result['eval_metrics']:
                for key, value in result['eval_metrics'].items():
                    if 'rouge' in key.lower():
                        self.log(f"  {key}: {value:.4f}")

        self.log("\n=ÃÂ YÃÂ 1ÃÂ¥:")
        if ensemble_metrics:
            for key, value in ensemble_metrics.items():
                self.log(f"  {key}: {value:.4f}")

        self.log("=" * 60)

        return results

    def save_results(self, results):
        """
        ÃÂ°ÃÂ¼ ÃÂ¥

        Args:
            results: YÃÂµ ÃÂ°ÃÂ¼ TÃÂ¬
        """
        result_path = self.output_dir / "multi_model_results.json"

        # ÃÂ¥ ÃÂ¥\ ÃÂ\ ÃÂX
        saveable_results = {
            'mode': results['mode'],
            'models': results['models'],
            'ensemble_strategy': results['ensemble_strategy'],
            'model_results': results['model_results'],
            'ensemble_metrics': results.get('ensemble_metrics', {})
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        self.log(f"\n=ÃÂ¾ ÃÂ°ÃÂ¼ ÃÂ¥: {result_path}")

    def _override_config(self, config):
        """
        ÃÂ9ÃÂ xÃÂ\ Config $ÃÂ|tÃÂ

        Args:
            config: Config ÃÂ´
        """
        # Epochs
        if hasattr(self.args, 'epochs') and self.args.epochs is not None:
            config.training.epochs = self.args.epochs

        # Batch size
        if hasattr(self.args, 'batch_size') and self.args.batch_size is not None:
            config.training.batch_size = self.args.batch_size

        # Learning rate
        if hasattr(self.args, 'learning_rate') and self.args.learning_rate is not None:
            config.training.learning_rate = self.args.learning_rate

    def _extract_eval_metrics(self, log_history):
        """
        YÃÂµ \ÃÂ¸ÃÂ ÃÂ TÃÂ¸ÃÂ­ ÃÂÃÂ

        Args:
            log_history: TrainerX \ÃÂ¸ ÃÂÃÂ¤ÃÂ ÃÂ¬

        Returns:
            dict: ÃÂ TÃÂ¸ÃÂ­
        """
        eval_metrics = {}

        # ÃÂÃÂÃÂ eval \ÃÂ¸ >0
        for log_entry in reversed(log_history):
            if 'eval_loss' in log_entry:
                for key, value in log_entry.items():
                    if key.startswith('eval_'):
                        eval_metrics[key] = value
                break

        return eval_metrics

    def _evaluate_ensemble(
        self,
        model_paths: List[str],
        eval_df,
        strategy: str
    ) -> Dict[str, float]:
        """
        YÃÂ ÃÂ

        Args:
            model_paths: ÃÂ¨x ÃÂ½\ ÃÂ¬ÃÂ¤ÃÂ¸
            eval_df: ÃÂ pt0ÃÂ
            strategy: YÃÂ ÃÂµ

        Returns:
            dict: YÃÂ ÃÂ TÃÂ¸ÃÂ­
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from rouge import Rouge
            import torch

            self.log(f"  YÃÂ ÃÂµ: {strategy}")

            # ÃÂ¨x  ÃÂ lÃÂt \ÃÂ
            models = []
            tokenizers = []

            for model_path in model_paths:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()

                models.append(model)
                tokenizers.append(tokenizer)

            # ModelManager\ YÃÂ ÃÂ1
            manager = ModelManager(logger=self.logger)
            manager.models = models
            manager.tokenizers = tokenizers
            manager.model_names = self.args.models

            # YÃÂ ÃÂµÃÂ 0| ÃÂ1
            if strategy in ['weighted_avg', 'rouge_weighted']:
                ensemble = manager.create_ensemble(
                    ensemble_type='weighted',
                    weights=self.args.ensemble_weights
                )
            else:  # majority_vote ÃÂ±
                ensemble = manager.create_ensemble(
                    ensemble_type='voting',
                    voting='hard'
                )

            # !
            dialogues = eval_df['dialogue'].tolist()[:100]  # ÃÂ ÃÂ
            references = eval_df['summary'].tolist()[:100]

            predictions = ensemble.predict(
                dialogues=dialogues,
                max_length=200,
                num_beams=4,
                batch_size=8
            )

            # ROUGE ÃÂÃÂ°
            rouge = Rouge()
            scores = rouge.get_scores(predictions, references, avg=True)

            ensemble_metrics = {
                'ensemble_rouge_1_f1': scores['rouge-1']['f'],
                'ensemble_rouge_2_f1': scores['rouge-2']['f'],
                'ensemble_rouge_l_f1': scores['rouge-l']['f'],
            }

            return ensemble_metrics

        except Exception as e:
            self.log(f"  ÃÂ  YÃÂ ÃÂ ÃÂ¤(: {e}")
            return {}


# ==================== ÃÂ¸X h ==================== #
def create_multi_model_trainer(args, logger, wandb_logger=None):
    """
    MultiModelEnsembleTrainer ÃÂ1 ÃÂ¸X h

    Args:
        args: ÃÂ9ÃÂ xÃÂ
        logger: Logger xÃÂ¤4ÃÂ¤
        wandb_logger: WandB Logger ( ÃÂ)

    Returns:
        MultiModelEnsembleTrainer xÃÂ¤4ÃÂ¤
    """
    return MultiModelEnsembleTrainer(args, logger, wandb_logger)
