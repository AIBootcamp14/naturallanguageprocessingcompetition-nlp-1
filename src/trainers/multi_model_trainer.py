# ==================== MultiModelEnsembleTrainer ==================== #
"""
ä ¨x YÁ Trainer

PRD 12: ä ¨x YÁ µ l
ìì ¨xD YµXà YÁ\ °iXì \ ! 
"""

# ---------------------- \ |tì¬ ---------------------- #
import json
from pathlib import Path
from typing import List, Dict, Any

# ---------------------- \¸ ¨È ---------------------- #
from src.trainers.base_trainer import BaseTrainer
from src.config import load_model_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer
from src.ensemble import ModelManager


# ==================== MultiModelEnsembleTrainer ==================== #
class MultiModelEnsembleTrainer(BaseTrainer):
    """ä ¨x YÁ Trainer"""

    def train(self):
        """
        ä ¨x Yµ  YÁ ä

        Returns:
            dict: Yµ °ü
                - mode: 'multi_model'
                - models: ¨x ¬¤¸
                - results:  ¨xÄ Yµ °ü
                - ensemble_strategy: YÁ µ
                - eval_metrics: YÁ É °ü
        """
        self.log("=" * 60)
        self.log("= MULTI MODEL ENSEMBLE ¨Ü Yµ Ü")
        self.log(f"=Ë ¨x: {', '.join(self.args.models)}")
        self.log(f"=' YÁ µ: {self.args.ensemble_strategy}")
        self.log("=" * 60)

        # 1. pt0 \Ü
        self.log("\n[1/4] pt0 \)...")
        train_df, eval_df = self.load_data()

        # 2.  ¨x Yµ
        self.log(f"\n[2/4] ¨x Yµ ({len(self.args.models)} ¨x)...")
        model_results = []
        model_paths = []

        for idx, model_name in enumerate(self.args.models):
            self.log(f"\n{'='*50}")
            self.log(f"¨x {idx+1}/{len(self.args.models)}: {model_name}")
            self.log(f"{'='*50}")

            # Config \Ü
            config = load_model_config(model_name)
            self._override_config(config)

            # ¨x   lt \Ü
            model, tokenizer = load_model_and_tokenizer(config, logger=self.logger)

            # Dataset Ý1
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

            # Trainer Ý1  Yµ
            model_output_dir = self.output_dir / f"model_{idx}_{model_name.replace('-', '_')}"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # ConfigÐ output_dir $
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

            # Yµ ä
            train_result = trainer.train()

            # ¨x ¥
            final_model_path = model_output_dir / 'final_model'
            trainer.save_model(str(final_model_path))
            model_paths.append(str(final_model_path))

            # °ü ¥
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

        # 3. YÁ É
        self.log(f"\n[3/4] YÁ É ...")
        ensemble_metrics = self._evaluate_ensemble(
            model_paths=model_paths,
            eval_df=eval_df,
            strategy=self.args.ensemble_strategy
        )

        # 4. °ü Ñ
        results = {
            'mode': 'multi_model',
            'models': self.args.models,
            'ensemble_strategy': self.args.ensemble_strategy,
            'model_results': model_results,
            'ensemble_metrics': ensemble_metrics
        }

        self.log("\n" + "=" * 60)
        self.log(" MULTI MODEL ENSEMBLE Yµ DÌ!")
        self.log("\n=Ê Ä ¨x 1¥:")
        for result in model_results:
            self.log(f"\n{result['model_name']}:")
            if result['eval_metrics']:
                for key, value in result['eval_metrics'].items():
                    if 'rouge' in key.lower():
                        self.log(f"  {key}: {value:.4f}")

        self.log("\n=Ê YÁ 1¥:")
        if ensemble_metrics:
            for key, value in ensemble_metrics.items():
                self.log(f"  {key}: {value:.4f}")

        self.log("=" * 60)

        return results

    def save_results(self, results):
        """
        °ü ¥

        Args:
            results: Yµ °ü T¬
        """
        result_path = self.output_dir / "multi_model_results.json"

        # ¥ ¥\ Ü\ ÀX
        saveable_results = {
            'mode': results['mode'],
            'models': results['models'],
            'ensemble_strategy': results['ensemble_strategy'],
            'model_results': results['model_results'],
            'ensemble_metrics': results.get('ensemble_metrics', {})
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        self.log(f"\n=¾ °ü ¥: {result_path}")

    def _override_config(self, config):
        """
        9 x\ Config $|tÜ

        Args:
            config: Config ´
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
        Yµ \øÐ É T¸­ 

        Args:
            log_history: TrainerX \ø ¤ ¬

        Returns:
            dict: É T¸­
        """
        eval_metrics = {}

        # ÈÀÉ eval \ø >0
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
        YÁ É

        Args:
            model_paths: ¨x ½\ ¬¤¸
            eval_df: É pt0
            strategy: YÁ µ

        Returns:
            dict: YÁ É T¸­
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from rouge import Rouge
            import torch

            self.log(f"  YÁ µ: {strategy}")

            # ¨x   lt \Ü
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

            # ModelManager\ YÁ Ý1
            manager = ModelManager(logger=self.logger)
            manager.models = models
            manager.tokenizers = tokenizers
            manager.model_names = self.args.models

            # YÁ µÐ 0| Ý1
            if strategy in ['weighted_avg', 'rouge_weighted']:
                ensemble = manager.create_ensemble(
                    ensemble_type='weighted',
                    weights=self.args.ensemble_weights
                )
            else:  # majority_vote ñ
                ensemble = manager.create_ensemble(
                    ensemble_type='voting',
                    voting='hard'
                )

            # !
            dialogues = eval_df['dialogue'].tolist()[:100]  # Ø É
            references = eval_df['summary'].tolist()[:100]

            predictions = ensemble.predict(
                dialogues=dialogues,
                max_length=200,
                num_beams=4,
                batch_size=8
            )

            # ROUGE Ä°
            rouge = Rouge()
            scores = rouge.get_scores(predictions, references, avg=True)

            ensemble_metrics = {
                'ensemble_rouge_1_f1': scores['rouge-1']['f'],
                'ensemble_rouge_2_f1': scores['rouge-2']['f'],
                'ensemble_rouge_l_f1': scores['rouge-l']['f'],
            }

            return ensemble_metrics

        except Exception as e:
            self.log(f"    YÁ É ä(: {e}")
            return {}


# ==================== ¸X h ==================== #
def create_multi_model_trainer(args, logger, wandb_logger=None):
    """
    MultiModelEnsembleTrainer Ý1 ¸X h

    Args:
        args: 9 x
        logger: Logger x¤4¤
        wandb_logger: WandB Logger ( Ý)

    Returns:
        MultiModelEnsembleTrainer x¤4¤
    """
    return MultiModelEnsembleTrainer(args, logger, wandb_logger)
