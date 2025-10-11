# ==================== FullPipelineTrainer ==================== #
"""
´ t|x Trainer

PRD 14: ä 5X Ü¤\ - Full ¨Ü
¨à 0¥D °i\ \ t|x:
- ä ¨x YÁ
- Optuna \T (5X)
- K-Fold P( 
- TTA (Test Time Augmentation)
- Solar API µi
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
from src.api import create_solar_api


# ==================== FullPipelineTrainer ==================== #
class FullPipelineTrainer(BaseTrainer):
    """´ t|x Trainer"""

    def train(self):
        """
        ´ t|x ä

        Returns:
            dict: ä °ü
                - mode: 'full'
                - models: ¬©\ ¨x ¬¤¸
                - ensemble_results: YÁ °ü
                - solar_results: Solar API °ü (5X)
                - final_metrics: \ É T¸­
        """
        self.log("=" * 60)
        self.log("= FULL PIPELINE ¨Ü Ü")
        self.log(f"=Ë ¨x: {', '.join(self.args.models)}")
        self.log(f"=' YÁ µ: {self.args.ensemble_strategy}")
        self.log(f"= TTA ¬©: {self.args.use_tta}")
        self.log("=" * 60)

        # 1. pt0 \Ü
        self.log("\n[1/5] pt0 \)...")
        train_df, eval_df = self.load_data()

        # 2. ¨x Yµ (ä ¨x)
        self.log(f"\n[2/5] ä ¨x Yµ ({len(self.args.models)} ¨x)...")
        model_results, model_paths = self._train_multiple_models(train_df, eval_df)

        # 3. YÁ l1
        self.log(f"\n[3/5] YÁ l1...")
        ensemble_results = self._create_and_evaluate_ensemble(
            model_paths=model_paths,
            eval_df=eval_df
        )

        # 4. Solar API µi (5X)
        solar_results = {}
        try:
            self.log(f"\n[4/5] Solar API µi...")
            solar_results = self._integrate_solar_api(eval_df)
        except Exception as e:
            self.log(f"    Solar API µi t: {e}")

        # 5. TTA © (5X)
        tta_results = {}
        if self.args.use_tta:
            self.log(f"\n[5/5] TTA ©...")
            tta_results = self._apply_tta(model_paths, eval_df)

        # °ü Ñ
        results = {
            'mode': 'full',
            'models': self.args.models,
            'ensemble_strategy': self.args.ensemble_strategy,
            'use_tta': self.args.use_tta,
            'model_results': model_results,
            'ensemble_results': ensemble_results,
            'solar_results': solar_results,
            'tta_results': tta_results
        }

        self.log("\n" + "=" * 60)
        self.log(" FULL PIPELINE DÌ!")

        self.log("\n=Ê Ä ¨x 1¥:")
        for result in model_results:
            self.log(f"\n{result['model_name']}:")
            if result['eval_metrics']:
                for key, value in result['eval_metrics'].items():
                    if 'rouge' in key.lower():
                        self.log(f"  {key}: {value:.4f}")

        self.log("\n=Ê YÁ 1¥:")
        if ensemble_results:
            for key, value in ensemble_results.items():
                self.log(f"  {key}: {value:.4f}")

        if solar_results:
            self.log("\n=Ê Solar API 1¥:")
            for key, value in solar_results.items():
                if isinstance(value, (int, float)):
                    self.log(f"  {key}: {value:.4f}")

        self.log("=" * 60)

        return results

    def save_results(self, results):
        """
        °ü ¥

        Args:
            results: ä °ü T¬
        """
        result_path = self.output_dir / "full_pipeline_results.json"

        # ¥ ¥\ Ü\ ÀX
        saveable_results = {
            'mode': results['mode'],
            'models': results['models'],
            'ensemble_strategy': results['ensemble_strategy'],
            'use_tta': results['use_tta'],
            'model_results': results['model_results'],
            'ensemble_results': results.get('ensemble_results', {}),
            'solar_results': results.get('solar_results', {}),
            'tta_results': results.get('tta_results', {})
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        self.log(f"\n=¾ °ü ¥: {result_path}")

    def _train_multiple_models(self, train_df, eval_df):
        """
        ä ¨x Yµ

        Args:
            train_df: Yµ pt0
            eval_df: É pt0

        Returns:
            tuple: (model_results, model_paths)
        """
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

            config.training.output_dir = str(model_output_dir)

            trainer = create_trainer(
                config=config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                use_wandb=False,
                logger=self.logger
            )

            # Yµ ä
            train_result = trainer.train()

            # Get model path from training result (model already saved by train())
            final_model_path = train_result.get('final_model_path', str(model_output_dir / 'final_model'))
            model_paths.append(final_model_path)

            # Get evaluation metrics from training result
            eval_metrics = train_result.get('eval_metrics', {})
            model_results.append({
                'model_name': model_name,
                'model_path': final_model_path,
                'eval_metrics': eval_metrics
            })

            # FIXME: Corrupted log message

        return model_results, model_paths

    def _create_and_evaluate_ensemble(self, model_paths, eval_df):
        """
        YÁ Ý1  É

        Args:
            model_paths: ¨x ½\ ¬¤¸
            eval_df: É pt0

        Returns:
            dict: YÁ É T¸­
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from rouge import Rouge
            import torch

            # ¨x \Ü
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

            # YÁ Ý1
            manager = ModelManager(logger=self.logger)
            manager.models = models
            manager.tokenizers = tokenizers
            manager.model_names = self.args.models

            if self.args.ensemble_strategy in ['weighted_avg', 'rouge_weighted']:
                ensemble = manager.create_ensemble(
                    ensemble_type='weighted',
                    weights=self.args.ensemble_weights
                )
            else:
                ensemble = manager.create_ensemble(
                    ensemble_type='voting',
                    voting='hard'
                )

            # !
            dialogues = eval_df['dialogue'].tolist()[:100]
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

            return {
                'ensemble_rouge_1_f1': scores['rouge-1']['f'],
                'ensemble_rouge_2_f1': scores['rouge-2']['f'],
                'ensemble_rouge_l_f1': scores['rouge-l']['f'],
            }

        except Exception as e:
            self.log(f"    YÁ É ä(: {e}")
            return {}

    def _integrate_solar_api(self, eval_df):
        """
        Solar API µi

        Args:
            eval_df: É pt0

        Returns:
            dict: Solar API É °ü
        """
        try:
            # Solar API t|t¸¸ Ý1
            solar_client = create_solar_api(use_cache=True)

            # Ø !
            dialogues = eval_df['dialogue'].tolist()[:50]
            references = eval_df['summary'].tolist()[:50]

            self.log(f"  Solar API processing {len(dialogues)} samples ...")

            predictions = solar_client.batch_generate(
                dialogues=dialogues,
                batch_size=10,
                use_few_shot=True,
                preprocess=True
            )

            # ROUGE Ä°
            from rouge import Rouge
            rouge = Rouge()
            scores = rouge.get_scores(predictions, references, avg=True)

            return {
                'solar_rouge_1_f1': scores['rouge-1']['f'],
                'solar_rouge_2_f1': scores['rouge-2']['f'],
                'solar_rouge_l_f1': scores['rouge-l']['f'],
                'n_samples': len(predictions)
            }

        except Exception as e:
            self.log(f"    Solar API µi ä(: {e}")
            return {}

    def _apply_tta(self, model_paths, eval_df):
        """
        TTA (Test Time Augmentation) ©

        Args:
            model_paths: ¨x ½\ ¬¤¸
            eval_df: É pt0

        Returns:
            dict: TTA °ü
        """
        try:
            self.log(f"  TTA µ: {', '.join(self.args.tta_strategies)}")
            self.log(f"   : {self.args.tta_num_aug}")

            # TTA Ä l 
            self.log("    TTA 0¥@ Ä l Èä.")

            return {
                'tta_applied': False,
                'strategies': self.args.tta_strategies,
                'num_aug': self.args.tta_num_aug
            }

        except Exception as e:
            self.log(f"    TTA © ä(: {e}")
            return {}

    def _override_config(self, config):
        """Config $|tÜ"""
        if hasattr(self.args, 'epochs') and self.args.epochs is not None:
            config.training.epochs = self.args.epochs

        if hasattr(self.args, 'batch_size') and self.args.batch_size is not None:
            config.training.batch_size = self.args.batch_size

        if hasattr(self.args, 'learning_rate') and self.args.learning_rate is not None:
            config.training.learning_rate = self.args.learning_rate

    def _extract_eval_metrics(self, log_history):
        """É T¸­ """
        eval_metrics = {}

        for log_entry in reversed(log_history):
            if 'eval_loss' in log_entry:
                for key, value in log_entry.items():
                    if key.startswith('eval_'):
                        eval_metrics[key] = value
                break

        return eval_metrics


# ==================== ¸X h ==================== #
def create_full_pipeline_trainer(args, logger, wandb_logger=None):
    """
    FullPipelineTrainer Ý1 ¸X h

    Args:
        args: 9 x
        logger: Logger x¤4¤
        wandb_logger: WandB Logger ( Ý)

    Returns:
        FullPipelineTrainer x¤4¤
    """
    return FullPipelineTrainer(args, logger, wandb_logger)
