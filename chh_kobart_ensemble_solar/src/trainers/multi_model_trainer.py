# ==================== MultiModelEnsembleTrainer ==================== #
"""
다중 모델 앙상블 Trainer

PRD 12: 다중 모델 앙상블 모듈
여러 모델을 학습하여 앙상블로 성능을 향상시키는 목적
"""

# ---------------------- 외부 라이브러리 ---------------------- #
import json
from pathlib import Path
from typing import List, Dict, Any

# ---------------------- 내부 모듈 ---------------------- #
from src.trainers.base_trainer import BaseTrainer
from src.config import load_model_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer
from src.ensemble import ModelManager


# ==================== MultiModelEnsembleTrainer ==================== #
class MultiModelEnsembleTrainer(BaseTrainer):
    """다중 모델 앙상블 Trainer"""

    def train(self):
        """
        다중 모델 학습 및 앙상블 실행

        Returns:
            dict: 학습 결과
                - mode: 'multi_model'
                - models: 모델 목록
                - results: 각 모델의 학습 결과
                - ensemble_strategy: 앙상블 전략
                - eval_metrics: 앙상블 평가 결과
        """
        self.log("=" * 60)
        self.log("📊 MULTI MODEL ENSEMBLE 모드 학습 시작")
        self.log(f"🔧 모델: {', '.join(self.args.models)}")
        self.log(f"🔢 앙상블 전략: {self.args.ensemble_strategy}")
        self.log("=" * 60)

        # 1. 데이터 로드
        self.log("\n[1/4] 데이터 로드...")
        train_df, eval_df = self.load_data()

        # 2. 각 모델 학습
        self.log(f"\n[2/4] 모델 학습 ({len(self.args.models)} 모델)...")
        model_results = []
        model_paths = []

        for idx, model_name in enumerate(self.args.models):
            self.log(f"\n{'='*50}")
            self.log(f"모델 {idx+1}/{len(self.args.models)}: {model_name}")
            self.log(f"{'='*50}")

            # Config 로드
            config = load_model_config(model_name)
            self._override_config(config)

            # 모델 및 토크나이저 로드
            model, tokenizer = load_model_and_tokenizer(config, logger=self.logger)

            # Dataset 생성
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

            # Trainer 생성 및 학습
            model_output_dir = self.output_dir / f"model_{idx}_{model_name.replace('-', '_')}"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Config에 output_dir 설정
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

            # 학습 실행
            train_result = trainer.train()

            # 모델 저장
            final_model_path = model_output_dir / 'final_model'
            trainer.save_model(str(final_model_path))
            model_paths.append(str(final_model_path))

            # 결과 저장
            eval_metrics = self._extract_eval_metrics(trainer.state.log_history)
            model_results.append({
                'model_name': model_name,
                'model_path': str(final_model_path),
                'eval_metrics': eval_metrics
            })

            # 평가 지표 출력
            if eval_metrics:
                for key, value in eval_metrics.items():
                    if 'rouge' in key.lower():
                        self.log(f"  {key}: {value:.4f}")

        # 3. 앙상블 평가
        self.log(f"\n[3/4] 앙상블 평가 중...")
        ensemble_metrics = self._evaluate_ensemble(
            model_paths=model_paths,
            eval_df=eval_df,
            strategy=self.args.ensemble_strategy
        )

        # 4. 결과 반환
        results = {
            'mode': 'multi_model',
            'models': self.args.models,
            'ensemble_strategy': self.args.ensemble_strategy,
            'model_results': model_results,
            'ensemble_metrics': ensemble_metrics
        }

        self.log("\n" + "=" * 60)
        self.log("✅ MULTI MODEL ENSEMBLE 학습 완료!")
        self.log("\n📈 개별 모델 성능:")
        for result in model_results:
            self.log(f"\n{result['model_name']}:")
            if result['eval_metrics']:
                for key, value in result['eval_metrics'].items():
                    if 'rouge' in key.lower():
                        self.log(f"  {key}: {value:.4f}")

        self.log("\n📈 앙상블 성능:")
        if ensemble_metrics:
            for key, value in ensemble_metrics.items():
                self.log(f"  {key}: {value:.4f}")

        self.log("=" * 60)

        return results

    def save_results(self, results):
        """
        결과 저장

        Args:
            results: 학습 결과 딕셔너리
        """
        result_path = self.output_dir / "multi_model_results.json"

        # 저장 가능한 형태로 변환
        saveable_results = {
            'mode': results['mode'],
            'models': results['models'],
            'ensemble_strategy': results['ensemble_strategy'],
            'model_results': results['model_results'],
            'ensemble_metrics': results.get('ensemble_metrics', {})
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        self.log(f"\n📂 결과 저장: {result_path}")

    def _extract_eval_metrics(self, log_history):
        """
        학습 로그에서 평가 지표 추출

        Args:
            log_history: Trainer의 로그 히스토리 기록

        Returns:
            dict: 평가 지표
        """
        eval_metrics = {}

        # 마지막 eval 로그 추출
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
        앙상블 평가

        Args:
            model_paths: 모델 경로 목록
            eval_df: 평가 데이터프레임
            strategy: 앙상블 전략

        Returns:
            dict: 앙상블 평가 지표
        """
        try:
            from transformers import (
                AutoConfig,
                AutoModelForSeq2SeqLM,
                AutoModelForCausalLM,
                AutoTokenizer
            )
            from rouge import Rouge
            import torch

            self.log(f"  앙상블 전략: {strategy}")

            # 모델 및 토크나이저 로드
            models = []
            tokenizers = []

            for model_path in model_paths:
                # 모델 타입 자동 감지
                config = AutoConfig.from_pretrained(model_path)
                is_encoder_decoder = config.is_encoder_decoder if hasattr(config, 'is_encoder_decoder') else False

                # 모델 타입에 따라 적절한 클래스 사용
                if is_encoder_decoder:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_path)

                tokenizer = AutoTokenizer.from_pretrained(model_path)

                # Decoder-only 모델의 경우 left padding 설정
                if not is_encoder_decoder:
                    tokenizer.padding_side = "left"
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()

                models.append(model)
                tokenizers.append(tokenizer)

            # ModelManager로 앙상블 생성
            manager = ModelManager(logger=self.logger)
            manager.models = models
            manager.tokenizers = tokenizers
            manager.model_names = self.args.models

            # 앙상블 전략에 따라 생성
            if strategy in ['weighted_avg', 'rouge_weighted']:
                ensemble = manager.create_ensemble(
                    ensemble_type='weighted',
                    weights=self.args.ensemble_weights
                )
            else:  # majority_vote 등
                ensemble = manager.create_ensemble(
                    ensemble_type='voting',
                    voting='hard'
                )

            # 예측
            dialogues = eval_df['dialogue'].tolist()[:100]  # 샘플 평가
            references = eval_df['summary'].tolist()[:100]

            predictions = ensemble.predict(
                dialogues=dialogues,
                max_new_tokens=200,
                min_new_tokens=30,
                num_beams=4,
                batch_size=8
            )

            # ROUGE 계산
            rouge = Rouge()
            scores = rouge.get_scores(predictions, references, avg=True)

            ensemble_metrics = {
                'ensemble_rouge_1_f1': scores['rouge-1']['f'],
                'ensemble_rouge_2_f1': scores['rouge-2']['f'],
                'ensemble_rouge_l_f1': scores['rouge-l']['f'],
            }

            return ensemble_metrics

        except Exception as e:
            self.log(f"    ⚠️  앙상블 평가 오류: {e}")
            return {}


# ==================== 편의 함수 ==================== #
def create_multi_model_trainer(args, logger, wandb_logger=None):
    """
    MultiModelEnsembleTrainer 생성 편의 함수

    Args:
        args: 명령행 인자
        logger: Logger 인스턴스
        wandb_logger: WandB Logger (선택)

    Returns:
        MultiModelEnsembleTrainer 인스턴스
    """
    return MultiModelEnsembleTrainer(args, logger, wandb_logger)
