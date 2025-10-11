# ==================== SingleModelTrainer ==================== #
"""
단일 모델 학습 Trainer

가장 기본적인 학습 모드로, 하나의 모델을 학습하고 평가
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import json
from pathlib import Path

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.trainers.base_trainer import BaseTrainer
from src.config import load_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer


# ==================== SingleModelTrainer ==================== #
class SingleModelTrainer(BaseTrainer):
    """단일 모델 학습 Trainer"""

    def train(self):
        """
        단일 모델 학습 실행

        Returns:
            dict: 학습 결과
                - mode: 'single'
                - model: 모델 이름
                - results: 학습 결과
                - model_path: 저장된 모델 경로
        """
        self.log("=" * 60)
        self.log("🚀 SINGLE MODEL 모드 학습 시작")
        self.log(f"📋 모델: {self.args.models[0]}")
        self.log("=" * 60)

        # 1. 데이터 로드
        self.log("\n[1/5] 데이터 로딩...")
        train_df, eval_df = self.load_data()

        # 2. Config 로드
        self.log("\n[2/5] Config 로딩...")
        config_path = self.get_config_path(self.args.models[0])
        config = load_config(config_path)

        # 명령행 인자로 Config 오버라이드
        self._override_config(config)

        self.log(f"  ✅ Config 로드 완료: {config_path}")

        # 3. 모델 및 토크나이저 로드
        self.log("\n[3/5] 모델 로딩...")
        model, tokenizer = load_model_and_tokenizer(config, logger=self.logger)
        self.log("  ✅ 모델 로드 완료")

        # 4. Dataset 생성
        self.log("\n[4/5] Dataset 생성...")

        # 모델 타입 가져오기 (기본값: encoder_decoder)
        model_type = config.model.get('type', 'encoder_decoder')

        train_dataset = DialogueSummarizationDataset(
            dialogues=train_df['dialogue'].tolist(),
            summaries=train_df['summary'].tolist(),
            tokenizer=tokenizer,
            encoder_max_len=config.tokenizer.encoder_max_len,
            decoder_max_len=config.tokenizer.decoder_max_len,
            preprocess=True,
            model_type=model_type  # PRD 08: LLM 지원
        )

        eval_dataset = DialogueSummarizationDataset(
            dialogues=eval_df['dialogue'].tolist(),
            summaries=eval_df['summary'].tolist(),
            tokenizer=tokenizer,
            encoder_max_len=config.tokenizer.encoder_max_len,
            decoder_max_len=config.tokenizer.decoder_max_len,
            preprocess=True,
            model_type=model_type  # PRD 08: LLM 지원
        )

        self.log(f"  ✅ 학습 Dataset: {len(train_dataset)}개")
        self.log(f"  ✅ 검증 Dataset: {len(eval_dataset)}개")

        # 5. Trainer 생성 및 학습
        self.log("\n[5/5] 학습 시작...")
        trainer = create_trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            use_wandb=getattr(self.args, 'use_wandb', False),
            logger=self.logger
        )

        # 학습 실행
        train_results = trainer.train()

        # 결과 수집
        results = {
            'mode': 'single',
            'model': self.args.models[0],
            'train_results': train_results,
            'model_path': str(self.output_dir / 'final_model')
        }

        # 최종 평가 메트릭 추가
        if hasattr(trainer, 'state') and trainer.state.log_history:
            eval_metrics = self._extract_eval_metrics(trainer.state.log_history)
            results['eval_metrics'] = eval_metrics

        self.log("\n" + "=" * 60)
        self.log("✅ SINGLE MODEL 학습 완료!")

        if 'eval_metrics' in results and results['eval_metrics']:
            self.log("\n📊 최종 평가 결과:")
            for key, value in results['eval_metrics'].items():
                if 'rouge' in key.lower():
                    self.log(f"  {key}: {value:.4f}")

        self.log("=" * 60)

        return results

    def save_results(self, results):
        """
        결과 저장

        Args:
            results: 학습 결과 딕셔너리
        """
        result_path = self.output_dir / "single_model_results.json"

        # 저장 가능한 형태로 변환
        saveable_results = {
            'mode': results['mode'],
            'model': results['model'],
            'model_path': results['model_path'],
            'eval_metrics': results.get('eval_metrics', {})
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        self.log(f"\n💾 결과 저장: {result_path}")

    def _override_config(self, config):
        """
        명령행 인자로 Config 오버라이드

        Args:
            config: Config 객체
        """
        # Epochs
        if hasattr(self.args, 'epochs') and self.args.epochs is not None:
            config.training.epochs = self.args.epochs
            self.log(f"  ⚙️ Epochs 오버라이드: {self.args.epochs}")

        # Batch size
        if hasattr(self.args, 'batch_size') and self.args.batch_size is not None:
            config.training.batch_size = self.args.batch_size
            self.log(f"  ⚙️ Batch size 오버라이드: {self.args.batch_size}")

        # Learning rate
        if hasattr(self.args, 'learning_rate') and self.args.learning_rate is not None:
            config.training.learning_rate = self.args.learning_rate
            self.log(f"  ⚙️ Learning rate 오버라이드: {self.args.learning_rate}")

    def _extract_eval_metrics(self, log_history):
        """
        학습 로그에서 평가 메트릭 추출

        Args:
            log_history: Trainer의 로그 히스토리

        Returns:
            dict: 평가 메트릭
        """
        eval_metrics = {}

        # 마지막 eval 로그 찾기
        for log_entry in reversed(log_history):
            if 'eval_loss' in log_entry:
                for key, value in log_entry.items():
                    if key.startswith('eval_'):
                        eval_metrics[key] = value
                break

        return eval_metrics
