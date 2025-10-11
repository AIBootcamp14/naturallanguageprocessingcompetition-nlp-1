"""
LLM 파인튜닝 Trainer

PRD 08: LLM 파인튜닝 전략 구현
- Causal LM 전용 Trainer
- QLoRA 최적화 (paged_adamw_32bit)
- Gradient checkpointing 지원
"""

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from omegaconf import DictConfig
from typing import Dict, Any, Optional


class LLMTrainer:
    """LLM 파인튜닝 Trainer (Causal LM 전용)"""

    def __init__(
        self,
        config: DictConfig,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        use_wandb: bool = True,
        logger=None
    ):
        """
        Args:
            config: 학습 설정
            model: Causal LM 모델
            tokenizer: Tokenizer
            train_dataset: 학습 데이터셋
            eval_dataset: 검증 데이터셋
            use_wandb: WandB 사용 여부
            logger: Logger 인스턴스
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.use_wandb = use_wandb
        self.logger = logger

        # WandB Logger 초기화
        self.wandb_logger = None
        if use_wandb and hasattr(config, 'wandb') and config.wandb.enabled:
            from src.logging.wandb_logger import WandbLogger
            self.wandb_logger = WandbLogger(
                project_name=config.wandb.project,
                entity=config.wandb.entity,
                experiment_name=config.experiment.name,
                config=dict(config),
                tags=config.experiment.tags
            )

        # Training Arguments 생성
        self.training_args = self._create_training_args()

        # Data Collator 생성
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM은 MLM 사용 안 함
        )

        # Trainer 생성
        self.trainer = self._create_trainer()

        self._log("LLMTrainer 초기화 완료")

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def _create_training_args(self) -> TrainingArguments:
        """
        HuggingFace TrainingArguments 생성 (Causal LM 전용)

        PRD 08:
        - QLoRA 최적화: paged_adamw_32bit
        - Gradient checkpointing: 메모리 절약
        - bf16/fp16: 모델별 dtype 매칭
        - eval_loss 기반 best model 선택
        """
        self._log("Training Arguments 생성 중...")

        # Output 디렉토리
        output_dir = f"outputs/{self.config.experiment.name}"

        # dtype 결정 (Llama: bf16, Qwen: fp16)
        use_bf16 = True
        use_fp16 = False
        if 'qwen' in self.config.model.checkpoint.lower():
            use_bf16 = False
            use_fp16 = True
            self._log("  - Qwen 모델: fp16 사용")
        else:
            self._log("  - Llama 모델: bf16 사용")

        training_args = TrainingArguments(
            # 출력 설정
            output_dir=output_dir,
            overwrite_output_dir=True,

            # 학습 하이퍼파라미터
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.get('eval_batch_size', self.config.training.batch_size),
            gradient_accumulation_steps=self.config.training.get('gradient_accumulation_steps', 8),

            # Learning Rate
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.get('lr_scheduler_type', 'cosine'),
            warmup_ratio=self.config.training.get('warmup_ratio', 0.1),
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.get('max_grad_norm', 1.2),  # PRD 08: 최적화

            # 평가 및 저장
            eval_strategy='epoch' if self.eval_dataset else 'no',
            save_strategy='epoch',
            save_total_limit=self.config.training.get('save_total_limit', 2),
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model='eval_loss',  # PRD 08: Causal LM은 loss 사용
            greater_is_better=False,  # Loss는 낮을수록 좋음

            # 로깅
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.config.training.get('logging_steps', 10),
            report_to=['wandb'] if self.use_wandb and self.wandb_logger else [],

            # 최적화 (PRD 08: QLoRA 최적화)
            optim="paged_adamw_32bit",  # QLoRA 최적화
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=self.config.training.get('gradient_checkpointing', True),
            gradient_checkpointing_kwargs={"use_reentrant": False},  # PyTorch 2.0+

            # 기타
            dataloader_num_workers=self.config.training.get('num_workers', 4),
            remove_unused_columns=False,  # Causal LM에서 필요
        )

        self._log("Training Arguments 생성 완료")
        self._log(f"  - Epochs: {training_args.num_train_epochs}")
        self._log(f"  - Batch size: {training_args.per_device_train_batch_size}")
        self._log(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
        self._log(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        self._log(f"  - Learning rate: {training_args.learning_rate}")
        self._log(f"  - Optimizer: {training_args.optim}")
        self._log(f"  - Gradient checkpointing: {training_args.gradient_checkpointing}")

        return training_args

    def _create_trainer(self) -> Trainer:
        """HuggingFace Trainer 생성"""
        self._log("Trainer 생성 중...")

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        self._log("Trainer 생성 완료")
        return trainer

    def train(self) -> Dict[str, Any]:
        """
        모델 학습 실행

        Returns:
            학습 결과 딕셔너리
        """
        self._log("=" * 60)
        self._log("모델 학습 시작")
        self._log("=" * 60)

        # WandB Run 시작
        if self.wandb_logger:
            self.wandb_logger.init_run()

        try:
            # 학습 실행
            train_result = self.trainer.train()

            # 최종 모델 저장
            final_model_path = f"outputs/{self.config.experiment.name}/final_model"
            self.trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)

            self._log("=" * 60)
            self._log("모델 학습 완료")
            self._log(f"최종 모델 저장: {final_model_path}")
            self._log("=" * 60)

            # 평가 실행
            eval_metrics = {}
            if self.eval_dataset:
                self._log("\n평가 시작...")
                eval_metrics = self.evaluate()
                self._log("\n평가 메트릭:")
                for key, value in eval_metrics.items():
                    self._log(f"  {key}: {value:.4f}")

            # 결과 반환
            return {
                'train_loss': train_result.training_loss,
                'train_runtime': train_result.metrics.get('train_runtime', 0),
                'eval_metrics': eval_metrics,
                'final_model_path': final_model_path
            }

        finally:
            # WandB Run 종료
            if self.wandb_logger:
                self.wandb_logger.finish()

    def evaluate(self) -> Dict[str, float]:
        """
        모델 평가 실행

        Returns:
            평가 메트릭 딕셔너리
        """
        if not self.eval_dataset:
            self._log("검증 데이터셋이 없습니다. 평가를 건너뜁니다.")
            return {}

        self._log("모델 평가 중...")

        eval_result = self.trainer.evaluate()

        self._log("평가 완료")

        return eval_result


def create_llm_trainer(
    config: DictConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    use_wandb: bool = True,
    logger=None
) -> LLMTrainer:
    """
    편의 함수: LLMTrainer 생성

    Args:
        config: 학습 설정
        model: Causal LM 모델
        tokenizer: Tokenizer
        train_dataset: 학습 데이터셋
        eval_dataset: 검증 데이터셋
        use_wandb: WandB 사용 여부
        logger: Logger 인스턴스

    Returns:
        LLMTrainer 인스턴스
    """
    return LLMTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        use_wandb=use_wandb,
        logger=logger
    )
