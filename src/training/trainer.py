# ==================== 모델 학습 모듈 ==================== #
"""
모델 학습 시스템

HuggingFace Seq2SeqTrainer를 래핑한 학습 시스템
- Config 기반 학습 설정
- WandB 로깅 통합
- 체크포인트 자동 관리
- ROUGE 평가 통합
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from typing import Optional, Dict, Any
from pathlib import Path
import warnings

# ---------------------- 서드파티 라이브러리 ---------------------- #
import torch
from transformers import (
    Seq2SeqTrainer,                                     # Seq2Seq 학습용 Trainer
    Seq2SeqTrainingArguments,                           # 학습 인자
    EarlyStoppingCallback,                              # 조기 종료
    PreTrainedModel,                                    # 모델 타입
    PreTrainedTokenizer                                 # 토크나이저 타입
)
from torch.utils.data import Dataset
from omegaconf import DictConfig
import numpy as np

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.evaluation import RougeCalculator              # ROUGE 계산
from src.logging.wandb_logger import WandbLogger        # WandB 로거
from src.utils.core.common import ensure_dir           # 디렉토리 생성 유틸


# ==================== ModelTrainer 클래스 정의 ==================== #
class ModelTrainer:
    """모델 학습 클래스"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(
        self,
        config: DictConfig,                             # 전체 Config
        model: PreTrainedModel,                         # 학습할 모델
        tokenizer: PreTrainedTokenizer,                 # 토크나이저
        train_dataset: Dataset,                         # 학습 데이터셋
        eval_dataset: Optional[Dataset] = None,         # 검증 데이터셋 (선택적)
        use_wandb: bool = True,                         # WandB 사용 여부
        logger=None                                     # Logger 인스턴스 (선택적)
    ):
        """
        Args:
            config: 전체 Config (training, model, wandb 등 포함)
            model: 학습할 모델
            tokenizer: 토크나이저
            train_dataset: 학습 데이터셋
            eval_dataset: 검증 데이터셋
            use_wandb: WandB 로깅 사용 여부
            logger: Logger 인스턴스 (선택적)
        """
        self.config = config                            # Config 저장
        self.model = model                              # 모델 저장
        self.tokenizer = tokenizer                      # 토크나이저 저장
        self.train_dataset = train_dataset              # 학습 데이터셋 저장
        self.eval_dataset = eval_dataset                # 검증 데이터셋 저장
        self.use_wandb = use_wandb                      # WandB 사용 여부 저장
        self.logger = logger                            # Logger 저장

        # -------------- WandB Logger 초기화 -------------- #
        self.wandb_logger = None                        # WandB Logger 초기값
        if use_wandb and hasattr(config, 'wandb') and config.wandb.enabled:
            self.wandb_logger = WandbLogger(            # WandB Logger 생성
                project_name=config.wandb.project,
                entity=config.wandb.entity,
                experiment_name=config.experiment.name,
                config=dict(config),                    # OmegaConf를 dict로 변환
                tags=config.experiment.tags
            )

        # -------------- ROUGE Calculator 초기화 -------------- #
        self.rouge_calculator = RougeCalculator()       # ROUGE 계산기 생성

        # -------------- 학습 인자 생성 -------------- #
        self.training_args = self._create_training_args()  # 학습 인자 생성

        # -------------- Trainer 초기화 -------------- #
        self.trainer = None                             # Trainer 초기값 (나중에 생성)


    # ---------------------- 학습 인자 생성 함수 ---------------------- #
    def _create_training_args(self) -> Seq2SeqTrainingArguments:
        """
        HuggingFace 학습 인자 생성

        Returns:
            Seq2SeqTrainingArguments: 학습 인자 객체
        """
        # -------------- Config에서 학습 설정 추출 -------------- #
        train_cfg = self.config.training                # 학습 Config
        model_cfg = self.config.model                   # 모델 Config

        # -------------- 출력 디렉토리 설정 -------------- #
        output_dir = Path(train_cfg.output_dir) / self.config.experiment.name  # 실험별 출력 디렉토리
        ensure_dir(output_dir)                          # 디렉토리 생성

        # -------------- 학습 인자 생성 -------------- #
        args = Seq2SeqTrainingArguments(
            # ---------- 기본 설정 ---------- #
            output_dir=str(output_dir),                 # 출력 디렉토리
            overwrite_output_dir=True,                  # 기존 출력 덮어쓰기

            # ---------- 학습 하이퍼파라미터 ---------- #
            num_train_epochs=train_cfg.epochs,          # 에포크 수
            per_device_train_batch_size=train_cfg.batch_size,  # 학습 배치 크기
            per_device_eval_batch_size=train_cfg.batch_size,   # 평가 배치 크기
            learning_rate=train_cfg.learning_rate,      # 학습률
            weight_decay=train_cfg.get('weight_decay', 0.01),  # 가중치 감쇠
            warmup_steps=train_cfg.get('warmup_steps', 500),   # Warmup 스텝

            # ---------- 평가 및 저장 ---------- #
            eval_strategy='epoch',                      # 에포크마다 평가
            save_strategy='epoch',                      # 에포크마다 저장
            save_total_limit=train_cfg.get('save_total_limit', 3),  # 최대 저장 개수
            load_best_model_at_end=True,                # 최종에 최상 모델 로드
            metric_for_best_model='eval_rouge_sum',     # 최상 모델 선택 메트릭

            # ---------- 로깅 ---------- #
            logging_dir=str(output_dir / 'logs'),       # 로깅 디렉토리
            logging_steps=train_cfg.get('logging_steps', 100),  # 로깅 주기
            report_to=['wandb'] if self.use_wandb and self.wandb_logger else [],  # 리포팅 대상

            # ---------- Seq2Seq 특화 설정 ---------- #
            predict_with_generate=True,                 # 생성 모드로 평가
            generation_max_length=self.config.get('tokenizer', {}).get('decoder_max_len', 100),    # 생성 최대 길이
            generation_num_beams=self.config.get('inference', {}).get('num_beams', 4),  # Beam 개수

            # ---------- 기타 ---------- #
            fp16=train_cfg.get('fp16', False),          # Config에서 FP16 설정 (기본값: False, QLoRA와 호환성 문제 방지)
            bf16=train_cfg.get('bf16', False),          # Config에서 BFloat16 설정 (기본값: False)
            dataloader_num_workers=train_cfg.get('num_workers', 4),  # 데이터로더 워커 수
            remove_unused_columns=True,                 # 사용하지 않는 컬럼 제거
            push_to_hub=False,                          # HuggingFace Hub 업로드 안 함
        )

        return args                                     # 학습 인자 반환


    # ---------------------- ROUGE 평가 함수 ---------------------- #
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        평가 메트릭 계산 (ROUGE)

        Args:
            eval_preds: (predictions, labels) 튜플

        Returns:
            Dict[str, float]: {'rouge1': 0.5, 'rouge2': 0.3, ...}
        """
        # -------------- 예측 결과 추출 -------------- #
        predictions, labels = eval_preds               # 예측과 레이블 분리

        # -------------- 토큰 ID를 텍스트로 디코딩 -------------- #
        # -100을 패딩 토큰으로 변경
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # 디코딩
        decoded_preds = self.tokenizer.batch_decode(    # 예측 디코딩
            predictions,
            skip_special_tokens=True                    # 특수 토큰 제외
        )
        decoded_labels = self.tokenizer.batch_decode(   # 레이블 디코딩
            labels,
            skip_special_tokens=True
        )

        # 앞뒤 공백 제거
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # -------------- ROUGE 점수 계산 -------------- #
        scores = self.rouge_calculator.calculate_batch( # 배치 ROUGE 계산
            decoded_preds,
            decoded_labels
        )

        # -------------- 결과 포맷팅 -------------- #
        result = {
            'rouge1': scores['rouge1']['fmeasure'],     # ROUGE-1 F1
            'rouge2': scores['rouge2']['fmeasure'],     # ROUGE-2 F1
            'rougeL': scores['rougeL']['fmeasure'],     # ROUGE-L F1
            'rouge_sum': scores['rouge_sum']['fmeasure']  # ROUGE Sum
        }

        return result                                   # 결과 반환


    # ---------------------- Trainer 생성 함수 ---------------------- #
    def _create_trainer(self) -> Seq2SeqTrainer:
        """
        HuggingFace Seq2SeqTrainer 생성

        Returns:
            Seq2SeqTrainer: 학습용 Trainer 객체
        """
        # -------------- Callback 설정 -------------- #
        callbacks = []                                  # Callback 리스트

        # Early Stopping 설정
        if self.config.training.get('early_stopping_patience'):
            callbacks.append(                           # Early Stopping 추가
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience
                )
            )

        # -------------- Trainer 생성 -------------- #
        trainer = Seq2SeqTrainer(
            model=self.model,                           # 모델
            args=self.training_args,                    # 학습 인자
            train_dataset=self.train_dataset,           # 학습 데이터셋
            eval_dataset=self.eval_dataset,             # 검증 데이터셋
            tokenizer=self.tokenizer,                   # 토크나이저
            compute_metrics=self.compute_metrics,       # 메트릭 계산 함수
            callbacks=callbacks                         # Callback 리스트
        )

        return trainer                                  # Trainer 반환


    # ---------------------- 학습 실행 함수 ---------------------- #
    def train(self) -> Dict[str, Any]:
        """
        모델 학습 실행

        Returns:
            Dict[str, Any]: 학습 결과 (메트릭, 체크포인트 경로 등)
        """
        msg = "=" * 60
        if self.logger:
            self.logger.write(msg)
            self.logger.write("모델 학습 시작")
            self.logger.write(msg)
        else:
            print(msg)
            print("모델 학습 시작")
            print(msg)

        # -------------- WandB 초기화 -------------- #
        if self.wandb_logger:
            self.wandb_logger.init_run()                # WandB Run 초기화

        # -------------- Trainer 생성 -------------- #
        self.trainer = self._create_trainer()           # Trainer 생성

        # -------------- 학습 실행 -------------- #
        msg = "\n학습 진행 중..."
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)
        train_result = self.trainer.train()             # 학습 실행

        # -------------- 최종 모델 저장 -------------- #
        msg = "\n최종 모델 저장 중..."
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)
        output_dir = Path(self.training_args.output_dir) / "final_model"  # 최종 모델 디렉토리
        self.trainer.save_model(str(output_dir))        # 모델 저장
        self.tokenizer.save_pretrained(str(output_dir))  # 토크나이저 저장

        msg = f"  → 모델 저장 위치: {output_dir}"
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

        # -------------- 평가 실행 (검증 데이터가 있는 경우) -------------- #
        eval_results = {}                               # 평가 결과 초기화
        if self.eval_dataset:
            msg = "\n최종 평가 중..."
            if self.logger:
                self.logger.write(msg)
            else:
                print(msg)
            eval_results = self.trainer.evaluate()      # 평가 실행

            # 결과 출력
            msg = "\n최종 평가 결과:"
            if self.logger:
                self.logger.write(msg)
            else:
                print(msg)
            for key, value in eval_results.items():
                if 'rouge' in key:                      # ROUGE 관련 메트릭만
                    msg = f"  {key}: {value:.4f}"
                    if self.logger:
                        self.logger.write(msg)
                    else:
                        print(msg)

        # -------------- 학습 결과 정리 -------------- #
        result = {
            'train_metrics': train_result.metrics,      # 학습 메트릭
            'eval_metrics': eval_results,               # 평가 메트릭
            'best_model_checkpoint': self.trainer.state.best_model_checkpoint,  # 최상 체크포인트
            'final_model_path': str(output_dir)         # 최종 모델 경로
        }

        # -------------- WandB 종료 -------------- #
        if self.wandb_logger:
            self.wandb_logger.finish()                  # WandB Run 종료

        msg = "=" * 60
        if self.logger:
            self.logger.write(msg)
            self.logger.write("✅ 학습 완료!")
            self.logger.write(msg)
        else:
            print(msg)
            print("✅ 학습 완료!")
            print(msg)

        return result                                   # 학습 결과 반환


    # ---------------------- 평가 함수 ---------------------- #
    def evaluate(self) -> Dict[str, float]:
        """
        모델 평가 실행

        Returns:
            Dict[str, float]: 평가 결과
        """
        if not self.trainer:                            # Trainer가 없는 경우
            self.trainer = self._create_trainer()       # Trainer 생성

        if not self.eval_dataset:                       # 검증 데이터가 없는 경우
            warnings.warn("검증 데이터셋이 없습니다.")
            return {}

        print("\n모델 평가 중...")
        eval_results = self.trainer.evaluate()          # 평가 실행

        return eval_results                             # 평가 결과 반환


# ==================== 편의 함수 ==================== #
# ---------------------- Trainer 생성 함수 ---------------------- #
def create_trainer(
    config: DictConfig,                                 # 전체 Config
    model: PreTrainedModel,                             # 모델
    tokenizer: PreTrainedTokenizer,                     # 토크나이저
    train_dataset: Dataset,                             # 학습 데이터셋
    eval_dataset: Optional[Dataset] = None,             # 검증 데이터셋
    use_wandb: bool = True,                             # WandB 사용 여부
    logger=None                                         # Logger 인스턴스 (선택적)
) -> ModelTrainer:
    """
    ModelTrainer 생성 편의 함수

    Args:
        config: 전체 Config
        model: 학습할 모델
        tokenizer: 토크나이저
        train_dataset: 학습 데이터셋
        eval_dataset: 검증 데이터셋
        use_wandb: WandB 로깅 사용 여부
        logger: Logger 인스턴스 (선택적)

    Returns:
        ModelTrainer: 생성된 Trainer 객체
    """
    return ModelTrainer(                                # Trainer 생성
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        use_wandb=use_wandb,
        logger=logger
    )
