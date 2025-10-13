# ==================== BaseTrainer 추상 클래스 ==================== #
"""
모든 Trainer의 기본 클래스

모든 Trainer는 이 클래스를 상속받아 train()과 save_results() 메서드를 구현해야 함
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------- 서드파티 라이브러리 ---------------------- #
import pandas as pd


# ==================== BaseTrainer ==================== #
class BaseTrainer(ABC):
    """모든 Trainer의 기본 추상 클래스"""

    def __init__(self, args, logger, wandb_logger=None):
        """
        Args:
            args: 명령행 인자 (ArgumentParser)
            logger: Logger 인스턴스
            wandb_logger: WandB Logger (선택)
        """
        self.args = args
        self.logger = logger
        self.wandb_logger = wandb_logger

        # 출력 디렉토리 생성
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        학습 메인 로직 (서브클래스에서 반드시 구현)

        Returns:
            학습 결과 딕셔너리
        """
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Any]):
        """
        결과 저장 (서브클래스에서 반드시 구현)

        Args:
            results: 학습 결과 딕셔너리
        """
        pass

    def log(self, message: str, level: str = 'INFO'):
        """
        통합 로깅 (Logger + WandB)

        Args:
            message: 로그 메시지
            level: 로그 레벨 (INFO, WARNING, ERROR 등)
        """
        # Logger에 기록
        self.logger.write(message)

        # WandB에도 기록 (활성화된 경우)
        if self.wandb_logger:
            self.wandb_logger.log_text(message, level=level)

    def load_data(self):
        """
        공통 데이터 로딩 로직

        Returns:
            train_df, eval_df (pandas DataFrame)
        """
        # 데이터 경로
        train_path = getattr(self.args, 'train_data', 'data/raw/train.csv')
        dev_path = getattr(self.args, 'dev_data', 'data/raw/dev.csv')

        # CSV 로드
        train_df = pd.read_csv(train_path)
        eval_df = pd.read_csv(dev_path)

        # 디버그 모드: 데이터 축소
        if getattr(self.args, 'debug', False):
            train_df = train_df.head(100)
            eval_df = eval_df.head(20)
            self.log(f"⚠️ 디버그 모드: 학습 {len(train_df)}개, 검증 {len(eval_df)}개")
        else:
            self.log(f"✅ 학습 데이터: {len(train_df)}개")
            self.log(f"✅ 검증 데이터: {len(eval_df)}개")

        # max_train_samples 옵션 적용 (빠른 테스트용)
        if hasattr(self.args, 'max_train_samples') and self.args.max_train_samples is not None:
            if self.args.max_train_samples < len(train_df):
                train_df = train_df.head(self.args.max_train_samples)
                self.log(f"⚙️ max_train_samples 적용: 학습 데이터 {len(train_df)}개로 제한")

        return train_df, eval_df

    def get_config_path(self, model_name: str) -> str:
        """
        모델별 Config 경로 반환

        Args:
            model_name: 모델 이름 (예: 'kobart', 'llama-3.2-korean-3b')

        Returns:
            Config 파일 경로
        """
        # 모델명 정규화 (하이픈 → 언더스코어)
        normalized_name = model_name.replace('-', '_')

        # Config 경로 우선순위
        config_paths = [
            f"configs/models/{normalized_name}.yaml",
            f"configs/experiments/{self.args.experiment_name}.yaml",
            "configs/train_config.yaml"  # Fallback
        ]

        for path in config_paths:
            if Path(path).exists():
                return path

        # 기본값
        return "configs/train_config.yaml"

    def create_experiment_name(self) -> str:
        """
        실험명 자동 생성

        Returns:
            실험명 (예: 'single_kobart_20251011_183045')
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = getattr(self.args, 'mode', 'single')
        model_name = getattr(self.args, 'models', ['default'])[0].replace('-', '_')

        return f"{mode}_{model_name}_{timestamp}"

    def _override_config(self, config):
        """
        명령행 인자로 Config 오버라이드 (공통 메서드)

        Args:
            config: Config 객체
        """
        # 기본 학습 설정
        if hasattr(self.args, 'epochs') and self.args.epochs is not None:
            config.training.epochs = self.args.epochs

        if hasattr(self.args, 'batch_size') and self.args.batch_size is not None:
            config.training.batch_size = self.args.batch_size

        if hasattr(self.args, 'learning_rate') and self.args.learning_rate is not None:
            config.training.learning_rate = self.args.learning_rate

        # 고급 학습 설정
        if hasattr(self.args, 'gradient_accumulation_steps') and self.args.gradient_accumulation_steps is not None:
            config.training.gradient_accumulation_steps = self.args.gradient_accumulation_steps

        if hasattr(self.args, 'warmup_ratio') and self.args.warmup_ratio is not None:
            config.training.warmup_ratio = self.args.warmup_ratio

        if hasattr(self.args, 'weight_decay') and self.args.weight_decay is not None:
            config.training.weight_decay = self.args.weight_decay

        if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm is not None:
            config.training.max_grad_norm = self.args.max_grad_norm

        if hasattr(self.args, 'label_smoothing') and self.args.label_smoothing is not None:
            config.training.label_smoothing = self.args.label_smoothing

        # Fine-tuning 전략
        if hasattr(self.args, 'use_full_finetuning') and self.args.use_full_finetuning:
            config.use_full_finetuning = True

        if hasattr(self.args, 'lora_rank') and self.args.lora_rank is not None:
            if hasattr(config.model, 'lora'):
                config.model.lora.r = self.args.lora_rank

        # 생성 파라미터
        if hasattr(self.args, 'num_beams') and self.args.num_beams is not None:
            if not hasattr(config, 'inference'):
                from omegaconf import OmegaConf
                config.inference = OmegaConf.create({})
            config.inference.num_beams = self.args.num_beams

        if hasattr(self.args, 'temperature') and self.args.temperature is not None:
            if not hasattr(config, 'inference'):
                from omegaconf import OmegaConf
                config.inference = OmegaConf.create({})
            config.inference.temperature = self.args.temperature

        if hasattr(self.args, 'top_p') and self.args.top_p is not None:
            if not hasattr(config, 'inference'):
                from omegaconf import OmegaConf
                config.inference = OmegaConf.create({})
            config.inference.top_p = self.args.top_p

        if hasattr(self.args, 'top_k') and self.args.top_k is not None:
            if not hasattr(config, 'inference'):
                from omegaconf import OmegaConf
                config.inference = OmegaConf.create({})
            config.inference.top_k = self.args.top_k

        if hasattr(self.args, 'repetition_penalty') and self.args.repetition_penalty is not None:
            if not hasattr(config, 'inference'):
                from omegaconf import OmegaConf
                config.inference = OmegaConf.create({})
            config.inference.repetition_penalty = self.args.repetition_penalty

        if hasattr(self.args, 'length_penalty') and self.args.length_penalty is not None:
            if not hasattr(config, 'inference'):
                from omegaconf import OmegaConf
                config.inference = OmegaConf.create({})
            config.inference.length_penalty = self.args.length_penalty

        if hasattr(self.args, 'no_repeat_ngram_size') and self.args.no_repeat_ngram_size is not None:
            if not hasattr(config, 'inference'):
                from omegaconf import OmegaConf
                config.inference = OmegaConf.create({})
            config.inference.no_repeat_ngram_size = self.args.no_repeat_ngram_size
