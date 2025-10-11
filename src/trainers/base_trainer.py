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
