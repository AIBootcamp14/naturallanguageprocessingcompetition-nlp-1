# ==================== Config 로더 모듈 ==================== #
"""
계층적 Config 로딩 시스템

config 파일들을 계층적으로 병합하여 최종 설정을 생성
- base/default.yaml: 기본 설정
- base/{model_type}.yaml: 모델 타입별 설정
- models/{model_name}.yaml: 모델별 설정
- strategies/*.yaml: 전략별 설정
- experiments/{experiment}.yaml: 실험 설정
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import os
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------- 서드파티 라이브러리 ---------------------- #
from omegaconf import OmegaConf, DictConfig

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.utils.core.common import require_file


# ==================== ConfigLoader 클래스 정의 ==================== #
class ConfigLoader:
    """계층적 Config 로더 클래스"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(self, config_root: str = "configs"):
        """
        Args:
            config_root: config 파일들이 위치한 루트 디렉토리
        """
        self.config_root = Path(config_root)            # config 루트 경로 저장

        # config 루트 디렉토리 존재 확인
        if not self.config_root.exists():
            raise FileNotFoundError(
                f"Config 루트 디렉토리가 없습니다: {self.config_root}\n"
                f"현재 작업 디렉토리: {os.getcwd()}"
            )


    # ---------------------- 기본 설정 로드 ---------------------- #
    def load_base(self) -> DictConfig:
        """
        기본 설정 로드 (base/default.yaml)

        Returns:
            DictConfig: 기본 설정 객체
        """
        path = self.config_root / "base" / "default.yaml"   # 기본 설정 경로
        require_file(str(path), "base/default.yaml 파일이 필요합니다")  # 파일 존재 확인
        return OmegaConf.load(path)                         # YAML 파일 로드


    # ---------------------- 모델 타입별 설정 로드 ---------------------- #
    def load_model_type(self, model_type: str) -> DictConfig:
        """
        모델 타입별 설정 로드 (base/{model_type}.yaml)

        Args:
            model_type: 모델 타입 (encoder_decoder, causal_lm)

        Returns:
            DictConfig: 모델 타입별 설정 (없으면 빈 설정)
        """
        path = self.config_root / "base" / f"{model_type}.yaml"    # 모델 타입 설정 경로

        # 파일이 존재하면 로드
        if path.exists():
            return OmegaConf.load(path)                     # YAML 파일 로드

        # 파일이 없으면 빈 설정 반환
        return OmegaConf.create({})                         # 빈 설정 생성


    # ---------------------- 모델별 설정 로드 ---------------------- #
    def load_model(self, model_name: str) -> DictConfig:
        """
        모델별 설정 로드 (models/{model_name}.yaml)

        Args:
            model_name: 모델 이름 (kobart, llama_3.2_3b 등)

        Returns:
            DictConfig: 모델별 설정 (없으면 빈 설정)
        """
        path = self.config_root / "models" / f"{model_name}.yaml"  # 모델 설정 경로

        # 파일이 존재하면 로드
        if path.exists():
            return OmegaConf.load(path)                     # YAML 파일 로드

        # 파일이 없으면 빈 설정 반환
        return OmegaConf.create({})                         # 빈 설정 생성


    # ---------------------- 전략별 설정 로드 ---------------------- #
    def load_strategy(self, strategy_name: str) -> DictConfig:
        """
        전략별 설정 로드 (strategies/{strategy_name}.yaml)

        Args:
            strategy_name: 전략 이름 (data_augmentation, ensemble 등)

        Returns:
            DictConfig: 전략별 설정 (없으면 빈 설정)
        """
        path = self.config_root / "strategies" / f"{strategy_name}.yaml"   # 전략 설정 경로

        # 파일이 존재하면 로드
        if path.exists():
            return OmegaConf.load(path)                     # YAML 파일 로드

        # 파일이 없으면 빈 설정 반환
        return OmegaConf.create({})                         # 빈 설정 생성


    # ---------------------- 실험 설정 로드 ---------------------- #
    def load_experiment(self, experiment_name: str) -> DictConfig:
        """
        실험 설정 로드 (experiments/{experiment_name}.yaml)

        Args:
            experiment_name: 실험 이름 (baseline_kobart, llama_finetune 등)

        Returns:
            DictConfig: 실험 설정 객체
        """
        path = self.config_root / "experiments" / f"{experiment_name}.yaml"    # 실험 설정 경로
        require_file(str(path), f"experiments/{experiment_name}.yaml 파일이 필요합니다")  # 파일 존재 확인
        return OmegaConf.load(path)                         # YAML 파일 로드


    # ---------------------- 모든 Config 병합 ---------------------- #
    def merge_configs(self, experiment_name: str) -> DictConfig:
        """
        모든 Config를 계층적으로 병합

        병합 순서:
        1. base/default.yaml
        2. base/{model_type}.yaml
        3. models/{model_name}.yaml
        4. strategies/*.yaml (활성화된 전략만)
        5. experiments/{experiment_name}.yaml

        Args:
            experiment_name: 실험 이름

        Returns:
            DictConfig: 병합된 최종 설정
        """
        # -------------- 1. 기본 설정 로드 -------------- #
        configs = [self.load_base()]                        # 기본 설정을 리스트에 추가

        # -------------- 2. 실험 설정 로드 -------------- #
        exp_config = self.load_experiment(experiment_name) # 실험 설정 로드

        # -------------- 3. 모델 타입별 설정 로드 -------------- #
        model_type = exp_config.get('model', {}).get('type', 'encoder_decoder')  # 모델 타입 추출
        configs.append(self.load_model_type(model_type))   # 모델 타입 설정 추가

        # -------------- 4. 모델별 설정 로드 -------------- #
        model_name = exp_config.get('model', {}).get('name', '')   # 모델 이름 추출
        if model_name:
            configs.append(self.load_model(model_name))     # 모델 설정 추가

        # -------------- 5. 전략별 설정 로드 -------------- #
        strategies = exp_config.get('strategies', {})       # 전략 설정 추출

        # 활성화된 전략만 로드
        for strategy_name, enabled in strategies.items():
            if enabled:                                     # 전략이 활성화된 경우
                configs.append(self.load_strategy(strategy_name))   # 전략 설정 추가

        # -------------- 6. 실험 설정 추가 -------------- #
        configs.append(exp_config)                          # 실험 설정을 마지막에 추가

        # -------------- 7. 모든 설정 병합 -------------- #
        # 나중에 추가된 설정이 우선순위가 높음
        merged = OmegaConf.merge(*configs)                  # 모든 설정 병합

        return merged                                       # 병합된 설정 반환


# ==================== 편의 함수 ==================== #
# ---------------------- Config 로드 함수 ---------------------- #
def load_config(experiment_name: str, config_root: str = "configs") -> DictConfig:
    """
    Config 로드 편의 함수

    Args:
        experiment_name: 실험 이름
        config_root: config 루트 디렉토리

    Returns:
        DictConfig: 병합된 최종 설정

    Example:
        >>> config = load_config("baseline_kobart")
        >>> print(config.model.name)
        kobart
    """
    loader = ConfigLoader(config_root)                      # Config 로더 생성
    return loader.merge_configs(experiment_name)            # 설정 병합 및 반환
