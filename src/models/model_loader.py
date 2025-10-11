# ==================== 모델 로더 모듈 ==================== #
"""
사전학습 모델 로더

HuggingFace 모델과 토크나이저를 로드하고 초기화하는 모듈
- 모델 자동 감지 및 로드
- 특수 토큰 추가
- 디바이스 배치
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from typing import Tuple, Optional, List
import warnings

# ---------------------- 서드파티 라이브러리 ---------------------- #
import torch
from transformers import (
    AutoModelForSeq2SeqLM,                              # Seq2Seq 모델 (BART, T5 등)
    AutoTokenizer,                                      # 토크나이저
    PreTrainedModel,                                    # 모델 베이스 클래스
    PreTrainedTokenizer                                 # 토크나이저 베이스 클래스
)
from omegaconf import DictConfig


# ==================== ModelLoader 클래스 정의 ==================== #
class ModelLoader:
    """모델 로더 클래스"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(self, config: DictConfig):
        """
        Args:
            config: 전체 Config (model, training 등 포함)
        """
        self.config = config                                # Config 저장
        self.model_config = config.model                    # 모델 Config 추출
        self.device = self._get_device()                    # 디바이스 결정


    # ---------------------- 디바이스 결정 함수 ---------------------- #
    def _get_device(self) -> torch.device:
        """
        사용할 디바이스 결정 (GPU/CPU)

        Returns:
            torch.device: 사용할 디바이스
        """
        # -------------- Config에서 디바이스 설정 확인 -------------- #
        if hasattr(self.config, 'training') and hasattr(self.config.training, 'device'):
            device_str = self.config.training.device        # Config에서 디바이스 가져오기

            # 'cuda'인 경우 GPU 사용 가능 여부 확인
            if device_str == 'cuda':
                if torch.cuda.is_available():               # GPU 사용 가능
                    return torch.device('cuda')
                else:                                       # GPU 사용 불가
                    warnings.warn("CUDA가 설정되었으나 사용 불가능합니다. CPU를 사용합니다.")
                    return torch.device('cpu')

            # 특정 GPU 지정 (예: 'cuda:0')
            elif device_str.startswith('cuda:'):
                if torch.cuda.is_available():
                    return torch.device(device_str)
                else:
                    warnings.warn(f"{device_str}가 설정되었으나 사용 불가능합니다. CPU를 사용합니다.")
                    return torch.device('cpu')

            # CPU 명시
            elif device_str == 'cpu':
                return torch.device('cpu')

        # -------------- 자동 감지 (Config에 없는 경우) -------------- #
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # ---------------------- 토크나이저 로드 함수 ---------------------- #
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        토크나이저 로드 및 특수 토큰 추가

        Returns:
            PreTrainedTokenizer: 초기화된 토크나이저
        """
        # -------------- 토크나이저 로드 -------------- #
        checkpoint = self.model_config.checkpoint           # 모델 체크포인트
        print(f"토크나이저 로딩: {checkpoint}")

        tokenizer = AutoTokenizer.from_pretrained(          # 토크나이저 로드
            checkpoint,
            use_fast=True                                   # Fast tokenizer 사용
        )

        # -------------- 특수 토큰 추가 -------------- #
        if hasattr(self.model_config, 'special_tokens') and self.model_config.special_tokens:
            special_tokens = list(self.model_config.special_tokens)  # 특수 토큰 리스트
            num_added = tokenizer.add_special_tokens(       # 특수 토큰 추가
                {'additional_special_tokens': special_tokens}
            )
            print(f"  → 특수 토큰 {num_added}개 추가됨")

        # -------------- 패딩 토큰 설정 -------------- #
        # BART 계열 모델은 pad_token이 없을 수 있으므로 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token       # EOS 토큰을 패딩으로 사용
            print(f"  → pad_token 설정: {tokenizer.pad_token}")

        return tokenizer                                    # 토크나이저 반환


    # ---------------------- 모델 로드 함수 ---------------------- #
    def load_model(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> PreTrainedModel:
        """
        사전학습 모델 로드

        Args:
            tokenizer: 토크나이저 (특수 토큰 추가 시 필요)

        Returns:
            PreTrainedModel: 로드된 모델
        """
        # -------------- 모델 로드 -------------- #
        checkpoint = self.model_config.checkpoint           # 모델 체크포인트
        print(f"모델 로딩: {checkpoint}")

        model = AutoModelForSeq2SeqLM.from_pretrained(      # Seq2Seq 모델 로드
            checkpoint
        )

        # -------------- 특수 토큰에 따른 임베딩 크기 조정 -------------- #
        if tokenizer is not None:
            vocab_size = len(tokenizer)                     # 토크나이저 어휘 크기
            model_vocab_size = model.config.vocab_size      # 모델 어휘 크기

            # 어휘 크기가 다른 경우 임베딩 리사이즈
            if vocab_size != model_vocab_size:
                print(f"  → 임베딩 크기 조정: {model_vocab_size} → {vocab_size}")
                model.resize_token_embeddings(vocab_size)   # 임베딩 리사이즈

        # -------------- 디바이스 배치 -------------- #
        model = model.to(self.device)                       # 모델을 디바이스로 이동
        print(f"  → 디바이스: {self.device}")

        # -------------- 모델 파라미터 정보 출력 -------------- #
        total_params = sum(p.numel() for p in model.parameters())  # 전체 파라미터 수
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 학습 가능 파라미터

        print(f"  → 전체 파라미터: {total_params:,}")
        print(f"  → 학습 가능 파라미터: {trainable_params:,}")

        return model                                        # 모델 반환


    # ---------------------- 모델 및 토크나이저 로드 함수 ---------------------- #
    def load_model_and_tokenizer(
        self
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        모델과 토크나이저를 함께 로드

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: (모델, 토크나이저)
        """
        print("=" * 60)
        print("모델 및 토크나이저 로딩 시작")
        print("=" * 60)

        # -------------- 1. 토크나이저 로드 -------------- #
        tokenizer = self.load_tokenizer()                   # 토크나이저 로드

        print()  # 빈 줄

        # -------------- 2. 모델 로드 -------------- #
        model = self.load_model(tokenizer)                  # 모델 로드

        print("=" * 60)
        print("✅ 모델 및 토크나이저 로딩 완료")
        print("=" * 60)

        return model, tokenizer                             # 모델, 토크나이저 반환


# ==================== 편의 함수 ==================== #
# ---------------------- Config에서 모델 로드 함수 ---------------------- #
def load_model_and_tokenizer(
    config: DictConfig
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Config에서 모델과 토크나이저 로드 편의 함수

    Args:
        config: 전체 Config

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: (모델, 토크나이저)
    """
    loader = ModelLoader(config)                            # 모델 로더 생성
    return loader.load_model_and_tokenizer()                # 모델 및 토크나이저 로드
