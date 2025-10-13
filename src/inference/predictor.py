# ==================== 추론 시스템 모듈 ==================== #
"""
추론 시스템

학습된 모델로 대화 요약 예측을 수행하는 모듈
- 배치 추론
- 제출 파일 생성
- 생성 파라미터 설정
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from typing import List, Dict, Optional, Union
from pathlib import Path
from tqdm import tqdm

# ---------------------- 서드파티 라이브러리 ---------------------- #
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,                                    # 모델 타입
    PreTrainedTokenizer                                 # 토크나이저 타입
)
from omegaconf import DictConfig

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.data import InferenceDataset                   # 추론용 데이터셋
from src.utils.core.common import ensure_dir           # 디렉토리 생성 유틸


# ==================== Predictor 클래스 정의 ==================== #
class Predictor:
    """추론 시스템 클래스"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(
        self,
        model: PreTrainedModel,                         # 학습된 모델
        tokenizer: PreTrainedTokenizer,                 # 토크나이저
        config: Optional[DictConfig] = None,            # Config (선택적)
        device: Optional[torch.device] = None,          # 디바이스 (선택적)
        logger=None                                     # Logger 인스턴스 (선택적)
    ):
        """
        Args:
            model: 학습된 모델
            tokenizer: 토크나이저
            config: 전체 Config (inference 파라미터 포함)
            device: 추론 디바이스
            logger: Logger 인스턴스 (선택적)
        """
        self.model = model                              # 모델 저장
        self.tokenizer = tokenizer                      # 토크나이저 저장
        self.config = config                            # Config 저장
        self.logger = logger                            # Logger 저장

        # -------------- 디바이스 설정 -------------- #
        if device is None:                              # 디바이스가 지정되지 않은 경우
            self.device = next(model.parameters()).device  # 모델의 디바이스 사용
        else:
            self.device = device                        # 지정된 디바이스 사용
            self.model = self.model.to(device)          # 모델을 디바이스로 이동

        # -------------- 생성 파라미터 설정 -------------- #
        self.generation_config = self._setup_generation_config()  # 생성 설정

        # 평가 모드로 전환
        self.model.eval()                               # 평가 모드


    # ---------------------- 생성 파라미터 설정 함수 ---------------------- #
    def _setup_generation_config(self) -> Dict:
        """
        생성 파라미터 설정

        Returns:
            Dict: 생성 파라미터 딕셔너리
        """
        # -------------- 기본 생성 파라미터 -------------- #
        # NOTE: use max_new_tokens to control generated token count (safer for encoder-decoder)
        default_config = {
            'max_length': 512,                          # 최대 토큰 길이 (input+output) - 여유있게 설정
            'max_new_tokens': 128,                      # 새로 생성할 최대 토큰 수 (권장)
            'num_beams': 4,                             # Beam 개수
            'early_stopping': True,                     # 조기 종료
            'no_repeat_ngram_size': 2,                  # 반복 방지 n-gram 크기
            'length_penalty': 1.0,                      # 길이 페널티
        }

        # -------------- Config에서 생성 파라미터 로드 -------------- #
        if self.config and hasattr(self.config, 'inference'):
            inference_cfg = self.config.inference       # 추론 Config

            # Config 값으로 오버라이드 (베이스라인과 동일하게)
            # allow overriding both max_length and max_new_tokens from config
            default_config.update({
                'max_length': inference_cfg.get('generate_max_length', 512),
                'max_new_tokens': inference_cfg.get('generate_max_new_tokens', 128),
                'num_beams': inference_cfg.get('num_beams', 4),
                'early_stopping': inference_cfg.get('early_stopping', True),
                'no_repeat_ngram_size': inference_cfg.get('no_repeat_ngram_size', 2),
                'length_penalty': inference_cfg.get('length_penalty', 1.0),
            })

        return default_config                           # 생성 파라미터 반환


    # ---------------------- 단일 예측 함수 ---------------------- #
    def predict_single(
        self,
        dialogue: str,                                  # 입력 대화
        **generation_kwargs                             # 생성 파라미터 (선택적)
    ) -> str:
        """
        단일 대화 요약 예측

        Args:
            dialogue: 입력 대화 텍스트
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            str: 예측된 요약
        """
        # -------------- 토크나이징 -------------- #
        inputs = self.tokenizer(
            dialogue,
            max_length=self.config.get('tokenizer', {}).get('encoder_max_len', 512) if self.config else 512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 디바이스로 이동
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if k in ['input_ids', 'attention_mask']     # BART는 이 두 개만 사용
        }

        # -------------- 생성 파라미터 병합 -------------- #
        gen_config = {**self.generation_config, **generation_kwargs}  # 기본 + 오버라이드

        # -------------- 추론 실행 -------------- #
        with torch.no_grad():                           # 그래디언트 계산 비활성화
            outputs = self.model.generate(              # 텍스트 생성
                **inputs,
                **gen_config
            )

        # -------------- 디코딩 -------------- #
        summary = self.tokenizer.decode(                # 토큰 → 텍스트
            outputs[0],
            skip_special_tokens=True                    # 특수 토큰 제외
        )

        return summary.strip()                          # 앞뒤 공백 제거 후 반환


    # ---------------------- 배치 예측 함수 ---------------------- #
    def predict_batch(
        self,
        dialogues: List[str],                           # 대화 리스트
        batch_size: int = 32,                           # 배치 크기
        show_progress: bool = True,                     # 진행 표시 여부
        **generation_kwargs                             # 생성 파라미터 (선택적)
    ) -> List[str]:
        """
        배치 대화 요약 예측

        Args:
            dialogues: 대화 리스트
            batch_size: 배치 크기
            show_progress: 진행 표시 여부
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            List[str]: 예측된 요약 리스트
        """
        # -------------- 데이터셋 생성 -------------- #
        dataset = InferenceDataset(                     # 추론용 데이터셋
            dialogues=dialogues,
            tokenizer=self.tokenizer,
            encoder_max_len=self.config.get('tokenizer', {}).get('encoder_max_len', 512) if self.config else 512,
            preprocess=True                             # 전처리 적용
        )

        # -------------- DataLoader 생성 -------------- #
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,                              # 순서 유지
            num_workers=0                               # 추론 시 워커 불필요
        )

        # -------------- 생성 파라미터 병합 -------------- #
        gen_config = {**self.generation_config, **generation_kwargs}  # 기본 + 오버라이드

        # -------------- 배치 추론 -------------- #
        summaries = []                                  # 요약 리스트

        # 진행 표시
        pbar = tqdm(dataloader, desc="Predicting") if show_progress else dataloader

        for batch in pbar:                              # 각 배치 반복
            # 디바이스로 이동
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ['input_ids', 'attention_mask']  # BART는 이 두 개만 사용
            }

            # 추론 실행
            with torch.no_grad():                       # 그래디언트 계산 비활성화
                outputs = self.model.generate(          # 텍스트 생성
                    **inputs,
                    **gen_config
                )

            # 디코딩
            batch_summaries = self.tokenizer.batch_decode(  # 배치 디코딩
                outputs,
                skip_special_tokens=True                # 특수 토큰 제외
            )

            # 앞뒤 공백 제거 후 추가
            summaries.extend([s.strip() for s in batch_summaries])

        return summaries                                # 요약 리스트 반환


    # ---------------------- DataFrame 예측 함수 ---------------------- #
    def predict_dataframe(
        self,
        df: pd.DataFrame,                               # 입력 DataFrame
        batch_size: int = 32,                           # 배치 크기
        show_progress: bool = True,                     # 진행 표시 여부
        **generation_kwargs                             # 생성 파라미터 (선택적)
    ) -> pd.DataFrame:
        """
        DataFrame에 대해 예측 수행

        Args:
            df: 입력 DataFrame (dialogue 컬럼 필요)
            batch_size: 배치 크기
            show_progress: 진행 표시 여부
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            pd.DataFrame: 예측 결과가 추가된 DataFrame
        """
        # -------------- 대화 추출 -------------- #
        dialogues = df['dialogue'].tolist()             # 대화 리스트 변환

        # -------------- 배치 예측 -------------- #
        summaries = self.predict_batch(                 # 배치 예측 실행
            dialogues=dialogues,
            batch_size=batch_size,
            show_progress=show_progress,
            **generation_kwargs
        )

        # -------------- 결과 추가 -------------- #
        df = df.copy()                                  # 원본 보존
        df['summary'] = summaries                       # 예측 결과 추가

        return df                                       # 결과 DataFrame 반환


    # ---------------------- 제출 파일 생성 함수 ---------------------- #
    def create_submission(
        self,
        test_df: pd.DataFrame,                          # 테스트 DataFrame
        output_path: Union[str, Path],                  # 출력 경로
        batch_size: int = 32,                           # 배치 크기
        show_progress: bool = True,                     # 진행 표시 여부
        **generation_kwargs                             # 생성 파라미터 (선택적)
    ) -> pd.DataFrame:
        """
        제출 파일 생성

        Args:
            test_df: 테스트 DataFrame (fname, dialogue 컬럼 필요)
            output_path: 출력 파일 경로
            batch_size: 배치 크기
            show_progress: 진행 표시 여부
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            pd.DataFrame: 제출용 DataFrame (fname, summary)
        """
        msg = "=" * 60
        if self.logger:
            self.logger.write(msg)
            self.logger.write("제출 파일 생성 시작")
            self.logger.write(msg)
        else:
            print(msg)
            print("제출 파일 생성 시작")
            print(msg)

        # -------------- 예측 수행 -------------- #
        msg = f"\n샘플 수: {len(test_df)}"
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

        result_df = self.predict_dataframe(             # DataFrame 예측
            test_df,
            batch_size=batch_size,
            show_progress=show_progress,
            **generation_kwargs
        )

        # -------------- 제출 형식으로 변환 -------------- #
        submission_df = result_df[['fname', 'summary']].copy()  # 필요한 컬럼만 선택

        # -------------- 파일 저장 -------------- #
        output_path = Path(output_path)                 # Path 객체로 변환
        ensure_dir(output_path.parent)                  # 부모 디렉토리 생성

        submission_df.to_csv(output_path, index=False, encoding='utf-8')  # CSV로 저장

        msg = f"\n✅ 제출 파일 저장 완료: {output_path}"
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

        msg = "=" * 60
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

        return submission_df                            # 제출 DataFrame 반환


# ==================== 편의 함수 ==================== #
# ---------------------- Predictor 생성 함수 ---------------------- #
def create_predictor(
    model: PreTrainedModel,                             # 학습된 모델
    tokenizer: PreTrainedTokenizer,                     # 토크나이저
    config: Optional[DictConfig] = None,                # Config (선택적)
    device: Optional[torch.device] = None,              # 디바이스 (선택적)
    logger=None                                         # Logger 인스턴스 (선택적)
) -> Predictor:
    """
    Predictor 생성 편의 함수

    Args:
        model: 학습된 모델
        tokenizer: 토크나이저
        config: 전체 Config
        device: 추론 디바이스
        logger: Logger 인스턴스 (선택적)

    Returns:
        Predictor: 생성된 Predictor 객체
    """
    return Predictor(                                   # Predictor 생성
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        logger=logger
    )
