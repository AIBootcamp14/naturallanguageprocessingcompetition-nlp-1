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
import re
import warnings

# Transformers 경고 메시지 필터링
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*num_labels.*id2label.*")

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


# ==================== 후처리 함수 ==================== #
def postprocess_summary(text: str) -> str:
    """
    요약문 후처리: 불완전한 문장을 정제하여 완전한 문장으로 변환

    주요 처리 과정:
    0. 불필요한 접두사 제거 ("대화 요약:", "Summary:" 등)
    1. 반복된 점들 제거 ("... . . ." 패턴)
    2. 불완전한 플레이스홀더 제거 (모든 패턴)
    3. 불완전한 마지막 문장 제거 (끊긴 문장 삭제)
    4. 불완전한 종결어 제거
    5. 문장 종결 보장 (마침표가 없으면 추가)

    Args:
        text: 모델이 생성한 원본 요약문

    Returns:
        정제된 요약문

    Examples:
        >>> postprocess_summary("대화 요약: Person1과 Person2는 #Mr.")
        'Person1과 Person2는.'

        >>> postprocess_summary("Summary: 회의 시간을 변경하자고 제")
        '회의 시간을 변경하자고.'

        >>> postprocess_summary("약속했... . . .")
        '약속했.'
    """
    text = text.strip()                                 # 앞뒤 공백 제거
    if not text:                                        # 빈 문자열 처리
        return text

    # -------------- 0. 불필요한 접두사 제거 -------------- #
    # 모든 형태의 접두사 제거
    patterns = [
        r'^대화\s*(내용)?\s*요약\s*[:：]\s*\n*',
        r'^Summary\s*[:：]\s*\n*',
        r'^요약\s*[:：]\s*\n*',
        r'^대화\s*[:：]\s*\n*',
        r'^대화\s*상대\s+[A-Z가-힣]*\s*(가|는|이|을|를)\s*',
        r'^대화에서는?\s+',
        r'^대화\s*참여자들은?\s+',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = text.lstrip('\n\t ')

    # -------------- 1. 반복된 점들 제거 -------------- #
    # "... . . ." 또는 연속된 점 패턴 제거
    text = re.sub(r'(\.{3,}|\s*\.\s*){2,}', '', text)
    text = text.strip()

    # -------------- 2. 불완전한 플레이스홀더 제거 (강화) -------------- #
    # 마지막에 불완전한 플레이스홀더 패턴 모두 제거
    # #Mr. #Mrs. #Fr. #Korea. #Person1 #Peron1 #PAerson1 등
    # 패턴 1: # 뒤에 대문자로 시작하고 . 으로 끝나는 경우
    text = re.sub(r'\s*#[A-Z][a-z]*\.$', '', text)
    # 패턴 2: # 뒤에 짧은 문자열 (10자 이하)
    text = re.sub(r'\s*#[A-Za-z가-힣]{0,15}$', '', text)
    text = text.strip()

    # -------------- 3. 마지막 불완전한 문장 제거 (강화) -------------- #
    # 마지막 완전한 문장부호까지만 남기기
    last_punct_idx = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] in '.!?。？！':
            last_punct_idx = i
            break

    if last_punct_idx > 0:
        # 마지막 문장부호 이후에 뭔가 있으면 (불완전한 문장)
        after_punct = text[last_punct_idx + 1:].strip()
        if after_punct:
            # 불완전한 조각이 있으면 제거
            # 30자 이하는 불완전한 것으로 간주하고 제거
            if len(after_punct) <= 30:
                text = text[:last_punct_idx + 1]
            else:
                # 30자 이상이지만 문장부호로 끝나지 않으면 제거
                if after_punct[-1] not in '.!?。？！':
                    text = text[:last_punct_idx + 1]

    text = text.strip()

    # -------------- 4. 불완전한 종결어 제거 (강화) -------------- #
    # "이후.", "그 후.", "그러고 나서." 같은 불완전한 종결 제거
    incomplete_endings = [
        '이후.', '그 후.', '그러고 나서.', '결국.', '최종적으로.',
        '이제.', '그리고.', '또한.', '하지만.', '그러나.',
        '그들은 지금 가자.', '그는 다시 진행한다.', '대화는 마무리됩니다.'
    ]

    for ending in incomplete_endings:
        if text.endswith(ending):
            text = text[:-len(ending)].strip()
            # 제거 후 마지막 문장부호 찾기
            last_punct_idx = -1
            for i in range(len(text) - 1, -1, -1):
                if text[i] in '.!?。？！':
                    last_punct_idx = i
                    break
            if last_punct_idx > 0:
                text = text[:last_punct_idx + 1]
            break

    text = text.strip()

    # -------------- 5. 짧은 마지막 조사/단어 제거 -------------- #
    # 문장부호 없이 끝나는 경우, 1-3자 조사 제거
    if text and text[-1] not in '.!?。？！':
        words = text.split()                            # 단어 분리
        if len(words) > 1:                              # 여러 단어가 있는 경우
            last_word = words[-1]                       # 마지막 단어
            # 마지막 단어가 짧은 조사인 경우 제거
            common_particles = [
                '은', '는', '을', '를', '이', '가',
                '의', '에', '로', '와', '과', '도',
                '만', '에서', '으로', '프', '또한', '그', '이에',
                '완', '시', '오', '각', '나서'
            ]
            if len(last_word) <= 4 and last_word in common_particles:
                text = ' '.join(words[:-1])             # 마지막 단어 제외
                text = text.strip()

    # -------------- 6. 문장 종결 보장 -------------- #
    # 마침표가 없으면 추가
    if text and text[-1] not in '.!?。？！':
        text += '.'

    return text


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

        # 후처리 적용: 불완전한 문장 정제 + 문장 종결 보장
        return postprocess_summary(summary)


    # ---------------------- 배치 예측 함수 ---------------------- #
    def predict_batch(
        self,
        dialogues: List[str],                           # 대화 리스트
        batch_size: int = 32,                           # 배치 크기
        show_progress: bool = True,                     # 진행 표시 여부
        use_pretrained_correction: bool = False,        # ✅ HF 보정 사용 여부
        correction_models: Optional[List[str]] = None,  # ✅ HF 모델 리스트
        correction_strategy: str = "quality_based",     # ✅ 보정 전략
        correction_threshold: float = 0.3,              # ✅ 품질 임계값
        checkpoint_dir: Optional[str] = None,           # ✅ 체크포인트 디렉토리
        **generation_kwargs                             # 생성 파라미터 (선택적)
    ) -> List[str]:
        """
        배치 대화 요약 예측

        Args:
            dialogues: 대화 리스트
            batch_size: 배치 크기
            show_progress: 진행 표시 여부
            use_pretrained_correction: HuggingFace 사전학습 모델 보정 사용 여부
            correction_models: 보정에 사용할 HF 모델 리스트 (예: ["gogamza/kobart-base-v2"])
            correction_strategy: 보정 전략 (quality_based, threshold, voting, weighted)
            checkpoint_dir: 체크포인트 디렉토리 (선택)
            correction_threshold: 품질 임계값 (0.0~1.0)
            **generation_kwargs: 생성 파라미터 오버라이드

        Returns:
            List[str]: 예측된 요약 리스트
        """
        # -------------- 데이터셋 생성 -------------- #
        if self.logger:
            self.logger.write("추론용 데이터셋 생성 중...")

        dataset = InferenceDataset(                     # 추론용 데이터셋
            dialogues=dialogues,
            tokenizer=self.tokenizer,
            encoder_max_len=self.config.get('tokenizer', {}).get('encoder_max_len', 512) if self.config else 512,
            preprocess=True                             # 전처리 적용
        )

        if self.logger:
            self.logger.write(f"✅ 데이터셋 생성 완료 (샘플 수: {len(dataset)})")

        # -------------- DataLoader 생성 -------------- #
        if self.logger:
            self.logger.write(f"DataLoader 생성 중 (batch_size={batch_size})...")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,                              # 순서 유지
            num_workers=0                               # 추론 시 워커 불필요
        )

        if self.logger:
            self.logger.write(f"✅ DataLoader 생성 완료 (총 배치 수: {len(dataloader)})")

        # -------------- 생성 파라미터 병합 -------------- #
        gen_config = {**self.generation_config, **generation_kwargs}  # 기본 + 오버라이드

        if self.logger:
            self.logger.write("생성 파라미터 설정 완료")
            # 주요 파라미터 로깅
            key_params = ['max_new_tokens', 'num_beams', 'repetition_penalty', 'no_repeat_ngram_size', 'min_new_tokens']
            param_str = ", ".join([f"{k}={gen_config.get(k, 'N/A')}" for k in key_params if k in gen_config])
            self.logger.write(f"  - {param_str}")

        # -------------- 배치 추론 -------------- #
        summaries = []                                  # 요약 리스트

        # 진행 표시
        pbar = tqdm(dataloader, desc="Predicting") if show_progress else dataloader

        if self.logger:
            self.logger.write(f"배치 추론 시작 (총 {len(dataloader)}개 배치)")

        for batch_idx, batch in enumerate(pbar, 1):     # 각 배치 반복
            if self.logger:
                self.logger.write(f"[배치 {batch_idx}/{len(dataloader)}] 추론 시작...")

            # 디바이스로 이동
            if self.logger:
                self.logger.write(f"[배치 {batch_idx}/{len(dataloader)}] 입력 데이터를 {self.device}로 전송 중...")

            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ['input_ids', 'attention_mask']  # BART는 이 두 개만 사용
            }

            if self.logger:
                self.logger.write(f"[배치 {batch_idx}/{len(dataloader)}] 모델 생성(generate) 실행 중...")

            # 추론 실행
            with torch.no_grad():                       # 그래디언트 계산 비활성화
                outputs = self.model.generate(          # 텍스트 생성
                    **inputs,
                    **gen_config
                )

            if self.logger:
                self.logger.write(f"[배치 {batch_idx}/{len(dataloader)}] 생성 완료, 디코딩 중...")

            # 디코딩
            batch_summaries = self.tokenizer.batch_decode(  # 배치 디코딩
                outputs,
                skip_special_tokens=True                # 특수 토큰 제외
            )

            if self.logger:
                self.logger.write(f"[배치 {batch_idx}/{len(dataloader)}] 후처리 적용 중...")

            # 후처리 적용: 불완전한 문장 정제 + 문장 종결 보장
            summaries.extend([postprocess_summary(s) for s in batch_summaries])

            if self.logger:
                self.logger.write(f"✅ [배치 {batch_idx}/{len(dataloader)}] 완료 (누적 요약 수: {len(summaries)})")

        if self.logger:
            self.logger.write(f"🎉 전체 배치 추론 완료 (총 요약 수: {len(summaries)})")

        # ✅ ==================== HuggingFace 보정 로직 추가 ==================== #
        if use_pretrained_correction and correction_models:
            if self.logger:
                self.logger.write("\n" + "=" * 60)
                self.logger.write("🔧 HuggingFace 사전학습 모델 보정 시작")

            try:
                # PretrainedCorrector 초기화
                from src.correction import create_pretrained_corrector

                # 체크포인트 디렉토리 설정 (HF correction 서브디렉토리)
                correction_checkpoint_dir = None
                if checkpoint_dir:
                    from pathlib import Path
                    correction_checkpoint_dir = str(Path(checkpoint_dir) / "hf_correction")

                corrector = create_pretrained_corrector(
                    model_names=correction_models,
                    correction_strategy=correction_strategy,
                    quality_threshold=correction_threshold,
                    device=self.device,
                    logger=self.logger,
                    checkpoint_dir=correction_checkpoint_dir
                )

                # 보정 수행
                summaries = corrector.correct_batch(
                    dialogues=dialogues,
                    candidate_summaries=summaries,
                    batch_size=batch_size,
                    **generation_kwargs
                )

                if self.logger:
                    self.logger.write("✅ HuggingFace 사전학습 모델 보정 완료")
                    self.logger.write("=" * 60 + "\n")

            except Exception as e:
                if self.logger:
                    self.logger.write(f"⚠️  HuggingFace 보정 실패: {str(e)}")
                    self.logger.write("   원본 요약 사용")
                # 보정 실패 시 원본 요약 그대로 사용
        # ==================== HuggingFace 보정 로직 끝 ==================== #

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
