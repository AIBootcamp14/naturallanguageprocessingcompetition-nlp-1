# ==================== 데이터셋 클래스 모듈 ==================== #
"""
대화 요약 데이터셋 클래스

학습/검증/추론을 위한 PyTorch Dataset 클래스 제공
- DialogueSummarizationDataset: 학습/검증용
- InferenceDataset: 추론용
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from typing import Dict, List, Optional

# ---------------------- 서드파티 라이브러리 ---------------------- #
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------- 프로젝트 모듈 ---------------------- #
from .preprocessor import DialoguePreprocessor


# ==================== DialogueSummarizationDataset 클래스 정의 ==================== #
class DialogueSummarizationDataset(Dataset):
    """학습/검증용 데이터셋 클래스"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(
        self,
        dialogues: List[str],                               # 대화 리스트
        summaries: List[str],                               # 요약 리스트
        tokenizer,                                          # 토크나이저
        encoder_max_len: int = 512,                         # 인코더 최대 길이
        decoder_max_len: int = 100,                         # 디코더 최대 길이
        preprocess: bool = True,                            # 전처리 여부
        model_type: str = 'encoder_decoder'                 # 모델 타입 (PRD 08)
    ):
        """
        Args:
            dialogues: 대화 텍스트 리스트
            summaries: 요약 텍스트 리스트
            tokenizer: HuggingFace 토크나이저
            encoder_max_len: 인코더 최대 길이
            decoder_max_len: 디코더 최대 길이
            preprocess: 전처리 적용 여부
            model_type: 모델 타입 (encoder_decoder 또는 causal_lm)
        """
        # -------------- 전처리 -------------- #
        if preprocess:                                      # 전처리 활성화 시
            preprocessor = DialoguePreprocessor()           # 전처리기 생성
            dialogues, summaries = preprocessor.preprocess_batch(  # 배치 전처리
                dialogues, summaries
            )

        # -------------- 데이터 저장 -------------- #
        self.dialogues = dialogues                          # 대화 리스트 저장
        self.summaries = summaries                          # 요약 리스트 저장
        self.tokenizer = tokenizer                          # 토크나이저 저장
        self.encoder_max_len = encoder_max_len              # 인코더 최대 길이 저장
        self.decoder_max_len = decoder_max_len              # 디코더 최대 길이 저장
        self.model_type = model_type                        # 모델 타입 저장 (PRD 08)


    # ---------------------- 길이 반환 함수 ---------------------- #
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.dialogues)                          # 대화 개수 반환


    # ---------------------- 아이템 반환 함수 ---------------------- #
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        인덱스에 해당하는 데이터 아이템 반환

        Args:
            idx: 데이터 인덱스

        Returns:
            Dict: {
                'input_ids': 입력 토큰,
                'attention_mask': 어텐션 마스크,
                'labels': 출력 토큰 (손실 계산용)
            }
        """
        # -------------- 데이터 추출 -------------- #
        dialogue = self.dialogues[idx]                      # idx번째 대화
        summary = self.summaries[idx]                       # idx번째 요약

        # -------------- 모델 타입별 처리 -------------- #
        if self.model_type == 'encoder_decoder':
            return self._get_encoder_decoder_item(dialogue, summary)
        elif self.model_type == 'causal_lm':
            return self._get_causal_lm_item(dialogue, summary)
        else:
            raise ValueError(f"지원하지 않는 model_type: {self.model_type}")


    # ---------------------- Encoder-Decoder 데이터 생성 ---------------------- #
    def _get_encoder_decoder_item(self, dialogue: str, summary: str) -> Dict[str, torch.Tensor]:
        """Encoder-Decoder 모델용 데이터"""
        # 인코더 입력 토크나이징
        encoder_inputs = self.tokenizer(
            dialogue,                                       # 입력 텍스트
            max_length=self.encoder_max_len,                # 최대 길이
            padding='max_length',                           # 최대 길이까지 패딩
            truncation=True,                                # 최대 길이 초과 시 자르기
            return_tensors='pt'                             # PyTorch 텐서 반환
        )

        # 디코더 출력 토크나이징
        decoder_outputs = self.tokenizer(
            summary,                                        # 출력 텍스트
            max_length=self.decoder_max_len,                # 최대 길이
            padding='max_length',                           # 최대 길이까지 패딩
            truncation=True,                                # 최대 길이 초과 시 자르기
            return_tensors='pt'                             # PyTorch 텐서 반환
        )

        # 텐서 차원 조정 (1, seq_len) → (seq_len)
        input_ids = encoder_inputs['input_ids'].squeeze()   # 인코더 입력 ID
        attention_mask = encoder_inputs['attention_mask'].squeeze()  # 어텐션 마스크
        labels = decoder_outputs['input_ids'].squeeze()     # 디코더 레이블

        # 패딩 토큰을 -100으로 변경 (손실 계산 시 무시)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


    # ---------------------- Causal LM 데이터 생성 ---------------------- #
    def _get_causal_lm_item(self, dialogue: str, summary: str) -> Dict[str, torch.Tensor]:
        """Causal LM 모델용 데이터 (Instruction Tuning 스타일)"""
        from src.models.llm_loader import format_llm_prompt

        # 프롬프트 생성
        prompt = format_llm_prompt(dialogue, self.tokenizer)
        full_text = prompt + summary + self.tokenizer.eos_token

        # 전체 텍스트 토크나이징
        encoding = self.tokenizer(
            full_text,
            max_length=self.encoder_max_len,                # 최대 길이
            padding='max_length',                           # 최대 길이까지 패딩
            truncation=True,                                # 최대 길이 초과 시 자르기
            return_tensors='pt'                             # PyTorch 텐서 반환
        )

        # Labels 생성 (프롬프트 부분은 -100으로 마스킹)
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.encoder_max_len,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]

        # Labels 설정
        labels = encoding['input_ids'].clone()
        labels[:, :prompt_length] = -100                    # 프롬프트 부분 마스킹
        labels[labels == self.tokenizer.pad_token_id] = -100  # 패딩 마스킹

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


# ==================== InferenceDataset 클래스 정의 ==================== #
class InferenceDataset(Dataset):
    """추론용 데이터셋 클래스"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(
        self,
        dialogues: List[str],                               # 대화 리스트
        tokenizer,                                          # 토크나이저
        encoder_max_len: int = 512,                         # 인코더 최대 길이
        preprocess: bool = True,                            # 전처리 여부
        fnames: Optional[List[str]] = None                  # 파일명 리스트 (선택적)
    ):
        """
        Args:
            dialogues: 대화 텍스트 리스트
            tokenizer: HuggingFace 토크나이저
            encoder_max_len: 인코더 최대 길이
            preprocess: 전처리 적용 여부
            fnames: 파일명 리스트 (제출용)
        """
        # -------------- 전처리 -------------- #
        if preprocess:                                      # 전처리 활성화 시
            preprocessor = DialoguePreprocessor()           # 전처리기 생성
            dialogues, _ = preprocessor.preprocess_batch(dialogues)  # 배치 전처리

        # -------------- 데이터 저장 -------------- #
        self.dialogues = dialogues                          # 대화 리스트 저장
        self.tokenizer = tokenizer                          # 토크나이저 저장
        self.encoder_max_len = encoder_max_len              # 인코더 최대 길이 저장
        self.fnames = fnames                                # 파일명 리스트 저장

        # -------------- 베이스라인 방식: 전체 데이터 한 번에 토크나이징 -------------- #
        self.tokenized_data = self.tokenizer(               # 전체 리스트 토크나이징
            dialogues,
            padding=True,                                   # 동적 패딩 (전체에서 최대값)
            max_length=encoder_max_len,
            truncation=True,
            return_tensors='pt'
        )


    # ---------------------- 길이 반환 함수 ---------------------- #
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.dialogues)                          # 대화 개수 반환


    # ---------------------- 아이템 반환 함수 ---------------------- #
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        인덱스에 해당하는 데이터 아이템 반환

        Args:
            idx: 데이터 인덱스

        Returns:
            Dict: {
                'input_ids': 인코더 입력 토큰,
                'attention_mask': 인코더 어텐션 마스크,
                'fname': 파일명 (있는 경우)
            }
        """
        # -------------- 미리 토크나이징된 데이터에서 추출 -------------- #
        result = {
            'input_ids': self.tokenized_data['input_ids'][idx],
            'attention_mask': self.tokenized_data['attention_mask'][idx]
        }

        # -------------- 파일명 추가 (선택적) -------------- #
        if self.fnames is not None:                         # 파일명 리스트가 있는 경우
            result['fname'] = self.fnames[idx]              # 파일명 추가

        return result                                       # 결과 반환


# ==================== 편의 함수 ==================== #
# ---------------------- DataFrame에서 데이터셋 생성 -------------- #
def create_dataset_from_dataframe(
    df: pd.DataFrame,                                       # DataFrame
    tokenizer,                                              # 토크나이저
    encoder_max_len: int = 512,                             # 인코더 최대 길이
    decoder_max_len: int = 100,                             # 디코더 최대 길이
    is_train: bool = True,                                  # 학습 모드 여부
    preprocess: bool = True                                 # 전처리 여부
) -> Dataset:
    """
    DataFrame에서 데이터셋 생성 편의 함수

    Args:
        df: 데이터 DataFrame
        tokenizer: HuggingFace 토크나이저
        encoder_max_len: 인코더 최대 길이
        decoder_max_len: 디코더 최대 길이
        is_train: 학습 모드 (True: 학습/검증, False: 추론)
        preprocess: 전처리 적용 여부

    Returns:
        Dataset: 생성된 데이터셋
    """
    # -------------- 대화 추출 -------------- #
    dialogues = df['dialogue'].tolist()                     # 대화 리스트 변환

    # -------------- 학습/검증 모드 -------------- #
    if is_train:
        summaries = df['summary'].tolist()                  # 요약 리스트 변환
        return DialogueSummarizationDataset(                # 학습/검증 데이터셋 생성
            dialogues=dialogues,
            summaries=summaries,
            tokenizer=tokenizer,
            encoder_max_len=encoder_max_len,
            decoder_max_len=decoder_max_len,
            preprocess=preprocess
        )

    # -------------- 추론 모드 -------------- #
    else:
        fnames = df['fname'].tolist() if 'fname' in df.columns else None  # 파일명 추출
        return InferenceDataset(                            # 추론 데이터셋 생성
            dialogues=dialogues,
            tokenizer=tokenizer,
            encoder_max_len=encoder_max_len,
            preprocess=preprocess,
            fnames=fnames
        )
