"""
LLM 파인튜닝을 위한 Dataset

PRD 08: LLM 파인튜닝 전략 구현
- Instruction Format 데이터 생성
- Chat Format 데이터 생성
- Prompt truncation 방지 (encoder_max_len=1024)
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional


class LLMSummarizationDataset(Dataset):
    """LLM 파인튜닝을 위한 대화 요약 Dataset"""

    def __init__(
        self,
        dialogues: List[str],
        summaries: List[str],
        tokenizer,
        encoder_max_len: int = 1024,  # PRD 08: 512 → 1024 (Prompt truncation 방지)
        decoder_max_len: int = 200,   # PRD 08: 100 → 200 (여유 확보)
        format_type: str = "instruction",  # "instruction" or "chat"
        instruction: str = "다음 대화를 간결하게 요약해주세요."
    ):
        """
        Args:
            dialogues: 대화 리스트
            summaries: 요약 리스트
            tokenizer: Tokenizer
            encoder_max_len: 입력 최대 길이 (1024 권장)
            decoder_max_len: 출력 최대 길이 (200 권장)
            format_type: 포맷 타입 ("instruction" or "chat")
            instruction: Instruction 텍스트
        """
        assert len(dialogues) == len(summaries)

        self.dialogues = dialogues
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.format_type = format_type
        self.instruction = instruction

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> dict:
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]

        # Prompt 생성
        if self.format_type == "instruction":
            prompt = self._format_instruction(dialogue, summary)
        else:  # chat
            prompt = self._format_chat(dialogue, summary)

        # 토큰화
        encodings = self.tokenizer(
            prompt,
            max_length=self.encoder_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # Label은 input_ids와 동일 (Causal LM)
        labels = input_ids.clone()

        # Padding 토큰은 -100으로 변경 (loss 계산 시 무시)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _format_instruction(self, dialogue: str, summary: str) -> str:
        """
        Instruction Format 생성

        Format:
            ### Instruction:
            다음 대화를 간결하게 요약해주세요.

            ### Input:
            {dialogue}

            ### Response:
            {summary}
        """
        prompt = f"""### Instruction:
{self.instruction}

### Input:
{dialogue}

### Response:
{summary}"""

        return prompt

    def _format_chat(self, dialogue: str, summary: str) -> str:
        """
        Chat Format 생성 (Llama/Qwen)

        Format:
            <|system|>
            당신은 대화를 요약하는 전문가입니다.
            <|user|>
            다음 대화를 요약해주세요:
            {dialogue}
            <|assistant|>
            {summary}
        """
        # 모델별 chat template 적용
        checkpoint = self.tokenizer.name_or_path.lower()

        if 'llama' in checkpoint:
            prompt = self._format_llama_chat(dialogue, summary)
        elif 'qwen' in checkpoint:
            prompt = self._format_qwen_chat(dialogue, summary)
        else:
            # 기본 format
            prompt = f"""<|system|>
당신은 대화를 요약하는 전문가입니다.
<|user|>
다음 대화를 요약해주세요:
{dialogue}
<|assistant|>
{summary}"""

        return prompt

    def _format_llama_chat(self, dialogue: str, summary: str) -> str:
        """Llama Chat Format"""
        prompt = f"""<|start_header_id|>system<|end_header_id|>

당신은 대화를 요약하는 전문가입니다.<|eot_id|><|start_header_id|>user<|end_header_id|>

다음 대화를 요약해주세요:
{dialogue}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{summary}<|eot_id|>"""

        return prompt

    def _format_qwen_chat(self, dialogue: str, summary: str) -> str:
        """Qwen Chat Format"""
        prompt = f"""<|im_start|>system
당신은 대화를 요약하는 전문가입니다.<|im_end|>
<|im_start|>user
다음 대화를 요약해주세요:
{dialogue}<|im_end|>
<|im_start|>assistant
{summary}<|im_end|>"""

        return prompt


class InstructionAugmentedDataset(Dataset):
    """
    PRD 08: Instruction Tuning
    다양한 instruction 템플릿으로 데이터 증강
    """

    INSTRUCTIONS = [
        "다음 대화를 요약해주세요.",
        "아래 대화의 핵심 내용을 정리해주세요.",
        "주어진 대화를 간단히 요약하세요.",
        "다음 대화에서 중요한 정보를 추출해 요약하세요.",
        "대화 내용을 한 문단으로 요약해주세요."
    ]

    def __init__(
        self,
        dialogues: List[str],
        summaries: List[str],
        tokenizer,
        encoder_max_len: int = 1024,
        decoder_max_len: int = 200,
        format_type: str = "instruction"
    ):
        """
        Args:
            dialogues: 대화 리스트
            summaries: 요약 리스트
            tokenizer: Tokenizer
            encoder_max_len: 입력 최대 길이
            decoder_max_len: 출력 최대 길이
            format_type: 포맷 타입
        """
        # 데이터 증강: 각 샘플에 5가지 instruction 적용
        self.augmented_dialogues = []
        self.augmented_summaries = []
        self.augmented_instructions = []

        for dialogue, summary in zip(dialogues, summaries):
            for instruction in self.INSTRUCTIONS:
                self.augmented_dialogues.append(dialogue)
                self.augmented_summaries.append(summary)
                self.augmented_instructions.append(instruction)

        # Base dataset 생성
        self.datasets = [
            LLMSummarizationDataset(
                [dialogue],
                [summary],
                tokenizer,
                encoder_max_len,
                decoder_max_len,
                format_type,
                instruction
            )
            for dialogue, summary, instruction in zip(
                self.augmented_dialogues,
                self.augmented_summaries,
                self.augmented_instructions
            )
        ]

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int) -> dict:
        return self.datasets[idx][0]


def create_llm_dataset(
    dialogues: List[str],
    summaries: List[str],
    tokenizer,
    encoder_max_len: int = 1024,
    decoder_max_len: int = 200,
    format_type: str = "instruction",
    use_instruction_augmentation: bool = False
) -> Dataset:
    """
    편의 함수: LLM Dataset 생성

    Args:
        dialogues: 대화 리스트
        summaries: 요약 리스트 (추론 시 None 가능)
        tokenizer: Tokenizer
        encoder_max_len: 입력 최대 길이 (1024 권장)
        decoder_max_len: 출력 최대 길이 (200 권장)
        format_type: 포맷 타입 ("instruction" or "chat")
        use_instruction_augmentation: Instruction 증강 사용 여부

    Returns:
        Dataset
    """
    if use_instruction_augmentation and summaries is not None:
        return InstructionAugmentedDataset(
            dialogues,
            summaries,
            tokenizer,
            encoder_max_len,
            decoder_max_len,
            format_type
        )
    else:
        return LLMSummarizationDataset(
            dialogues,
            summaries if summaries is not None else [""] * len(dialogues),
            tokenizer,
            encoder_max_len,
            decoder_max_len,
            format_type
        )
