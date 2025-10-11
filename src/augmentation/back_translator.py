"""
역번역 기반 데이터 증강 모듈

PRD 04: 성능 개선 전략 - 데이터 증강
역번역 (Back-translation)을 통한 대화 데이터 증강 기능 제공
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import logging
from typing import List, Optional, Union
# logging : 로깅 시스템
# typing  : 타입 힌팅

# ------------------------- 서드파티 라이브러리 ------------------------- #
import torch
from transformers import MarianMTModel, MarianTokenizer
# torch        : PyTorch 메인 모듈
# transformers : Hugging Face Transformers (MarianMT 모델)

# 로깅 설정은 __init__에서 logging.getLogger(__name__) 사용


# ==================== 역번역 클래스 정의 ==================== #
# ---------------------- BackTranslator 클래스 ---------------------- #
class BackTranslator:
    """
    MarianMT 기반 역번역 증강 시스템

    한국어 → 영어 → 한국어 변환을 통해 의미를 유지하면서
    표현을 다양화한 텍스트 생성

    Args:
        device: 추론 디바이스 ('cuda' 또는 'cpu')
        ko_to_en_model: 한→영 번역 모델명 (기본값: Helsinki-NLP/opus-mt-ko-en)
        en_to_ko_model: 영→한 번역 모델명 (기본값: Helsinki-NLP/opus-mt-en-ko)
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
    """

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ko_to_en_model: str = "Helsinki-NLP/opus-mt-ko-en",
        en_to_ko_model: str = "Helsinki-NLP/opus-mt-en-ko",
        batch_size: int = 8,
        max_length: int = 512
    ):
        """BackTranslator 초기화"""

        # -------------- 기본 설정 초기화 -------------- #
        self.device = device                        # 추론 디바이스 설정
        self.batch_size = batch_size                # 배치 크기
        self.max_length = max_length                # 최대 시퀀스 길이
        self.logger = logging.getLogger(__name__)   # 로거 초기화

        # -------------- 한→영 번역 모델 로드 -------------- #
        self.logger.info(f"한→영 번역 모델 로드 중: {ko_to_en_model}")
        try:
            self.ko_to_en_tokenizer = MarianTokenizer.from_pretrained(ko_to_en_model)
            self.ko_to_en_model = MarianMTModel.from_pretrained(ko_to_en_model).to(device)
            self.ko_to_en_model.eval()              # 평가 모드로 설정
            self.logger.info("한→영 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"한→영 모델 로드 실패: {e}")
            raise

        # -------------- 영→한 번역 모델 로드 -------------- #
        self.logger.info(f"영→한 번역 모델 로드 중: {en_to_ko_model}")
        try:
            self.en_to_ko_tokenizer = MarianTokenizer.from_pretrained(en_to_ko_model)
            self.en_to_ko_model = MarianMTModel.from_pretrained(en_to_ko_model).to(device)
            self.en_to_ko_model.eval()              # 평가 모드로 설정
            self.logger.info("영→한 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"영→한 모델 로드 실패: {e}")
            raise

        self.logger.info(f"BackTranslator 초기화 완료 (device={device})")

    # ---------------------- 단일 텍스트 역번역 ---------------------- #
    def back_translate(
        self,
        text: str,
        num_beams: int = 5,
        temperature: float = 1.0
    ) -> str:
        """
        단일 텍스트 역번역

        Args:
            text: 원본 한국어 텍스트
            num_beams: 빔 서치 크기 (높을수록 품질 향상, 속도 감소)
            temperature: 샘플링 온도 (높을수록 다양성 증가)

        Returns:
            역번역된 한국어 텍스트
        """

        # -------------- 입력 검증 -------------- #
        # 빈 문자열 처리
        if not text or not text.strip():
            self.logger.warning("빈 텍스트 입력됨, 원본 반환")
            return text

        # -------------- 1단계: 한국어 → 영어 번역 -------------- #
        try:
            # 토크나이징
            ko_inputs = self.ko_to_en_tokenizer(
                text,
                return_tensors="pt",                # PyTorch 텐서 반환
                padding=True,                       # 패딩 적용
                truncation=True,                    # 길이 초과 시 자르기
                max_length=self.max_length          # 최대 길이 제한
            ).to(self.device)

            # 추론 실행 (그래디언트 계산 비활성화)
            with torch.no_grad():
                en_outputs = self.ko_to_en_model.generate(
                    **ko_inputs,
                    num_beams=num_beams,            # 빔 서치 크기
                    temperature=temperature,         # 샘플링 온도
                    max_length=self.max_length,     # 최대 생성 길이
                    early_stopping=True             # 조기 종료 활성화
                )

            # 디코딩
            en_text = self.ko_to_en_tokenizer.decode(
                en_outputs[0],
                skip_special_tokens=True            # 특수 토큰 제거
            )

            self.logger.debug(f"한→영 번역: {text[:50]} → {en_text[:50]}")

        except Exception as e:
            self.logger.error(f"한→영 번역 실패: {e}")
            return text                             # 실패 시 원본 반환

        # -------------- 2단계: 영어 → 한국어 번역 -------------- #
        try:
            # 토크나이징
            en_inputs = self.en_to_ko_tokenizer(
                en_text,
                return_tensors="pt",                # PyTorch 텐서 반환
                padding=True,                       # 패딩 적용
                truncation=True,                    # 길이 초과 시 자르기
                max_length=self.max_length          # 최대 길이 제한
            ).to(self.device)

            # 추론 실행 (그래디언트 계산 비활성화)
            with torch.no_grad():
                ko_outputs = self.en_to_ko_model.generate(
                    **en_inputs,
                    num_beams=num_beams,            # 빔 서치 크기
                    temperature=temperature,         # 샘플링 온도
                    max_length=self.max_length,     # 최대 생성 길이
                    early_stopping=True             # 조기 종료 활성화
                )

            # 디코딩
            back_translated = self.en_to_ko_tokenizer.decode(
                ko_outputs[0],
                skip_special_tokens=True            # 특수 토큰 제거
            )

            self.logger.debug(f"영→한 번역: {en_text[:50]} → {back_translated[:50]}")

            return back_translated

        except Exception as e:
            self.logger.error(f"영→한 번역 실패: {e}")
            return text                             # 실패 시 원본 반환

    # ---------------------- 배치 텍스트 역번역 ---------------------- #
    def back_translate_batch(
        self,
        texts: List[str],
        num_beams: int = 5,
        temperature: float = 1.0
    ) -> List[str]:
        """
        여러 텍스트 배치 역번역

        Args:
            texts: 원본 한국어 텍스트 리스트
            num_beams: 빔 서치 크기
            temperature: 샘플링 온도

        Returns:
            역번역된 한국어 텍스트 리스트
        """

        results = []                                # 결과 리스트 초기화

        # -------------- 배치 단위로 처리 -------------- #
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]   # 배치 추출

            # 배치 내 각 텍스트 처리
            for text in batch:
                result = self.back_translate(
                    text,
                    num_beams=num_beams,
                    temperature=temperature
                )
                results.append(result)              # 결과 추가

            # 진행 상황 로깅
            if (i + self.batch_size) % 100 == 0:
                self.logger.info(f"역번역 진행: {min(i + self.batch_size, len(texts))}/{len(texts)}")

        return results

    # ---------------------- 대화 데이터 증강 ---------------------- #
    def augment_dialogue(
        self,
        dialogue: Union[str, List[str]],
        num_augmentations: int = 1,
        num_beams: int = 5,
        temperature: float = 1.0
    ) -> List[Union[str, List[str]]]:
        """
        대화 데이터 증강 (단일 텍스트 또는 발화 리스트)

        Args:
            dialogue: 원본 대화 (단일 문자열 또는 발화 리스트)
            num_augmentations: 증강할 샘플 수
            num_beams: 빔 서치 크기
            temperature: 샘플링 온도

        Returns:
            증강된 대화 리스트
        """

        augmented_samples = []                      # 증강 샘플 리스트 초기화

        # -------------- 입력 타입에 따른 처리 분기 -------------- #
        # 단일 문자열인 경우
        if isinstance(dialogue, str):
            for i in range(num_augmentations):
                # 다양성을 위해 온도 약간 증가
                temp = temperature + (i * 0.1)      # 온도 점진적 증가
                augmented = self.back_translate(
                    dialogue,
                    num_beams=num_beams,
                    temperature=min(temp, 2.0)      # 최대 온도 2.0으로 제한
                )
                augmented_samples.append(augmented)

        # 발화 리스트인 경우
        elif isinstance(dialogue, list):
            for i in range(num_augmentations):
                # 다양성을 위해 온도 약간 증가
                temp = temperature + (i * 0.1)      # 온도 점진적 증가

                # 각 발화 역번역
                augmented_turns = []                # 증강된 발화 리스트
                for turn in dialogue:
                    augmented_turn = self.back_translate(
                        turn,
                        num_beams=num_beams,
                        temperature=min(temp, 2.0)  # 최대 온도 2.0으로 제한
                    )
                    augmented_turns.append(augmented_turn)

                augmented_samples.append(augmented_turns)

        # 지원하지 않는 타입
        else:
            self.logger.error(f"지원하지 않는 dialogue 타입: {type(dialogue)}")
            raise TypeError("dialogue는 str 또는 List[str] 타입이어야 함")

        return augmented_samples

    # ---------------------- GPU 메모리 정리 ---------------------- #
    def cleanup(self):
        """GPU 메모리 정리 및 모델 해제"""

        # -------------- CUDA 메모리 정리 -------------- #
        if self.device == "cuda":
            # 모델을 CPU로 이동
            self.ko_to_en_model.cpu()               # 한→영 모델 CPU 이동
            self.en_to_ko_model.cpu()               # 영→한 모델 CPU 이동

            # CUDA 캐시 비우기
            torch.cuda.empty_cache()                # GPU 메모리 캐시 정리

            self.logger.info("CUDA 메모리 정리 완료")

        self.logger.info("BackTranslator 정리 완료")


# ==================== 헬퍼 함수들 ==================== #
# ---------------------- BackTranslator 생성 함수 ---------------------- #
def create_back_translator(
    device: Optional[str] = None,
    **kwargs
) -> BackTranslator:
    """
    BackTranslator 인스턴스 생성 헬퍼

    Args:
        device: 추론 디바이스 (None이면 자동 감지)
        **kwargs: BackTranslator 생성자 인자

    Returns:
        BackTranslator 인스턴스
    """

    # -------------- 디바이스 자동 감지 -------------- #
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------- BackTranslator 생성 -------------- #
    translator = BackTranslator(device=device, **kwargs)

    return translator


# ---------------------- 메인 실행부 ---------------------- #
if __name__ == "__main__":
    # 간단한 테스트 코드

    # BackTranslator 생성
    translator = create_back_translator()

    # 테스트 텍스트
    test_text = "안녕하세요. 오늘 날씨가 참 좋네요."

    # 역번역 실행
    result = translator.back_translate(test_text)

    print(f"원본: {test_text}")
    print(f"역번역: {result}")

    # 정리
    translator.cleanup()
