"""
동의어 기반 데이터 증강 모듈

PRD 04: 성능 개선 전략 - 데이터 증강
동의어 치환 (Synonym Replacement)을 통한 대화 데이터 증강 기능 제공
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import logging
import random
from typing import List, Dict, Optional, Union
# logging : 로깅 시스템
# random  : 무작위 샘플링
# typing  : 타입 힌팅

# ------------------------- 서드파티 라이브러리 ------------------------- #
# 현재 한국어 동의어 사전 API는 외부 의존성 없이 내장 사전 사용


# ==================== 동의어 사전 정의 ==================== #
# ---------------------- 한국어 동의어 매핑 ---------------------- #
# 대화 요약 도메인에서 자주 사용되는 단어들의 동의어 사전
KOREAN_SYNONYMS = {
    # 동사
    "말하다": ["얘기하다", "이야기하다", "언급하다"],
    "생각하다": ["여기다", "판단하다", "느끼다"],
    "보다": ["살펴보다", "확인하다", "체크하다"],
    "하다": ["수행하다", "진행하다", "실시하다"],
    "가다": ["이동하다", "향하다", "출발하다"],
    "오다": ["도착하다", "찾아오다"],
    "듣다": ["청취하다", "경청하다"],
    "쓰다": ["작성하다", "기록하다"],
    "알다": ["인지하다", "파악하다", "이해하다"],
    "모르다": ["파악 못하다", "알지 못하다"],
    
    # 형용사
    "좋다": ["훌륭하다", "괜찮다", "양호하다", "만족스럽다"],
    "나쁘다": ["안 좋다", "불량하다", "불만스럽다"],
    "크다": ["거대하다", "대형이다"],
    "작다": ["소형이다", "미세하다"],
    "빠르다": ["신속하다", "즉시", "재빠르다"],
    "느리다": ["더디다", "천천히"],
    "많다": ["다수이다", "풍부하다"],
    "적다": ["소수이다", "부족하다"],
    "새롭다": ["신규이다", "최신이다"],
    "오래되다": ["낡다", "구형이다"],
    
    # 명사
    "사람": ["인물", "인원"],
    "회사": ["기업", "업체", "법인"],
    "제품": ["상품", "물건", "아이템"],
    "문제": ["이슈", "과제"],
    "방법": ["수단", "방식"],
    "시간": ["기간", "시각"],
    "장소": ["위치", "곳"],
    "일": ["업무", "작업", "과제"],
    "돈": ["금액", "비용", "자금"],
    "정보": ["데이터", "자료"],
    
    # 부사
    "매우": ["아주", "무척", "굉장히", "상당히"],
    "조금": ["약간", "다소", "살짝"],
    "빨리": ["신속히", "즉시", "바로"],
    "천천히": ["서서히", "조금씩"],
    "항상": ["언제나", "늘", "계속"],
    "가끔": ["때때로", "종종", "이따금"],
    
    # 접속사
    "그래서": ["따라서", "그러므로", "그러니까"],
    "하지만": ["그러나", "그런데", "그렇지만"],
    "그리고": ["또한", "더불어", "아울러"],
    "또는": ["혹은", "내지"],
    
    # 감탄사
    "네": ["예", "알겠습니다", "그렇습니다"],
    "아니요": ["아닙니다", "그렇지 않습니다"],
    "감사합니다": ["고맙습니다", "감사드립니다"],
    "죄송합니다": ["미안합니다", "죄송해요"],
    "안녕하세요": ["안녕하십니까", "반갑습니다"],
}


# ==================== Paraphraser 클래스 정의 ==================== #
# ---------------------- Paraphraser 클래스 ---------------------- #
class Paraphraser:
    """
    동의어 치환 기반 증강 시스템

    한국어 텍스트에서 특정 단어를 동의어로 치환하여
    의미를 유지하면서 표현을 다양화

    Args:
        synonym_dict: 동의어 사전 (None이면 기본 사전 사용)
        replace_ratio: 치환 비율 (0.0 ~ 1.0, 기본값 0.3)
        max_replacements: 최대 치환 단어 수 (None이면 무제한)
        seed: 랜덤 시드 (재현성 보장)
    """

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(
        self,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        replace_ratio: float = 0.3,
        max_replacements: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """Paraphraser 초기화"""

        # -------------- 기본 설정 초기화 -------------- #
        self.synonym_dict = synonym_dict or KOREAN_SYNONYMS  # 동의어 사전
        self.replace_ratio = replace_ratio                   # 치환 비율
        self.max_replacements = max_replacements             # 최대 치환 수
        self.logger = logging.getLogger(__name__)            # 로거 초기화

        # -------------- 랜덤 시드 설정 -------------- #
        if seed is not None:
            random.seed(seed)                                # 시드 고정

        # -------------- 동의어 사전 통계 -------------- #
        total_words = len(self.synonym_dict)                 # 총 단어 수
        total_synonyms = sum(len(syns) for syns in self.synonym_dict.values())

        self.logger.info(f"Paraphraser 초기화 완료")
        self.logger.info(f"  - 동의어 사전 크기: {total_words} 단어")
        self.logger.info(f"  - 총 동의어 수: {total_synonyms} 개")
        self.logger.info(f"  - 치환 비율: {replace_ratio:.1%}")
        if max_replacements:
            self.logger.info(f"  - 최대 치환 수: {max_replacements}")

    # ---------------------- 단일 텍스트 패러프레이징 ---------------------- #
    def paraphrase(
        self,
        text: str,
        replace_ratio: Optional[float] = None
    ) -> str:
        """
        단일 텍스트 패러프레이징 (동의어 치환)

        Args:
            text: 원본 한국어 텍스트
            replace_ratio: 치환 비율 (None이면 초기화 시 설정값 사용)

        Returns:
            패러프레이징된 한국어 텍스트
        """

        # -------------- 입력 검증 -------------- #
        # 빈 문자열 처리
        if not text or not text.strip():
            self.logger.warning("빈 텍스트 입력됨, 원본 반환")
            return text

        # -------------- 치환 비율 설정 -------------- #
        ratio = replace_ratio if replace_ratio is not None else self.replace_ratio

        # -------------- 텍스트를 단어 단위로 분리 -------------- #
        # 간단한 공백 기반 토큰화 (실제 형태소 분석기 사용 가능)
        words = text.split()                                 # 공백으로 단어 분리

        # 치환 가능한 단어 찾기
        replaceable_indices = []                             # 치환 가능 인덱스 리스트
        for i, word in enumerate(words):
            # 동의어 사전에 있는 단어만 치환 대상
            if word in self.synonym_dict:
                replaceable_indices.append(i)                # 인덱스 추가

        # -------------- 치환할 단어 선택 -------------- #
        if not replaceable_indices:
            self.logger.debug("치환 가능한 단어 없음")
            return text                                      # 치환 불가 시 원본 반환

        # 치환할 단어 개수 계산
        num_to_replace = max(1, int(len(replaceable_indices) * ratio))

        # 최대 치환 수 제한 적용
        if self.max_replacements:
            num_to_replace = min(num_to_replace, self.max_replacements)

        # 무작위로 치환할 인덱스 선택
        indices_to_replace = random.sample(
            replaceable_indices,
            min(num_to_replace, len(replaceable_indices))
        )

        # -------------- 동의어로 치환 -------------- #
        replaced_count = 0                                   # 치환 횟수 카운터
        for idx in indices_to_replace:
            original_word = words[idx]                       # 원본 단어
            synonyms = self.synonym_dict[original_word]      # 동의어 리스트

            # 무작위로 동의어 선택
            synonym = random.choice(synonyms)                # 동의어 선택
            words[idx] = synonym                             # 단어 치환
            replaced_count += 1

            self.logger.debug(f"치환: '{original_word}' → '{synonym}'")

        # -------------- 치환된 텍스트 생성 -------------- #
        paraphrased_text = " ".join(words)                   # 단어 재결합

        self.logger.debug(f"패러프레이징 완료 ({replaced_count}개 치환)")

        return paraphrased_text

    # ---------------------- 배치 텍스트 패러프레이징 ---------------------- #
    def paraphrase_batch(
        self,
        texts: List[str],
        replace_ratio: Optional[float] = None
    ) -> List[str]:
        """
        여러 텍스트 배치 패러프레이징

        Args:
            texts: 원본 한국어 텍스트 리스트
            replace_ratio: 치환 비율

        Returns:
            패러프레이징된 한국어 텍스트 리스트
        """

        results = []                                         # 결과 리스트 초기화

        # -------------- 각 텍스트 패러프레이징 -------------- #
        for i, text in enumerate(texts):
            result = self.paraphrase(text, replace_ratio)    # 패러프레이징 실행
            results.append(result)                           # 결과 추가

            # 진행 상황 로깅
            if (i + 1) % 100 == 0:
                self.logger.info(f"패러프레이징 진행: {i + 1}/{len(texts)}")

        return results

    # ---------------------- 대화 데이터 증강 ---------------------- #
    def augment_dialogue(
        self,
        dialogue: Union[str, List[str]],
        num_augmentations: int = 1,
        replace_ratio: Optional[float] = None
    ) -> List[Union[str, List[str]]]:
        """
        대화 데이터 증강 (단일 텍스트 또는 발화 리스트)

        Args:
            dialogue: 원본 대화 (단일 문자열 또는 발화 리스트)
            num_augmentations: 증강할 샘플 수
            replace_ratio: 치환 비율

        Returns:
            증강된 대화 리스트
        """

        augmented_samples = []                               # 증강 샘플 리스트 초기화

        # -------------- 입력 타입에 따른 처리 분기 -------------- #
        # 단일 문자열인 경우
        if isinstance(dialogue, str):
            for i in range(num_augmentations):
                # 다양성을 위해 치환 비율 약간 변화
                ratio = replace_ratio or self.replace_ratio
                varied_ratio = ratio * (1.0 + random.uniform(-0.2, 0.2))  # ±20% 변화
                varied_ratio = max(0.1, min(1.0, varied_ratio))  # 0.1~1.0 범위 제한

                augmented = self.paraphrase(
                    dialogue,
                    replace_ratio=varied_ratio
                )
                augmented_samples.append(augmented)

        # 발화 리스트인 경우
        elif isinstance(dialogue, list):
            for i in range(num_augmentations):
                # 다양성을 위해 치환 비율 약간 변화
                ratio = replace_ratio or self.replace_ratio
                varied_ratio = ratio * (1.0 + random.uniform(-0.2, 0.2))  # ±20% 변화
                varied_ratio = max(0.1, min(1.0, varied_ratio))  # 0.1~1.0 범위 제한

                # 각 발화 패러프레이징
                augmented_turns = []                         # 증강된 발화 리스트
                for turn in dialogue:
                    augmented_turn = self.paraphrase(
                        turn,
                        replace_ratio=varied_ratio
                    )
                    augmented_turns.append(augmented_turn)

                augmented_samples.append(augmented_turns)

        # 지원하지 않는 타입
        else:
            self.logger.error(f"지원하지 않는 dialogue 타입: {type(dialogue)}")
            raise TypeError("dialogue는 str 또는 List[str] 타입이어야 함")

        return augmented_samples

    # ---------------------- 동의어 사전 추가 ---------------------- #
    def add_synonym(self, word: str, synonyms: List[str]):
        """
        동의어 사전에 새 단어 추가

        Args:
            word: 원본 단어
            synonyms: 동의어 리스트
        """

        # -------------- 기존 단어 확인 -------------- #
        if word in self.synonym_dict:
            # 기존 동의어에 추가
            self.synonym_dict[word].extend(synonyms)
            # 중복 제거
            self.synonym_dict[word] = list(set(self.synonym_dict[word]))
            self.logger.info(f"동의어 추가: '{word}' (+{len(synonyms)}개)")
        else:
            # 새 단어 추가
            self.synonym_dict[word] = synonyms
            self.logger.info(f"새 단어 추가: '{word}' ({len(synonyms)}개 동의어)")

    # ---------------------- 동의어 사전 저장 ---------------------- #
    def save_dictionary(self, filepath: str):
        """
        동의어 사전을 JSON 파일로 저장

        Args:
            filepath: 저장 경로
        """
        import json
        from pathlib import Path

        # -------------- 파일 저장 -------------- #
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)   # 디렉토리 생성

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                self.synonym_dict,
                f,
                ensure_ascii=False,                          # 한글 유지
                indent=2                                     # 들여쓰기
            )

        self.logger.info(f"동의어 사전 저장 완료: {filepath}")

    # ---------------------- 동의어 사전 로드 ---------------------- #
    @classmethod
    def load_dictionary(cls, filepath: str) -> "Paraphraser":
        """
        JSON 파일에서 동의어 사전 로드

        Args:
            filepath: 사전 파일 경로

        Returns:
            Paraphraser 인스턴스
        """
        import json

        # -------------- 파일 로드 -------------- #
        with open(filepath, 'r', encoding='utf-8') as f:
            synonym_dict = json.load(f)

        # -------------- Paraphraser 생성 -------------- #
        paraphraser = cls(synonym_dict=synonym_dict)

        paraphraser.logger.info(f"동의어 사전 로드 완료: {filepath}")

        return paraphraser


# ==================== 헬퍼 함수들 ==================== #
# ---------------------- Paraphraser 생성 함수 ---------------------- #
def create_paraphraser(
    synonym_dict: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> Paraphraser:
    """
    Paraphraser 인스턴스 생성 헬퍼

    Args:
        synonym_dict: 동의어 사전 (None이면 기본 사전 사용)
        **kwargs: Paraphraser 생성자 인자

    Returns:
        Paraphraser 인스턴스
    """

    # -------------- Paraphraser 생성 -------------- #
    paraphraser = Paraphraser(synonym_dict=synonym_dict, **kwargs)

    return paraphraser


# ---------------------- 메인 실행부 ---------------------- #
if __name__ == "__main__":
    # 간단한 테스트 코드

    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    # Paraphraser 생성
    paraphraser = create_paraphraser(replace_ratio=0.5)

    # 테스트 텍스트
    test_text = "안녕하세요. 오늘 회사에서 새로운 제품에 대해 말하다가 좋은 아이디어가 생각났어요."

    # 패러프레이징 실행
    result = paraphraser.paraphrase(test_text)

    print(f"원본: {test_text}")
    print(f"패러프레이징: {result}")

    # 여러 번 증강
    augmented = paraphraser.augment_dialogue(test_text, num_augmentations=3)
    print("\n증강된 샘플들:")
    for i, aug in enumerate(augmented, 1):
        print(f"{i}. {aug}")
