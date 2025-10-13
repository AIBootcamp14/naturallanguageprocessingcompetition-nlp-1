"""
데이터 증강 시스템

PRD 04: 성능 개선 전략 구현
- Back-translation (한→영→한)
- Paraphrase 생성
- 문장 순서 섞기
- 동의어 치환
- Dialogue Sampling
"""

import random
from typing import List, Tuple, Optional
from transformers import MarianMTModel, MarianTokenizer, pipeline
import re


class DataAugmenter:
    """데이터 증강 시스템"""

    def __init__(self, logger=None):
        """
        Args:
            logger: Logger 인스턴스
        """
        self.logger = logger
        self._log("DataAugmenter 초기화")

        # Back-translation 모델 (지연 로딩)
        self.ko_en_model = None
        self.ko_en_tokenizer = None
        self.en_ko_model = None
        self.en_ko_tokenizer = None

        # 증강 방법 등록
        self.augmenters = {
            'back_translate': BackTranslationAugmenter(),
            'paraphrase': ParaphraseAugmenter(),
            'shuffle': ShuffleAugmenter(),
            'synonym': SynonymReplacementAugmenter(),
            'sample': DialogueSamplingAugmenter()
        }

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def augment(
        self,
        dialogues: List[str],
        summaries: List[str],
        methods: List[str] = ["shuffle"],
        samples_per_method: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        데이터 증강 실행

        Args:
            dialogues: 대화 리스트
            summaries: 요약 리스트
            methods: 증강 방법 리스트
                     ["back_translate", "paraphrase", "shuffle", "synonym", "sample"]
            samples_per_method: 방법당 생성할 샘플 수

        Returns:
            (증강된 dialogues, 증강된 summaries)
        """
        augmented_dialogues = []
        augmented_summaries = []

        self._log(f"\n데이터 증강 시작")
        self._log(f"  - 원본 데이터: {len(dialogues)}개")
        self._log(f"  - 증강 방법: {methods}")
        self._log(f"  - 방법당 샘플 수: {samples_per_method}")

        for dialogue, summary in zip(dialogues, summaries):
            # 원본 추가
            augmented_dialogues.append(dialogue)
            augmented_summaries.append(summary)

            # 증강 데이터 생성
            for method in methods:
                for _ in range(samples_per_method):
                    try:
                        if method == "back_translate":
                            aug_dialogue = self.back_translate(dialogue)
                        elif method == "paraphrase":
                            aug_dialogue = self.paraphrase(dialogue)
                        elif method == "shuffle":
                            aug_dialogue = self.shuffle_turns(dialogue)
                        elif method == "synonym":
                            aug_dialogue = self.synonym_replacement(dialogue)
                        elif method == "sample":
                            aug_dialogue = self.sample_dialogue(dialogue)
                        else:
                            self._log(f"알 수 없는 증강 방법: {method}")
                            continue

                        if aug_dialogue and aug_dialogue != dialogue:
                            augmented_dialogues.append(aug_dialogue)
                            augmented_summaries.append(summary)

                    except Exception as e:
                        self._log(f"증강 실패 ({method}): {str(e)}")
                        continue

        self._log(f"데이터 증강 완료: {len(augmented_dialogues)}개")
        return augmented_dialogues, augmented_summaries

    def back_translate(self, text: str) -> str:
        """
        Back-translation (한→영→한)

        Args:
            text: 한국어 텍스트

        Returns:
            역번역된 한국어 텍스트
        """
        # 모델 로딩 (지연 로딩)
        if self.ko_en_model is None:
            self._log("Back-translation 모델 로딩 중... (Helsinki-NLP/opus-mt-ko-en)")
            self.ko_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
            self.ko_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

        if self.en_ko_model is None:
            self._log("Back-translation 모델 로딩 중... (Helsinki-NLP/opus-mt-en-ko)")
            self.en_ko_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ko")
            self.en_ko_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ko")

        try:
            # 한→영
            inputs = self.ko_en_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            translated_en = self.ko_en_model.generate(**inputs, max_length=512)
            en_text = self.ko_en_tokenizer.decode(translated_en[0], skip_special_tokens=True)

            # 영→한
            inputs = self.en_ko_tokenizer(en_text, return_tensors="pt", truncation=True, max_length=512)
            translated_ko = self.en_ko_model.generate(**inputs, max_length=512)
            ko_text = self.en_ko_tokenizer.decode(translated_ko[0], skip_special_tokens=True)

            return ko_text

        except Exception as e:
            self._log(f"Back-translation 실패: {str(e)}")
            return text

    def paraphrase(self, text: str) -> str:
        """
        Paraphrase 생성 (간단한 규칙 기반)

        Note: 실제 프로덕션에서는 T5/KoGPT 모델 사용 권장

        Args:
            text: 원본 텍스트

        Returns:
            패러프레이즈된 텍스트
        """
        # 간단한 동의어 치환 규칙
        replacements = {
            "안녕하세요": ["안녕", "반갑습니다", "환영합니다"],
            "감사합니다": ["고맙습니다", "감사해요", "고마워요"],
            "죄송합니다": ["미안합니다", "죄송해요", "미안해요"],
            "네": ["예", "알겠습니다", "그렇습니다"],
            "아니요": ["아닙니다", "아니에요", "그렇지 않습니다"],
        }

        paraphrased = text
        for original, synonyms in replacements.items():
            if original in paraphrased:
                paraphrased = paraphrased.replace(original, random.choice(synonyms))

        return paraphrased

    def shuffle_turns(self, dialogue: str, preserve_ratio: float = 0.3) -> str:
        """
        대화 턴 순서 섞기 (일부만)

        Args:
            dialogue: 대화 텍스트
            preserve_ratio: 유지할 턴 비율 (처음/끝 보존)

        Returns:
            턴이 섞인 대화
        """
        # Person 태그로 턴 분리
        turns = re.split(r'(#Person\d+#:)', dialogue)
        turns = [t.strip() for t in turns if t.strip()]

        if len(turns) < 6:  # 너무 짧으면 섞지 않음
            return dialogue

        # Person 태그와 내용을 쌍으로 묶기
        paired_turns = []
        for i in range(0, len(turns) - 1, 2):
            if i + 1 < len(turns):
                paired_turns.append(turns[i] + " " + turns[i + 1])

        if len(paired_turns) < 3:
            return dialogue

        # 처음/끝 보존, 중간만 섞기
        preserve_count = max(1, int(len(paired_turns) * preserve_ratio))
        start_turns = paired_turns[:preserve_count]
        end_turns = paired_turns[-preserve_count:]
        middle_turns = paired_turns[preserve_count:-preserve_count]

        # 중간 턴 섞기
        random.shuffle(middle_turns)

        # 재조합
        shuffled_turns = start_turns + middle_turns + end_turns
        return " ".join(shuffled_turns)

    def synonym_replacement(self, text: str, n: int = 3) -> str:
        """
        동의어 치환 (간단한 버전)

        Args:
            text: 원본 텍스트
            n: 치환할 단어 수

        Returns:
            동의어가 치환된 텍스트
        """
        # 한국어 동의어 사전 (간단한 예시)
        synonyms = {
            "좋다": ["훌륭하다", "멋지다", "괜찮다"],
            "나쁘다": ["안좋다", "별로다", "형편없다"],
            "크다": ["거대하다", "넓다", "광대하다"],
            "작다": ["적다", "미미하다", "소소하다"],
            "빠르다": ["신속하다", "재빠르다", "날쌔다"],
            "느리다": ["더디다", "굼뜨다", "늦다"],
        }

        result = text
        replaced_count = 0

        for original, synonym_list in synonyms.items():
            if original in result and replaced_count < n:
                result = result.replace(original, random.choice(synonym_list), 1)
                replaced_count += 1

        return result

    def sample_dialogue(self, dialogue: str, ratio: float = 0.8) -> str:
        """
        대화 샘플링 (일부 턴 선택)

        Args:
            dialogue: 대화 텍스트
            ratio: 유지할 턴 비율

        Returns:
            샘플링된 대화
        """
        # Person 태그로 턴 분리
        turns = re.split(r'(#Person\d+#:)', dialogue)
        turns = [t.strip() for t in turns if t.strip()]

        if len(turns) < 4:  # 너무 짧으면 샘플링 안 함
            return dialogue

        # Person 태그와 내용을 쌍으로 묶기
        paired_turns = []
        for i in range(0, len(turns) - 1, 2):
            if i + 1 < len(turns):
                paired_turns.append(turns[i] + " " + turns[i + 1])

        # 유지할 턴 수 계산
        keep_count = max(2, int(len(paired_turns) * ratio))

        # 중요한 턴 우선 선택 (처음, 끝, 랜덤)
        if len(paired_turns) <= keep_count:
            return dialogue

        # 처음과 끝은 항상 유지
        sampled_turns = [paired_turns[0]]

        # 중간에서 랜덤 선택
        middle_turns = paired_turns[1:-1]
        if middle_turns:
            sample_size = keep_count - 2  # 처음/끝 제외
            sampled_middle = random.sample(middle_turns, min(sample_size, len(middle_turns)))
            sampled_turns.extend(sampled_middle)

        # 마지막 턴 추가
        sampled_turns.append(paired_turns[-1])

        return " ".join(sampled_turns)


def augment_data(
    dialogues: List[str],
    summaries: List[str],
    methods: List[str] = ["shuffle"],
    samples_per_method: int = 1,
    logger=None
) -> Tuple[List[str], List[str]]:
    """
    편의 함수: 데이터 증강

    Args:
        dialogues: 대화 리스트
        summaries: 요약 리스트
        methods: 증강 방법 리스트
        samples_per_method: 방법당 샘플 수
        logger: Logger 인스턴스

    Returns:
        (증강된 dialogues, 증강된 summaries)
    """
    augmenter = DataAugmenter(logger=logger)
    return augmenter.augment(dialogues, summaries, methods, samples_per_method)


# 개별 증강기 클래스들
class BackTranslationAugmenter:
    """역번역 증강기"""

    def __init__(self):
        self.ko_en_model = None
        self.ko_en_tokenizer = None
        self.en_ko_model = None
        self.en_ko_tokenizer = None

    def augment(self, dialogue: str, summary: str) -> Tuple[str, str]:
        """역번역 수행"""
        # 모델 로딩 (지연 로딩)
        if self.ko_en_model is None:
            self.ko_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
            self.ko_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
            self.en_ko_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ko")
            self.en_ko_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ko")

        # 대화 역번역
        inputs = self.ko_en_tokenizer(dialogue, return_tensors="pt", truncation=True, max_length=512)
        translated_en = self.ko_en_model.generate(**inputs, max_length=512)
        en_text = self.ko_en_tokenizer.decode(translated_en[0], skip_special_tokens=True)

        inputs = self.en_ko_tokenizer(en_text, return_tensors="pt", truncation=True, max_length=512)
        translated_ko = self.en_ko_model.generate(**inputs, max_length=512)
        aug_dialogue = self.en_ko_tokenizer.decode(translated_ko[0], skip_special_tokens=True)

        # 요약 역번역
        inputs = self.ko_en_tokenizer(summary, return_tensors="pt", truncation=True, max_length=512)
        translated_en = self.ko_en_model.generate(**inputs, max_length=512)
        en_text = self.ko_en_tokenizer.decode(translated_en[0], skip_special_tokens=True)

        inputs = self.en_ko_tokenizer(en_text, return_tensors="pt", truncation=True, max_length=512)
        translated_ko = self.en_ko_model.generate(**inputs, max_length=512)
        aug_summary = self.en_ko_tokenizer.decode(translated_ko[0], skip_special_tokens=True)

        return aug_dialogue, aug_summary


class ParaphraseAugmenter:
    """의역 증강기"""

    def __init__(self):
        self.replacements = {
            "안녕하세요": ["안녕", "반갑습니다", "환영합니다"],
            "감사합니다": ["고맙습니다", "감사해요", "고마워요"],
            "죄송합니다": ["미안합니다", "죄송해요", "미안해요"],
            "네": ["예", "알겠습니다", "그렇습니다"],
            "아니요": ["아닙니다", "아니에요", "그렇지 않습니다"],
        }

    def augment(self, dialogue: str, summary: str) -> Tuple[str, str]:
        """의역 수행"""
        aug_dialogue = dialogue
        for original, synonyms in self.replacements.items():
            if original in aug_dialogue:
                aug_dialogue = aug_dialogue.replace(original, random.choice(synonyms))

        aug_summary = summary
        for original, synonyms in self.replacements.items():
            if original in aug_summary:
                aug_summary = aug_summary.replace(original, random.choice(synonyms))

        return aug_dialogue, aug_summary


class ShuffleAugmenter:
    """턴 섞기 증강기"""

    def __init__(self, preserve_ratio: float = 0.3):
        self.preserve_ratio = preserve_ratio

    def augment(self, dialogue: str, summary: str) -> Tuple[str, str]:
        """턴 섞기 수행"""
        # 줄바꿈으로 턴 분리 (일반적인 대화 형식)
        turns = [t.strip() for t in dialogue.split('\n') if t.strip()]

        if len(turns) < 3:
            return dialogue, summary

        # 처음/끝 보존, 중간만 섞기
        preserve_count = max(1, int(len(turns) * self.preserve_ratio))
        start_turns = turns[:preserve_count]
        end_turns = turns[-preserve_count:] if preserve_count > 0 else []
        middle_turns = turns[preserve_count:-preserve_count] if preserve_count > 0 else turns

        if middle_turns:
            random.shuffle(middle_turns)

        shuffled_turns = start_turns + middle_turns + end_turns
        return '\n'.join(shuffled_turns), summary


class SynonymReplacementAugmenter:
    """동의어 치환 증강기"""

    def __init__(self, replace_ratio: float = 0.3):
        self.replace_ratio = replace_ratio
        self.synonyms = {
            "좋다": ["훌륭하다", "멋지다", "괜찮다"],
            "나쁘다": ["안좋다", "별로다", "형편없다"],
            "크다": ["거대하다", "넓다", "광대하다"],
            "작다": ["적다", "미미하다", "소소하다"],
            "빠르다": ["신속하다", "재빠르다", "날쌔다"],
            "느리다": ["더디다", "굼뜨다", "늦다"],
            "밥": ["식사", "음식", "끼니"],
            "먹다": ["섭취하다", "드시다", "식사하다"],
        }

    def augment(self, dialogue: str, summary: str) -> Tuple[str, str]:
        """동의어 치환 수행"""
        aug_dialogue = dialogue
        for original, synonym_list in self.synonyms.items():
            if original in aug_dialogue:
                aug_dialogue = aug_dialogue.replace(original, random.choice(synonym_list))

        return aug_dialogue, summary


class DialogueSamplingAugmenter:
    """대화 샘플링 증강기"""

    def __init__(self, sample_ratio: float = 0.7):
        self.sample_ratio = sample_ratio

    def augment(self, dialogue: str, summary: str) -> Tuple[str, str]:
        """대화 샘플링 수행"""
        turns = [t.strip() for t in dialogue.split('\n') if t.strip()]

        if len(turns) < 3:
            return dialogue, summary

        # 유지할 턴 수 계산
        keep_count = max(2, int(len(turns) * self.sample_ratio))

        if len(turns) <= keep_count:
            return dialogue, summary

        # 처음과 끝은 항상 유지
        sampled_turns = [turns[0]]

        # 중간에서 랜덤 선택
        middle_turns = turns[1:-1]
        if middle_turns and keep_count > 2:
            sample_size = keep_count - 2
            sampled_middle = random.sample(middle_turns, min(sample_size, len(middle_turns)))
            sampled_turns.extend(sampled_middle)

        # 마지막 턴 추가
        if len(turns) > 1:
            sampled_turns.append(turns[-1])

        return '\n'.join(sampled_turns), summary


def augment_dataset(
    dialogues: List[str],
    summaries: List[str],
    methods: List[str] = ["shuffle"],
    n_aug: int = 1
) -> Tuple[List[str], List[str]]:
    """
    편의 함수: 데이터셋 증강

    Args:
        dialogues: 대화 리스트
        summaries: 요약 리스트
        methods: 증강 방법 리스트
        n_aug: 방법당 생성할 증강 데이터 수

    Returns:
        (증강된 dialogues, 증강된 summaries)
    """
    augmenters_map = {
        'back_translate': BackTranslationAugmenter(),
        'paraphrase': ParaphraseAugmenter(),
        'shuffle': ShuffleAugmenter(),
        'synonym': SynonymReplacementAugmenter(),
        'sample': DialogueSamplingAugmenter()
    }

    augmented_dialogues = []
    augmented_summaries = []

    for dialogue, summary in zip(dialogues, summaries):
        for method in methods:
            for _ in range(n_aug):
                if method in augmenters_map:
                    try:
                        aug_dialogue, aug_summary = augmenters_map[method].augment(dialogue, summary)
                        augmented_dialogues.append(aug_dialogue)
                        augmented_summaries.append(aug_summary)
                    except Exception as e:
                        print(f"증강 실패 ({method}): {str(e)}")
                        continue

    return augmented_dialogues, augmented_summaries
