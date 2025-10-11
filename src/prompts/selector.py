"""
프롬프트 동적 선택 시스템

PRD 15: 프롬프트 엔지니어링 전략
"""

from typing import Dict, Optional, List, Any
import re

from .template import PromptLibrary, PromptTemplate


class PromptSelector:
    """
    대화 특성에 따라 최적 프롬프트를 동적으로 선택하는 클래스

    Features:
    - 대화 길이 기반 선택
    - 참여자 수 기반 선택
    - 토큰 예산 기반 선택
    - 카테고리별 선택 전략
    """

    def __init__(self, prompt_library: Optional[PromptLibrary] = None):
        """
        초기화

        Args:
            prompt_library: PromptLibrary 인스턴스
        """
        self.library = prompt_library or PromptLibrary()

        # 길이 임계값 (단어 수 기준)
        self.length_thresholds = {
            'short': 200,
            'medium': 500,
            'long': float('inf')
        }

        # 참여자 수 임계값
        self.speaker_thresholds = {
            'two': 2,
            'small_group': 4,
            'large_group': float('inf')
        }

    def select_by_length(self, dialogue: str) -> PromptTemplate:
        """
        대화 길이에 따라 프롬프트 선택

        Args:
            dialogue: 대화 텍스트

        Returns:
            선택된 PromptTemplate
        """
        word_count = len(dialogue.split())

        if word_count < self.length_thresholds['short']:
            return self.library.get_template('short_dialogue')
        elif word_count < self.length_thresholds['medium']:
            return self.library.get_template('medium_dialogue')
        else:
            return self.library.get_template('long_dialogue')

    def select_by_speakers(self, dialogue: str) -> PromptTemplate:
        """
        참여자 수에 따라 프롬프트 선택

        Args:
            dialogue: 대화 텍스트

        Returns:
            선택된 PromptTemplate
        """
        num_speakers = self._count_speakers(dialogue)

        if num_speakers <= self.speaker_thresholds['two']:
            return self.library.get_template('two_speakers')
        elif num_speakers <= self.speaker_thresholds['small_group']:
            return self.library.get_template('group_small')
        else:
            return self.library.get_template('group_large')

    def select_by_token_budget(
        self,
        dialogue: str,
        token_budget: int = 512
    ) -> PromptTemplate:
        """
        토큰 예산에 따라 프롬프트 선택

        Args:
            dialogue: 대화 텍스트
            token_budget: 최대 토큰 수

        Returns:
            선택된 PromptTemplate
        """
        dialogue_tokens = self._estimate_tokens(dialogue)

        # 대화가 토큰 예산의 80% 이상을 차지하면 압축 템플릿
        if dialogue_tokens > token_budget * 0.8:
            return self.library.get_template('compressed_minimal')
        elif dialogue_tokens > token_budget * 0.6:
            return self.library.get_template('compressed_concise')
        else:
            # 여유가 있으면 상세한 템플릿
            return self.library.get_template('zero_shot_detailed')

    def select_by_category(
        self,
        category: str,
        dialogue: str,
        **kwargs
    ) -> PromptTemplate:
        """
        카테고리별 최적 프롬프트 선택

        Args:
            category: 카테고리 (zero_shot, few_shot, cot, etc.)
            dialogue: 대화 텍스트
            **kwargs: 추가 파라미터

        Returns:
            선택된 PromptTemplate
        """
        if category == "zero_shot":
            return self._select_zero_shot(dialogue)
        elif category == "few_shot":
            return self._select_few_shot(dialogue, **kwargs)
        elif category == "cot":
            return self._select_cot(dialogue)
        elif category == "length_specific":
            return self.select_by_length(dialogue)
        elif category == "speaker_specific":
            return self.select_by_speakers(dialogue)
        elif category == "compressed":
            token_budget = kwargs.get('token_budget', 512)
            return self.select_by_token_budget(dialogue, token_budget)
        else:
            raise ValueError(f"Unknown category: {category}")

    def select_adaptive(
        self,
        dialogue: str,
        token_budget: Optional[int] = None,
        prefer_category: Optional[str] = None
    ) -> PromptTemplate:
        """
        대화 특성을 종합적으로 분석하여 최적 프롬프트 선택

        Args:
            dialogue: 대화 텍스트
            token_budget: 토큰 예산 (optional)
            prefer_category: 선호 카테고리 (optional)

        Returns:
            선택된 PromptTemplate
        """
        # 1. 토큰 예산이 주어진 경우 우선 고려
        if token_budget:
            dialogue_tokens = self._estimate_tokens(dialogue)
            if dialogue_tokens > token_budget * 0.7:
                return self.select_by_token_budget(dialogue, token_budget)

        # 2. 선호 카테고리가 있으면 해당 카테고리 내에서 선택
        if prefer_category:
            if prefer_category == "zero_shot":
                return self._select_zero_shot(dialogue)
            elif prefer_category == "cot":
                return self._select_cot(dialogue)

        # 3. 기본 전략: 대화 길이 기반
        return self.select_by_length(dialogue)

    def _select_zero_shot(self, dialogue: str) -> PromptTemplate:
        """Zero-shot 카테고리 내에서 최적 템플릿 선택"""
        word_count = len(dialogue.split())

        if word_count < 150:
            return self.library.get_template('zero_shot_basic')
        else:
            return self.library.get_template('zero_shot_detailed')

    def _select_few_shot(self, dialogue: str, **kwargs) -> PromptTemplate:
        """Few-shot 카테고리 내에서 최적 템플릿 선택"""
        # 예시 개수에 따라 선택
        num_examples = kwargs.get('num_examples', 1)

        if num_examples == 1:
            return self.library.get_template('few_shot_1shot')
        elif num_examples == 2:
            return self.library.get_template('few_shot_2shot')
        else:
            return self.library.get_template('few_shot_3shot')

    def _select_cot(self, dialogue: str) -> PromptTemplate:
        """CoT 카테고리 내에서 최적 템플릿 선택"""
        word_count = len(dialogue.split())

        # 긴 대화는 분석적 CoT가 더 효과적
        if word_count > 300:
            return self.library.get_template('cot_analytical')
        else:
            return self.library.get_template('cot_step_by_step')

    def _count_speakers(self, dialogue: str) -> int:
        """
        대화 참여자 수 추정

        Args:
            dialogue: 대화 텍스트

        Returns:
            참여자 수
        """
        # Person1, Person2, ... 또는 A:, B:, ... 패턴 찾기
        person_pattern = r'#?Person(\d+)#?:|[A-Z]:'
        matches = re.findall(person_pattern, dialogue)

        if not matches:
            return 2  # 기본값

        # Person 숫자 추출
        person_nums = []
        for match in re.finditer(person_pattern, dialogue):
            if match.group(1):  # Person숫자 패턴
                person_nums.append(int(match.group(1)))
            else:  # A:, B: 패턴
                # 알파벳을 숫자로 변환
                letter = match.group(0)[0]
                person_nums.append(ord(letter) - ord('A') + 1)

        return len(set(person_nums)) if person_nums else 2

    def _estimate_tokens(self, text: str) -> int:
        """
        토큰 수 추정

        Args:
            text: 텍스트

        Returns:
            추정 토큰 수
        """
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - korean_chars - english_chars

        estimated = (korean_chars / 2.5) + (english_chars / 4) + (other_chars / 3)
        return int(estimated)

    def get_selection_info(self, dialogue: str) -> Dict[str, Any]:
        """
        대화 분석 정보 반환

        Args:
            dialogue: 대화 텍스트

        Returns:
            분석 정보 딕셔너리
        """
        word_count = len(dialogue.split())
        num_speakers = self._count_speakers(dialogue)
        estimated_tokens = self._estimate_tokens(dialogue)

        # 각 기준별 추천 템플릿
        length_template = self.select_by_length(dialogue)
        speaker_template = self.select_by_speakers(dialogue)
        token_template = self.select_by_token_budget(dialogue, 512)

        return {
            'word_count': word_count,
            'num_speakers': num_speakers,
            'estimated_tokens': estimated_tokens,
            'recommendations': {
                'by_length': length_template.name,
                'by_speakers': speaker_template.name,
                'by_token_budget': token_template.name
            },
            'characteristics': {
                'is_short': word_count < self.length_thresholds['short'],
                'is_long': word_count >= self.length_thresholds['medium'],
                'is_two_person': num_speakers == 2,
                'is_group': num_speakers > 2
            }
        }


def create_prompt_selector(prompt_library: Optional[PromptLibrary] = None) -> PromptSelector:
    """
    PromptSelector 편의 생성 함수

    Args:
        prompt_library: PromptLibrary 인스턴스

    Returns:
        PromptSelector 인스턴스
    """
    return PromptSelector(prompt_library)
