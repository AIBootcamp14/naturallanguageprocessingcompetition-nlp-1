"""
프롬프트 템플릿 관리 시스템

PRD 15: 프롬프트 엔지니어링 전략
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import re


@dataclass
class PromptTemplate:
    """
    프롬프트 템플릿 클래스

    Attributes:
        name: 템플릿 이름
        template: 프롬프트 템플릿 문자열
        description: 템플릿 설명
        category: 카테고리 (zero_shot, few_shot, cot)
        variables: 템플릿 변수 목록
    """
    name: str
    template: str
    description: str
    category: str
    variables: List[str]

    def format(self, **kwargs) -> str:
        """
        템플릿에 변수를 채워 완성된 프롬프트 생성

        Args:
            **kwargs: 템플릿 변수값

        Returns:
            완성된 프롬프트 문자열
        """
        # 필수 변수 확인
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # 템플릿 포맷팅
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Invalid variable in template: {e}")


class PromptLibrary:
    """
    프롬프트 템플릿 라이브러리

    다양한 프롬프트 템플릿을 관리하고 제공
    """

    def __init__(self):
        """초기화 및 기본 템플릿 로드"""
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """기본 프롬프트 템플릿 로드"""

        # ==================== Zero-shot 템플릿 ==================== #

        self.add_template(PromptTemplate(
            name="zero_shot_basic",
            template="""다음 대화를 요약해주세요:

{dialogue}

요약:""",
            description="기본 Zero-shot 템플릿",
            category="zero_shot",
            variables=["dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="zero_shot_detailed",
            template="""아래 대화를 읽고 핵심 내용을 요약해주세요.

요구사항:
- 3-5문장으로 작성
- 주요 주제와 결론 포함
- 중요한 정보 누락 없이

대화:
{dialogue}

요약:""",
            description="상세한 Zero-shot 템플릿",
            category="zero_shot",
            variables=["dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="zero_shot_structured",
            template="""[태스크] 대화 요약
[형식] 한 문단, 3-5문장
[스타일] 객관적, 간결함

대화 내용:
{dialogue}

요약 결과:""",
            description="구조화된 Zero-shot 템플릿",
            category="zero_shot",
            variables=["dialogue"]
        ))

        # ==================== Few-shot 템플릿 ==================== #

        self.add_template(PromptTemplate(
            name="few_shot_1shot",
            template="""대화 요약 예시를 참고하여 새로운 대화를 요약해주세요.

예시:
대화: {example1_dialogue}
요약: {example1_summary}

이제 다음 대화를 요약해주세요:
대화: {dialogue}
요약:""",
            description="1-shot Few-shot 템플릿",
            category="few_shot",
            variables=["example1_dialogue", "example1_summary", "dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="few_shot_2shot",
            template="""대화 요약 예시를 참고하여 마지막 대화를 요약해주세요.

예시 1:
대화: {example1_dialogue}
요약: {example1_summary}

예시 2:
대화: {example2_dialogue}
요약: {example2_summary}

이제 다음 대화를 요약해주세요:
대화: {dialogue}
요약:""",
            description="2-shot Few-shot 템플릿",
            category="few_shot",
            variables=["example1_dialogue", "example1_summary",
                      "example2_dialogue", "example2_summary", "dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="few_shot_3shot",
            template="""다음은 대화 요약 예시입니다. 패턴을 학습하여 적용해주세요.

[예시 1]
대화: {example1_dialogue}
요약: {example1_summary}

[예시 2]
대화: {example2_dialogue}
요약: {example2_summary}

[예시 3]
대화: {example3_dialogue}
요약: {example3_summary}

[실제 과제]
대화: {dialogue}
요약:""",
            description="3-shot Few-shot 템플릿",
            category="few_shot",
            variables=["example1_dialogue", "example1_summary",
                      "example2_dialogue", "example2_summary",
                      "example3_dialogue", "example3_summary", "dialogue"]
        ))

        # ==================== Chain-of-Thought 템플릿 ==================== #

        self.add_template(PromptTemplate(
            name="cot_step_by_step",
            template="""다음 대화를 단계별로 분석하여 요약해주세요.

단계 1: 대화의 주요 주제 파악
단계 2: 핵심 정보와 결정사항 추출
단계 3: 부수적 정보 제거
단계 4: 간결한 문장으로 정리

대화:
{dialogue}

최종 요약:""",
            description="Step-by-step CoT 템플릿",
            category="cot",
            variables=["dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="cot_analytical",
            template="""대화를 분석적으로 요약해주세요.

대화: {dialogue}

먼저 생각해봅시다:
- 이 대화의 목적은 무엇인가?
- 어떤 결론에 도달했는가?
- 가장 중요한 정보는 무엇인가?

이를 바탕으로 한 요약:""",
            description="분석적 CoT 템플릿",
            category="cot",
            variables=["dialogue"]
        ))

        # ==================== 대화 길이별 템플릿 ==================== #

        self.add_template(PromptTemplate(
            name="short_dialogue",
            template="""짧은 대화입니다. 핵심만 간단히 요약해주세요:

{dialogue}

요약:""",
            description="짧은 대화용 템플릿",
            category="length_specific",
            variables=["dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="medium_dialogue",
            template="""다음 대화의 주요 내용을 3-4문장으로 요약해주세요:

{dialogue}

요약:""",
            description="중간 길이 대화용 템플릿",
            category="length_specific",
            variables=["dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="long_dialogue",
            template="""긴 대화입니다. 주요 주제별로 구조화하여 5-6문장으로 요약해주세요:

{dialogue}

요약:""",
            description="긴 대화용 템플릿",
            category="length_specific",
            variables=["dialogue"]
        ))

        # ==================== 참여자 수별 템플릿 ==================== #

        self.add_template(PromptTemplate(
            name="two_speakers",
            template="""두 사람의 대화입니다. 각자의 입장과 합의점을 중심으로 요약해주세요:

{dialogue}

요약:""",
            description="2인 대화용 템플릿",
            category="speaker_specific",
            variables=["dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="group_small",
            template="""소그룹 대화입니다. 주요 의견들과 결론을 정리해주세요:

{dialogue}

요약:""",
            description="소그룹(3-4인) 대화용 템플릿",
            category="speaker_specific",
            variables=["dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="group_large",
            template="""다수가 참여한 대화입니다. 핵심 주제와 주요 결정사항 중심으로 요약해주세요:

{dialogue}

요약:""",
            description="대규모(5인 이상) 대화용 템플릿",
            category="speaker_specific",
            variables=["dialogue"]
        ))

        # ==================== 압축된 템플릿 (토큰 최적화) ==================== #

        self.add_template(PromptTemplate(
            name="compressed_minimal",
            template="""{dialogue}

요약:""",
            description="최소 토큰 템플릿",
            category="compressed",
            variables=["dialogue"]
        ))

        self.add_template(PromptTemplate(
            name="compressed_concise",
            template="""대화: {dialogue}

핵심 요약:""",
            description="간결한 템플릿",
            category="compressed",
            variables=["dialogue"]
        ))

    def add_template(self, template: PromptTemplate):
        """
        템플릿 추가

        Args:
            template: PromptTemplate 객체
        """
        self.templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        템플릿 조회

        Args:
            name: 템플릿 이름

        Returns:
            PromptTemplate 또는 None
        """
        return self.templates.get(name)

    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """
        템플릿 목록 조회

        Args:
            category: 카테고리 필터 (optional)

        Returns:
            템플릿 이름 목록
        """
        if category:
            return [
                name for name, template in self.templates.items()
                if template.category == category
            ]
        return list(self.templates.keys())

    def get_templates_by_category(self, category: str) -> List[PromptTemplate]:
        """
        카테고리별 템플릿 조회

        Args:
            category: 카테고리명

        Returns:
            PromptTemplate 목록
        """
        return [
            template for template in self.templates.values()
            if template.category == category
        ]

    def estimate_tokens(self, template_name: str, **kwargs) -> int:
        """
        템플릿 토큰 수 추정

        Args:
            template_name: 템플릿 이름
            **kwargs: 템플릿 변수값

        Returns:
            추정 토큰 수
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        prompt = template.format(**kwargs)

        # 토큰 추정 (한글: 2.5자/토큰, 영어: 4자/토큰)
        korean_chars = len(re.findall(r'[가-힣]', prompt))
        english_chars = len(re.findall(r'[a-zA-Z]', prompt))
        other_chars = len(prompt) - korean_chars - english_chars

        estimated = (korean_chars / 2.5) + (english_chars / 4) + (other_chars / 3)
        return int(estimated)


def create_prompt_library() -> PromptLibrary:
    """
    PromptLibrary 편의 생성 함수

    Returns:
        PromptLibrary 인스턴스
    """
    return PromptLibrary()
