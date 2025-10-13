# ==================== 프롬프트 템플릿 ==================== #
"""
프롬프트 엔지니어링 템플릿

PRD 10: 프롬프트 엔지니어링 모듈
- Few-shot 템플릿
- Zero-shot 템플릿
- Chain-of-Thought 템플릿
- 역할 기반 템플릿
"""

# ---------------------- 외부 라이브러리 ---------------------- #
from typing import List, Dict, Optional
from dataclasses import dataclass


# ==================== 프롬프트 템플릿 데이터 클래스 ==================== #
@dataclass
class PromptTemplate:
    """프롬프트 템플릿 데이터 클래스"""
    name: str
    template: str
    description: str
    example_input: Optional[str] = None
    example_output: Optional[str] = None


# ==================== Zero-shot 템플릿 ==================== #
ZERO_SHOT_SIMPLE = PromptTemplate(
    name="zero_shot_simple",
    template="""다음 대화를 간략하게 요약해주세요:

{dialogue}

요약:""",
    description="가장 단순한 zero-shot 템플릿",
    example_input="#Person1#: 안녕하세요. 오늘 날씨가 정말 좋네요.\n#Person2#: 네, 맞아요. 산책하기 좋은 날씨네요.",
    example_output="두 사람이 날씨에 대해 이야기하고 산책하기 좋다고 동의함."
)

ZERO_SHOT_DETAILED = PromptTemplate(
    name="zero_shot_detailed",
    template="""당신은 대화 요약 전문가입니다. 다음 대화의 핵심 내용을 파악하여 간략하게 한 문장으로 요약해주세요.

대화:
{dialogue}

핵심 요약:""",
    description="역할과 지시사항을 추가한 zero-shot 템플릿"
)

ZERO_SHOT_STRUCTURED = PromptTemplate(
    name="zero_shot_structured",
    template="""### 지시사항
대화 내용을 분석하여 핵심 정보만 포함된 간략한 요약문을 작성하세요.

### 대화
{dialogue}

### 요약
""",
    description="구조화된 zero-shot 템플릿"
)


# ==================== Few-shot 템플릿 ==================== #
FEW_SHOT_BASIC = PromptTemplate(
    name="few_shot_basic",
    template="""다음은 대화를 요약한 예시입니다:

예시 1:
대화: #Person1#: 내일 몇 시에 만날까요? #Person2#: 오후 3시는 어떠세요? 저는 괜찮습니다. #Person1#: 좋아요! 그럼 내일 뵙겠습니다.
요약: 두 사람이 내일 오후 3시에 만나기로 약속을 정했다.

예시 2:
대화: #Person1#: 저녁 메뉴 뭐예요? #Person2#: 오늘 저녁은 파스타입니다. #Person1#: 감사합니다. 맛있겠네요.
요약: 저녁 메뉴 오늘 저녁은 파스타라고 안내받았다.

이제 다음 대화를 요약해주세요:
대화: {dialogue}
요약:""",
    description="기본 few-shot 템플릿 (2 예시)"
)

FEW_SHOT_EXTENDED = PromptTemplate(
    name="few_shot_extended",
    template="""당신은 대화 요약 전문가입니다. 다음 예시들을 참고하여 대화를 간략하게 요약하세요.

[예시 1]
대화: #Person1#: 안녕하세요. 오늘 날씨가 정말 좋네요. #Person2#: 네, 맞아요. 산책하기 좋은 날씨네요.
요약: 두 사람이 날씨에 대해 이야기하고 산책하기 좋다고 동의함.

[예시 2]
대화: #Person1#: 이번 주말 영화 볼래요? #Person2#: 좋아요! 몇 시에 만날까요? #Person1#: 감사합니다. 오후 2시는 어떠세요.
요약: 주말에 영화 보러 가기로 오후 2시에 만나기로 약속함.

[예시 3]
대화: #Person1#: 오늘 간단히 점심 먹을까요? #Person2#: 좋아요. 샌드위치가 어떠세요. #Person1#: 감사합니다. 샌드위치 괜찮아 보여요.
요약: 간단히 점심으로 두 사람이 샌드위치를 먹기로 결정함.

[새로운 대화]
대화: {dialogue}
요약:""",
    description="확장된 few-shot 템플릿 (3 예시)"
)


# ==================== Chain-of-Thought 템플릿 ==================== #
COT_STEP_BY_STEP = PromptTemplate(
    name="cot_step_by_step",
    template="""다음 대화를 단계별로 분석하여 요약해주세요:

대화:
{dialogue}

분석 단계:
1. 대화 주제:
2. 핵심 내용:
3. 핵심 결론:
4. 최종 요약문:

간략한 최종 요약:""",
    description="단계별 사고 과정을 포함한 CoT 템플릿"
)

COT_REASONING = PromptTemplate(
    name="cot_reasoning",
    template="""당신은 대화 분석 전문가입니다. 다음 대화를 논리적으로 분석하여 요약하세요.

대화:
{dialogue}

분석 과정:
- 먼저, 이 대화의 참여자는 누구인가요?
- 다음으로, 대화자들이 나누고 있는 핵심 내용은 무엇인가요?
- 마지막으로, 대화의 결론이나 결정사항은 무엇인가요?

위 과정을 거쳐 간략한 최종 요약:""",
    description="논리적 사고 과정을 명시한 CoT 템플릿"
)


# ==================== 역할 기반 템플릿 ==================== #
ROLE_PROFESSIONAL = PromptTemplate(
    name="role_professional",
    template="""당신은 10년차 경력의 전문 요약 전문가입니다.
다음 대화를 분석하여 핵심 내용만 담아서 간략하고 명료하게 한 문장으로 요약해주세요.

대화:
{dialogue}

전문가 요약:""",
    description="전문 역할 기반 템플릿"
)

ROLE_ASSISTANT = PromptTemplate(
    name="role_assistant",
    template="""당신은 사용자를 돕는 AI 어시스턴트입니다.
사용자가 이해하기 쉽도록 다음 대화를 간략하게 요약해주세요.

대화:
{dialogue}

어시스턴트 요약:""",
    description="AI 어시스턴트 역할 기반 템플릿"
)


# ==================== 특수 목적 템플릿 ==================== #
KOREAN_FOCUSED = PromptTemplate(
    name="korean_focused",
    template="""다음은 한국어로 된 대화입니다. 자연스러운 한국어로 요약해주세요.

대화:
{dialogue}

자연스러운 한국어 요약:""",
    description="한국어 특화 템플릿"
)

CONCISE_FOCUS = PromptTemplate(
    name="concise_focus",
    template="""대화의 핵심 내용을 담아서 짧고 간결하게 요약하세요.
불필요한 세부사항은 제외하고 꼭 필요한 정보만 포함하세요.

대화:
{dialogue}

간결한 요약:""",
    description="간결성에 집중한 템플릿"
)

DETAILED_CONTEXT = PromptTemplate(
    name="detailed_context",
    template="""다음 대화의 요약문을 맥락과 흐름을 유지하며 상세하게 작성해주세요.
대화자들의 의도까지 잘 드러나도록 하세요.

대화:
{dialogue}

상세한 요약:""",
    description="상세한 요약문을 요구하는 템플릿"
)


# ==================== 템플릿 레지스트리 ==================== #
TEMPLATE_REGISTRY: Dict[str, PromptTemplate] = {
    # Zero-shot
    "zero_shot_simple": ZERO_SHOT_SIMPLE,
    "zero_shot_detailed": ZERO_SHOT_DETAILED,
    "zero_shot_structured": ZERO_SHOT_STRUCTURED,

    # Few-shot
    "few_shot_basic": FEW_SHOT_BASIC,
    "few_shot_extended": FEW_SHOT_EXTENDED,

    # Chain-of-Thought
    "cot_step_by_step": COT_STEP_BY_STEP,
    "cot_reasoning": COT_REASONING,

    # 역할 기반
    "role_professional": ROLE_PROFESSIONAL,
    "role_assistant": ROLE_ASSISTANT,

    # 특수 목적
    "korean_focused": KOREAN_FOCUSED,
    "concise_focus": CONCISE_FOCUS,
    "detailed_context": DETAILED_CONTEXT,
}


# ==================== 템플릿 접근 함수 ==================== #
def get_template(name: str) -> PromptTemplate:
    """
    템플릿 이름으로 템플릿 가져오기

    Args:
        name: 템플릿 이름

    Returns:
        PromptTemplate 객체

    Raises:
        KeyError: 이름에 해당하는 템플릿 없음
    """
    if name not in TEMPLATE_REGISTRY:
        available = ", ".join(TEMPLATE_REGISTRY.keys())
        raise KeyError(
            f"템플릿 '{name}'을 찾을 수 없습니다. "
            f"사용 가능한 템플릿: {available}"
        )

    return TEMPLATE_REGISTRY[name]


def list_templates() -> List[str]:
    """
    사용 가능한 모든 템플릿 이름 목록

    Returns:
        템플릿 이름 리스트
    """
    return list(TEMPLATE_REGISTRY.keys())


def get_templates_by_category(category: str) -> Dict[str, PromptTemplate]:
    """
    카테고리별로 템플릿 가져오기

    Args:
        category: 카테고리명 ("zero_shot", "few_shot", "cot", "role", "special")

    Returns:
        카테고리에 속한 템플릿 딕셔너리
    """
    category_prefixes = {
        "zero_shot": "zero_shot",
        "few_shot": "few_shot",
        "cot": "cot",
        "role": "role",
        "special": ["korean", "concise", "detailed"]
    }

    if category not in category_prefixes:
        raise ValueError(f"지원하지 않는 카테고리: {category}")

    prefix = category_prefixes[category]

    if isinstance(prefix, list):
        # 복수 카테고리 (여러 prefix)
        return {
            name: template
            for name, template in TEMPLATE_REGISTRY.items()
            if any(name.startswith(p) for p in prefix)
        }
    else:
        # 단일 prefix
        return {
            name: template
            for name, template in TEMPLATE_REGISTRY.items()
            if name.startswith(prefix)
        }
