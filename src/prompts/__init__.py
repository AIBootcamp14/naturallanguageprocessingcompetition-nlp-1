"""
프롬프트 엔지니어링 시스템

PRD 10: 프롬프트 엔지니어링 전략 구현
"""

from .templates import (
    PromptTemplate,
    get_template,
    list_templates,
    get_templates_by_category,
    TEMPLATE_REGISTRY
)

from .prompt_manager import (
    PromptManager,
    create_prompt_manager,
    quick_format
)

__all__ = [
    # 템플릿
    'PromptTemplate',
    'get_template',
    'list_templates',
    'get_templates_by_category',
    'TEMPLATE_REGISTRY',

    # 매니저
    'PromptManager',
    'create_prompt_manager',
    'quick_format',
]
