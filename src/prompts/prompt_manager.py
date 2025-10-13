# ==================== 프롬프트 관리자 ==================== #
"""
프롬프트 엔지니어링 관리자

PRD 10: 프롬프트 엔지니어링 모듈
- 템플릿 관리
- 동적 프롬프트 생성
- A/B 테스트 지원
"""

# ---------------------- 외부 라이브러리 ---------------------- #
from typing import List, Dict, Optional, Any
from pathlib import Path
import json

# ---------------------- 내부 모듈 ---------------------- #
from .templates import (
    PromptTemplate,
    TEMPLATE_REGISTRY,
    get_template,
    list_templates,
    get_templates_by_category
)


# ==================== PromptManager 클래스 ==================== #
class PromptManager:
    """프롬프트 관리자 - 템플릿 관리 및 프롬프트 생성"""

    def __init__(
        self,
        default_template: str = "zero_shot_simple",
        logger=None
    ):
        """
        Args:
            default_template: 기본 템플릿 이름
            logger: Logger 인스턴스
        """
        self.default_template = default_template
        self.logger = logger

        # 현재 템플릿
        self.current_template = get_template(default_template)

        # 프롬프트 히스토리 (A/B 테스트)
        self.prompt_history: List[Dict[str, Any]] = []

        self._log(f"PromptManager 초기화 (템플릿: {default_template})")

    def _log(self, msg: str):
        """로그 출력"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def set_template(self, template_name: str):
        """
        템플릿 변경

        Args:
            template_name: 새로운 템플릿 이름
        """
        self.current_template = get_template(template_name)
        self._log(f"템플릿 변경: {template_name}")

    def get_current_template(self) -> PromptTemplate:
        """
        현재 템플릿 가져오기

        Returns:
            현재 PromptTemplate 객체
        """
        return self.current_template

    def format_prompt(
        self,
        dialogue: str,
        template_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        프롬프트 생성

        Args:
            dialogue: 대화 문자열
            template_name: 사용할 템플릿 이름 (None이면 현재 템플릿)
            **kwargs: 추가 인자

        Returns:
            완성된 프롬프트
        """
        # 템플릿 선택
        if template_name is not None:
            template = get_template(template_name)
        else:
            template = self.current_template

        # 프롬프트 생성
        prompt = template.template.format(dialogue=dialogue, **kwargs)

        # 히스토리 기록
        self.prompt_history.append({
            'template_name': template.name,
            'dialogue': dialogue,
            'prompt': prompt,
            **kwargs
        })

        return prompt

    def batch_format(
        self,
        dialogues: List[str],
        template_name: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        배치 프롬프트 생성

        Args:
            dialogues: 대화 목록
            template_name: 템플릿 이름
            **kwargs: 추가 인자

        Returns:
            완성된 프롬프트 목록
        """
        prompts = []

        for dialogue in dialogues:
            prompt = self.format_prompt(
                dialogue=dialogue,
                template_name=template_name,
                **kwargs
            )
            prompts.append(prompt)

        return prompts

    def create_custom_template(
        self,
        name: str,
        template: str,
        description: str = ""
    ) -> PromptTemplate:
        """
        커스텀 템플릿 생성

        Args:
            name: 템플릿 이름
            template: 템플릿 문자열 ({dialogue} 플레이스홀더 포함)
            description: 설명

        Returns:
            생성된 PromptTemplate 객체
        """
        custom_template = PromptTemplate(
            name=name,
            template=template,
            description=description
        )

        # 레지스트리에 추가
        TEMPLATE_REGISTRY[name] = custom_template

        self._log(f"커스텀 템플릿 생성: {name}")

        return custom_template

    def compare_templates(
        self,
        dialogue: str,
        template_names: List[str]
    ) -> Dict[str, str]:
        """
        여러 템플릿으로 프롬프트 생성하여 비교

        Args:
            dialogue: 대화 문자열
            template_names: 비교할 템플릿 이름 목록

        Returns:
            템플릿별 프롬프트 딕셔너리
        """
        results = {}

        for template_name in template_names:
            prompt = self.format_prompt(
                dialogue=dialogue,
                template_name=template_name
            )
            results[template_name] = prompt

        return results

    def get_template_info(self, template_name: Optional[str] = None) -> Dict[str, Any]:
        """
        템플릿 정보 조회

        Args:
            template_name: 템플릿 이름 (None이면 현재 템플릿)

        Returns:
            템플릿 정보 딕셔너리
        """
        if template_name is not None:
            template = get_template(template_name)
        else:
            template = self.current_template

        return {
            'name': template.name,
            'description': template.description,
            'template': template.template,
            'example_input': template.example_input,
            'example_output': template.example_output
        }

    def list_available_templates(self, category: Optional[str] = None) -> List[str]:
        """
        사용 가능한 템플릿 목록 조회

        Args:
            category: 카테고리 필터링 (None이면 전체)

        Returns:
            템플릿 이름 목록
        """
        if category is None:
            return list_templates()
        else:
            templates = get_templates_by_category(category)
            return list(templates.keys())

    def save_history(self, output_path: str):
        """
        프롬프트 히스토리 저장

        Args:
            output_path: 저장 경로
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.prompt_history, f, indent=2, ensure_ascii=False)

        self._log(f"프롬프트 히스토리 저장: {output_path}")

    def clear_history(self):
        """히스토리 초기화"""
        self.prompt_history = []
        self._log("프롬프트 히스토리 초기화")

    def get_statistics(self) -> Dict[str, Any]:
        """
        사용 통계 조회

        Returns:
            통계 정보 딕셔너리
        """
        if not self.prompt_history:
            return {'total': 0}

        # 템플릿별 사용 횟수
        template_counts = {}
        for entry in self.prompt_history:
            template_name = entry['template_name']
            template_counts[template_name] = template_counts.get(template_name, 0) + 1

        return {
            'total': len(self.prompt_history),
            'template_counts': template_counts,
            'most_used_template': max(template_counts, key=template_counts.get)
        }


# ==================== 편의 함수 ==================== #
def create_prompt_manager(
    default_template: str = "zero_shot_simple",
    logger=None
) -> PromptManager:
    """
    PromptManager 생성 편의 함수

    Args:
        default_template: 기본 템플릿
        logger: Logger 인스턴스

    Returns:
        PromptManager 인스턴스
    """
    return PromptManager(
        default_template=default_template,
        logger=logger
    )


def quick_format(
    dialogue: str,
    template_name: str = "zero_shot_simple"
) -> str:
    """
    빠른 프롬프트 생성 (단일 사용)

    Args:
        dialogue: 대화 문자열
        template_name: 템플릿 이름

    Returns:
        완성된 프롬프트
    """
    template = get_template(template_name)
    return template.template.format(dialogue=dialogue)
