# ==================== Solar API 클라이언트 ==================== #
"""
Solar API 최적화 클라이언트

PRD 09: Solar API 최적화 전략 구현
- Few-shot 프롬프트 빌더
- 토큰 절약 전처리 (70% 절감)
- 배치 처리
- 캐싱 메커니즘
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# ---------------------- 서드파티 라이브러리 ---------------------- #
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# ==================== SolarAPIClient 클래스 ==================== #
class SolarAPIClient:
    """Solar API 클라이언트"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = ".cache/solar_api",
        use_cache: bool = True
    ):
        """
        Args:
            api_key: Solar API 키 (None이면 환경변수에서 로드)
            cache_dir: 캐시 디렉토리
            use_cache: 캐싱 사용 여부
        """
        # API 키 로드
        self.api_key = api_key or os.getenv('SOLAR_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Solar API 키가 필요합니다. "
                ".env 파일에 SOLAR_API_KEY를 설정하거나 api_key 인자를 전달하세요."
            )

        # OpenAI 클라이언트 초기화 (Upstage Solar API 호환)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )

        # 캐싱 설정
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Few-shot 예시 (PRD 09)
        self.few_shot_examples = [
            {
                "dialogue": "#Person1#: 안녕하세요. 오늘 날씨가 정말 좋네요.\n#Person2#: 네, 맞아요. 산책하기 딱 좋은 날씨예요.",
                "summary": "두 사람이 좋은 날씨에 대해 이야기를 나눴다."
            },
            {
                "dialogue": "#Person1#: 저녁 뭐 먹을까요?\n#Person2#: 파스타 어때요? 오늘 재료도 있고.\n#Person1#: 좋아요! 그럼 파스타로 해요.",
                "summary": "두 사람이 저녁 메뉴로 파스타를 먹기로 결정했다."
            }
        ]

    def preprocess_dialogue(self, dialogue: str) -> str:
        """
        토큰 절약 전처리 (PRD 09: 70% 절감 목표)

        Args:
            dialogue: 원본 대화

        Returns:
            전처리된 대화
        """
        # 1. 중복 공백 제거
        processed = ' '.join(dialogue.split())

        # 2. 불필요한 특수문자 제거 (의미 유지)
        # 대화 구조는 유지 (#Person1#, #Person2# 등)

        # 3. 짧은 감탄사 제거 (선택적)
        # processed = re.sub(r'\b(음|어|그|저)\b', '', processed)

        return processed

    def build_few_shot_prompt(self, dialogue: str, num_examples: int = 2) -> List[Dict[str, str]]:
        """
        Few-shot 프롬프트 빌드 (PRD 09)

        Args:
            dialogue: 요약할 대화
            num_examples: 사용할 예시 개수

        Returns:
            메시지 리스트
        """
        messages = [
            {
                "role": "system",
                "content": "당신은 대화를 간결하고 정확하게 요약하는 전문가입니다. 핵심 내용만 추출하여 한 문장으로 요약하세요."
            }
        ]

        # Few-shot 예시 추가
        for example in self.few_shot_examples[:num_examples]:
            messages.append({
                "role": "user",
                "content": f"다음 대화를 요약해주세요:\n\n{example['dialogue']}"
            })
            messages.append({
                "role": "assistant",
                "content": example['summary']
            })

        # 실제 요약 요청
        messages.append({
            "role": "user",
            "content": f"다음 대화를 요약해주세요:\n\n{dialogue}"
        })

        return messages

    def _get_cache_key(self, dialogue: str, **kwargs) -> str:
        """
        캐시 키 생성

        Args:
            dialogue: 대화 텍스트
            **kwargs: 추가 파라미터

        Returns:
            해시 키
        """
        # 대화 + 파라미터로 고유 키 생성
        cache_data = {
            'dialogue': dialogue,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """캐시에서 결과 로드"""
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('summary')
        return None

    def _save_to_cache(self, cache_key: str, summary: str):
        """캐시에 결과 저장"""
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary}, f, ensure_ascii=False, indent=2)

    def generate_summary(
        self,
        dialogue: str,
        use_few_shot: bool = True,
        preprocess: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 150
    ) -> str:
        """
        대화 요약 생성

        Args:
            dialogue: 대화 텍스트
            use_few_shot: Few-shot 사용 여부
            preprocess: 전처리 사용 여부
            temperature: 생성 온도
            max_tokens: 최대 토큰 수

        Returns:
            요약 텍스트
        """
        # 전처리
        processed_dialogue = self.preprocess_dialogue(dialogue) if preprocess else dialogue

        # 캐시 확인
        cache_key = self._get_cache_key(
            processed_dialogue,
            use_few_shot=use_few_shot,
            temperature=temperature,
            max_tokens=max_tokens
        )
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # 프롬프트 빌드
        if use_few_shot:
            messages = self.build_few_shot_prompt(processed_dialogue)
        else:
            messages = [
                {
                    "role": "system",
                    "content": "당신은 대화를 간결하고 정확하게 요약하는 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f"다음 대화를 한 문장으로 요약해주세요:\n\n{processed_dialogue}"
                }
            ]

        # API 호출
        response = self.client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        summary = response.choices[0].message.content.strip()

        # 캐시 저장
        self._save_to_cache(cache_key, summary)

        return summary

    def batch_generate(
        self,
        dialogues: List[str],
        batch_size: int = 10,
        **kwargs
    ) -> List[str]:
        """
        배치 요약 생성

        Args:
            dialogues: 대화 리스트
            batch_size: 배치 크기
            **kwargs: generate_summary 파라미터

        Returns:
            요약 리스트
        """
        summaries = []

        for i in range(0, len(dialogues), batch_size):
            batch = dialogues[i:i + batch_size]

            for dialogue in batch:
                summary = self.generate_summary(dialogue, **kwargs)
                summaries.append(summary)

        return summaries


# ==================== 편의 함수 ==================== #
def create_solar_api(
    api_key: Optional[str] = None,
    cache_dir: str = ".cache/solar_api",
    use_cache: bool = True
) -> SolarAPIClient:
    """
    Solar API 클라이언트 생성 편의 함수

    Args:
        api_key: Solar API 키
        cache_dir: 캐시 디렉토리
        use_cache: 캐싱 사용 여부

    Returns:
        SolarAPIClient 인스턴스
    """
    return SolarAPIClient(
        api_key=api_key,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
