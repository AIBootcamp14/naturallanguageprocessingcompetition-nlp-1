"""
Solar API 통합

PRD 09: Solar API 최적화 전략 구현
- Few-shot Learning
- 토큰 최적화
- 배치 처리
- 캐싱
"""

import os
import re
import time
import hashlib
import pickle
from typing import List, Dict, Optional
from pathlib import Path


class SolarAPI:
    """Solar API 클라이언트 (토큰 최적화)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        token_limit: int = 512,
        cache_dir: str = "cache/solar",
        logger=None
    ):
        """
        Args:
            api_key: Solar API 키 (None이면 환경 변수에서 읽음)
            token_limit: 대화당 최대 토큰 수
            cache_dir: 캐시 디렉토리
            logger: Logger 인스턴스
        """
        self.api_key = api_key or os.getenv("SOLAR_API_KEY")
        self.token_limit = token_limit
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        # 캐시 로드
        self.cache = self._load_cache()

        # OpenAI 클라이언트 초기화 (Solar API 호환)
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.upstage.ai/v1/solar"
                )
                self._log("Solar API 클라이언트 초기화 성공")
            except ImportError:
                self._log("⚠️  OpenAI 라이브러리가 설치되지 않음 (pip install openai)")
        else:
            self._log("⚠️  Solar API 키가 설정되지 않음")

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def _load_cache(self) -> Dict:
        """캐시 로드"""
        cache_file = self.cache_dir / "solar_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                self._log(f"캐시 로드 완료: {len(cache)}개 항목")
                return cache
            except:
                return {}
        return {}

    def _save_cache(self):
        """캐시 저장"""
        cache_file = self.cache_dir / "solar_cache.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def preprocess_dialogue(self, dialogue: str) -> str:
        """
        대화 전처리 (토큰 절약)

        Args:
            dialogue: 원본 대화

        Returns:
            전처리된 대화
        """
        # 1. 불필요한 공백 제거
        dialogue = ' '.join(dialogue.split())

        # 2. Person 태그 간소화
        dialogue = dialogue.replace('#Person1#:', 'A:')
        dialogue = dialogue.replace('#Person2#:', 'B:')
        dialogue = dialogue.replace('#Person3#:', 'C:')
        dialogue = dialogue.replace('#Person4#:', 'D:')

        # 3. 스마트 절단
        dialogue = self.smart_truncate(dialogue, self.token_limit)

        return dialogue

    def smart_truncate(self, text: str, max_tokens: int = 512) -> str:
        """
        스마트 절단 (문장 단위)

        Args:
            text: 원본 텍스트
            max_tokens: 최대 토큰 수

        Returns:
            절단된 텍스트
        """
        # 토큰 수 추정 (한글 평균 2.5자 = 1토큰)
        estimated_tokens = self.estimate_tokens(text)

        if estimated_tokens <= max_tokens:
            return text

        # 문장 단위로 자르기
        sentences = re.split(r'([.!?]\s+)', text)
        truncated = []
        current_tokens = 0

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]

            sentence_tokens = self.estimate_tokens(sentence)

            if current_tokens + sentence_tokens > max_tokens:
                break

            truncated.append(sentence)
            current_tokens += sentence_tokens

        result = ''.join(truncated)
        return result if result else text[:int(max_tokens * 2.5)]

    def estimate_tokens(self, text: str) -> int:
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

    def build_few_shot_prompt(
        self,
        dialogue: str,
        example_dialogue: Optional[str] = None,
        example_summary: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Few-shot 프롬프트 생성

        Args:
            dialogue: 입력 대화
            example_dialogue: 예시 대화
            example_summary: 예시 요약

        Returns:
            메시지 리스트
        """
        system_prompt = "You are an expert in the field of dialogue summarization. Summarize the given dialogue in a concise manner in Korean."

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Few-shot 예시 추가
        if example_dialogue and example_summary:
            messages.append({
                "role": "user",
                "content": f"Dialogue:\n{example_dialogue}\nSummary:"
            })
            messages.append({
                "role": "assistant",
                "content": example_summary
            })

        # 실제 입력
        messages.append({
            "role": "user",
            "content": f"Dialogue:\n{dialogue}\nSummary:"
        })

        return messages

    def summarize(
        self,
        dialogue: str,
        example_dialogue: Optional[str] = None,
        example_summary: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.3
    ) -> str:
        """
        단일 대화 요약

        Args:
            dialogue: 입력 대화
            example_dialogue: Few-shot 예시 대화
            example_summary: Few-shot 예시 요약
            temperature: Temperature (낮을수록 일관성)
            top_p: Top-p (낮을수록 일관성)

        Returns:
            요약 결과
        """
        if not self.client:
            raise RuntimeError("Solar API 클라이언트가 초기화되지 않음")

        # 캐시 확인
        cache_key = hashlib.md5(dialogue.encode()).hexdigest()
        if cache_key in self.cache:
            self._log(f"캐시 히트: {cache_key[:8]}")
            return self.cache[cache_key]

        # 전처리
        processed_dialogue = self.preprocess_dialogue(dialogue)

        # 프롬프트 생성
        messages = self.build_few_shot_prompt(
            processed_dialogue,
            example_dialogue,
            example_summary
        )

        # API 호출
        try:
            response = self.client.chat.completions.create(
                model="solar-1-mini-chat",
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=200
            )

            summary = response.choices[0].message.content.strip()

            # 캐시 저장
            self.cache[cache_key] = summary
            self._save_cache()

            return summary

        except Exception as e:
            self._log(f"❌ API 호출 실패: {str(e)}")
            return ""

    def summarize_batch(
        self,
        dialogues: List[str],
        example_dialogue: Optional[str] = None,
        example_summary: Optional[str] = None,
        batch_size: int = 10,
        delay: float = 1.0
    ) -> List[str]:
        """
        배치 요약 (Rate limit 고려)

        Args:
            dialogues: 대화 리스트
            example_dialogue: Few-shot 예시 대화
            example_summary: Few-shot 예시 요약
            batch_size: 배치 크기
            delay: 배치 간 대기 시간 (초)

        Returns:
            요약 리스트
        """
        summaries = []

        self._log(f"\n배치 요약 시작: {len(dialogues)}개 대화")
        self._log(f"  - 배치 크기: {batch_size}")
        self._log(f"  - Rate limit 대기: {delay}초")

        for i in range(0, len(dialogues), batch_size):
            batch = dialogues[i:i + batch_size]

            self._log(f"\n배치 {i//batch_size + 1}/{(len(dialogues)-1)//batch_size + 1} 처리 중...")

            batch_summaries = []
            for dialogue in batch:
                summary = self.summarize(
                    dialogue,
                    example_dialogue,
                    example_summary
                )
                batch_summaries.append(summary)

            summaries.extend(batch_summaries)

            # Rate limiting
            if i + batch_size < len(dialogues):
                time.sleep(delay)

        self._log(f"\n배치 요약 완료: {len(summaries)}개")

        return summaries


def create_solar_api(
    api_key: Optional[str] = None,
    token_limit: int = 512,
    cache_dir: str = "cache/solar",
    logger=None
) -> SolarAPI:
    """
    편의 함수: Solar API 생성

    Args:
        api_key: Solar API 키
        token_limit: 토큰 제한
        cache_dir: 캐시 디렉토리
        logger: Logger 인스턴스

    Returns:
        SolarAPI 인스턴스
    """
    return SolarAPI(api_key, token_limit, cache_dir, logger)
