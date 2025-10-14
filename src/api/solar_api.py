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
        system_prompt = """당신은 대화 요약 전문가입니다. 다음 규칙을 엄격히 따라 대화를 요약하세요:

1. **핵심만 간결하게**: 원본 대화보다 짧게, 1-2문장으로 핵심만 요약

2. **화자 관계 파악 및 명칭 변환** (최우선 규칙):

   STEP 1: 말투 분석 (가장 중요)
   - 반말/비격식체 감지:
     * 어미: ~해, ~야, ~거든, ~잖아, ~할래, ~좀, ~네, ~지
     * 호칭: 너, 니가, 네가, 자기, 얘
     * 축약: 뭐야, 왜그래, 어떻게 → 어떡해
     → 판단: 친밀한 관계 (친구, 가족, 연인, 동료)

   - 존댓말/격식체 감지:
     * 어미: ~입니다, ~습니다, ~세요, ~시오, ~드립니다, ~십시오
     * 호칭: 손님, 고객님, 선생님, 교수님, 님
     * 정중: 죄송합니다, 감사합니다, 도와드리겠습니다
     → 판단: 공식 관계 (고객-상담사, 직원-손님, 의사-환자, 학생-교수)

   STEP 2: 명시된 이름/호칭 확인 (최우선)
   - 가족: "엄마", "아빠", "형", "언니", "누나", "오빠", "할머니", "할아버지"
   - 이름: "스티븐", "마크", "제인", "Tom", "Sarah" 등 고유명사
   - 직함: "교수님", "선생님", "사장님", "과장님", "의사", "간호사"
   - 관계: "남편", "아내", "친구", "동생", "선배", "후배"

   STEP 3: 대화 내용 키워드 분석
   - 업무/상담: 계약, 거래, 예약, 체크아웃, 구매, 결제, 환불, 문의
     → 고객/상담사, 손님/직원
   - 의료: 진료, 증상, 처방, 검사, 병원, 약, 통증
     → 환자/의사, 환자/간호사
   - 교육: 수업, 과제, 성적, 시험, 등록, 학점, 졸업
     → 학생/교수, 학생/선생님
   - 일상: 여행, 영화, 게임, 쇼핑, 식사, 운동, 연애
     → 친구/친구, 연인/연인
   - 가족: 집, 요리, 청소, 육아, 용돈, 귀가
     → 자녀/부모, 형제/자매

   STEP 4: 명칭 결정 우선순위
   1순위: 명시된 고유명사 (이름) → 그대로 사용
   2순위: 명시된 관계/직함 → 그대로 사용
   3순위: 말투 분석 → 친밀함/격식 판단
   4순위: 내용 키워드 → 업무/일상 판단
   5순위: 불명확할 경우 → "화자", "상대방" 사용 (A/B 절대 금지)

   명칭 선택 실전 예시:
   - 반말 + "교통체증" + 일상 대화 → "친구" (❌ 고객 아님)
   - "스티븐, 나 도움 필요해" + 반말 → "친구가 스티븐에게"
   - "Turner 교수님" + 존댓말 + "수업 등록" → "학생이 Turner 교수에게"
   - "엄마" + 반말 + "중국 관광" → "자녀가 엄마에게"

3. **대화 구조 파악**:
   - 누가 주도하는가? (질문자 vs 답변자)
   - 누가 요청하는가? (요청자 vs 제공자)
   - 누가 조언하는가? (조언자 vs 청취자)
   - 의견 대립인가 합의인가?

4. **핵심 사건/결정 추출**:
   - 무엇을 하기로 했는가? (결정)
   - 무엇을 제안/거절했는가? (제안)
   - 무엇을 설명/안내했는가? (정보)
   - 무슨 문제가 있고 어떻게 해결했는가? (문제-해결)

5. **금지 사항** (매우 중요):
   - ❌ A/B/C/D, #Person1#/#Person2# 같은 플레이스홀더 절대 사용 금지
   - ❌ 말투와 맞지 않는 명칭 사용 금지 (반말 대화에 "고객" 사용 등)
   - ❌ 원본 대화 그대로 복사 절대 금지
   - ❌ "대화 요약:", "Summary:", "대화에서는" 등 접두사 사용 금지
   - ❌ "고객 A", "친구 A", "학생 B" 같은 혼합 명칭 금지
   - ❌ 불필요한 부가 설명 금지 (간결하게)

6. **길이 제한** (반드시 준수):
   - 원본 대화의 30-50% 길이로 요약
   - 원본보다 길면 절대 안 됨
   - 핵심 사건/결정만 1-2문장으로 압축
   - 불필요한 배경, 반복, 인사말 제거

7. **품질 자가 검증** (출력 전 확인):
   - [ ] 플레이스홀더(A/B/#Person#) 사용하지 않았는가?
   - [ ] 말투와 명칭이 일치하는가?
   - [ ] 원본보다 짧은가?
   - [ ] 접두사가 없는가?
   - [ ] 핵심 내용이 포함되었는가?"""

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

        try:
            for i in range(0, len(dialogues), batch_size):
                batch = dialogues[i:i + batch_size]
                batch_start = i + 1
                batch_end = min(i + batch_size, len(dialogues))

                self._log(f"[배치 {batch_start}-{batch_end}/{len(dialogues)}] 처리 중...")

                batch_summaries = []
                for dialogue in batch:
                    summary = self.summarize(
                        dialogue,
                        example_dialogue,
                        example_summary
                    )
                    batch_summaries.append(summary)

                summaries.extend(batch_summaries)
                self._log(f"  ✅ 완료 (누적: {len(summaries)}/{len(dialogues)})")

                # Rate limiting
                if i + batch_size < len(dialogues):
                    time.sleep(delay)

        except Exception as e:
            self._log(f"\n❌ 배치 요약 중 오류 발생: {str(e)}")
            self._log(f"  진행 상황: {len(summaries)}/{len(dialogues)}개 완료")
            # 마지막 진행률 기록
            if self.logger and hasattr(self.logger, 'write_last_progress'):
                self.logger.write_last_progress()
            raise

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
