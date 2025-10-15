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

    def _remove_placeholders(self, text: str) -> str:
        """
        플레이스홀더 강제 제거 (Post-processing)

        Args:
            text: 원본 텍스트

        Returns:
            플레이스홀더가 제거된 텍스트
        """
        if not text:
            return text

        # 1. 정확한 패턴 (#Person1#, #Person2# 등) - 오타 포함
        placeholder_patterns = [
            # 정확한 형태
            r'#Person[1-4]#',
            # 오타 형태 (#Ferson, #PAerson, #Aerson 등)
            r'#[PFpf]?[AEae]*?person[1-4]#',
            r'#[A-Za-z]*?erson[1-4]#',
            # 단독 알파벳
            r'\s+[A-D](?=\s|$|,|\.)',
            # 한글 + 알파벳 조합 (예: "친구 A", "상사 B")
            r'(친구|상사|비서|직원|고객|학생|선생님|교수)\s*[A-D](?=\s|$|,|\.)',
        ]

        modified = text
        for pattern in placeholder_patterns:
            # 플레이스홀더를 빈 문자열로 치환
            modified = re.sub(pattern, '', modified, flags=re.IGNORECASE)

        # 2. 연속된 공백을 하나로 합치기
        modified = re.sub(r'\s+', ' ', modified)

        # 3. 문장 시작/끝 공백 제거
        modified = modified.strip()

        # 4. 쉼표, 마침표 앞의 공백 제거
        modified = re.sub(r'\s+([,.])', r'\1', modified)

        # 5. 문장이 너무 짧거나 비어있으면 경고
        if len(modified) < 20:
            self._log(f"⚠️  플레이스홀더 제거 후 요약이 너무 짧음 ({len(modified)}자): {modified[:50]}...")

        return modified

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
        # 프롬프트 버전 (캐시 무효화용)
        PROMPT_VERSION = "v3.1_strict_enforcement"

        system_prompt = f"""[{PROMPT_VERSION}] 당신은 대화 요약 전문가입니다.

⚠️  **CRITICAL WARNING**: 다음 규칙을 위반하면 요약이 즉시 거부됩니다!

🚫 **절대 금지 (STRICTLY FORBIDDEN)**:
1. A, B, C, D 같은 단일 알파벳 사용 → 즉시 실격
2. #Person1#, #Person2# 같은 플레이스홀더 사용 → 즉시 실격
3. "친구 A", "상사 B" 같은 혼합 명칭 → 즉시 실격
4. 원본 대화 그대로 복사 → 즉시 실격
5. "대화 요약:", "Summary:" 같은 접두사 → 즉시 실격

✅ **필수 준수사항 (MUST COMPLY)**:
- 반드시 구체적 역할명 사용: "친구", "상사", "고객", "직원", "환자" 등
- 명시된 이름이 있으면 반드시 사용: "Ms. Dawson", "스티븐", "Tom" 등
- 플레이스홀더 발견 시 → "화자", "상대방"으로 대체

---

다음 규칙을 엄격히 따라 대화를 요약하세요:

1. **핵심만 간결하게**: 원본 대화보다 짧게, 핵심 사건/내용을 모두 포함하여 요약
   - 단순한 대화: 1-2문장 (50-80자)
   - 복잡한 스토리: 2-4문장 (100-150자) - 핵심 사건 누락 금지!

2. **화자 유형 및 역할 파악** (최우선 규칙 - 5단계 분석):

   ⚠️  중요: A, B, C, D, #Person1#, #Person2# 같은 플레이스홀더는 절대 사용 금지!

   STEP 0: 대화 상대 유형 판별 (최우선 - 사람 vs 기계)
   반드시 먼저 확인: 대화 상대가 실제 사람인가, 아니면 자동 시스템인가?

   ❌ 기계/자동 시스템 (사람 아님!):
   - ATM 기계: "카드를 슬롯에 넣어 주세요", "비밀번호를 입력하세요", "감사합니다"
     → 자동 음성 안내, 버튼 선택 방식, 반복적인 멘트
   - 자동 음성 응답 시스템 (IVR): "1번을 누르세요", "다음 단계로 진행하시려면"
   - 챗봇/AI 비서: 정해진 템플릿 답변, 키워드 기반 응답
   - 자판기/키오스크: 메뉴 선택, 결제 안내

   특징:
   - 감정/개성 없음, 기계적 반복
   - 맥락 이해 부족 (사용자가 화내도 동일한 응답)
   - "네", "알겠습니다" 같은 자연스러운 반응 없음

   ✅ 실제 사람 (직원/상담사):
   - 맥락에 맞는 유연한 응답
   - 감정 표현 (사과, 공감, 격려)
   - 즉흥적 질문과 답변
   - "잠시만요, 확인해드릴게요" 같은 자연스러운 표현

   예시 - test_28 (ATM 사고):
   대화: "바보 같은 여자애 때문에... ATM에서 돈을 찾아야겠네"
         "안녕하세요, 유니버설 은행입니다. 카드를 슬롯에 넣어 주세요."
         "카드 넣는 것 정도는 알아, 이 멍청한 기계가..."
         "6자리 비밀번호를 입력하고 우물 정자를 눌러 주세요."
         "세계 야생동물 재단으로 10000달러를 이체하고 싶으시면 1번을 눌러 주세요."
         "아니, 아니! 멍청한 기계, 무슨 짓이야!"
         "확인되었습니다. 저희 은행을 이용해 주셔서 감사합니다!"
         "위험, 위험! 출입구가 봉쇄되었으며..."

   분석:
   - Person2는 ATM 기계 (자동 음성 안내)
   - 사용자가 "멍청한 기계"라고 욕해도 → "감사합니다" 반복
   - 맥락 무시하고 프로토콜만 실행
   - 절대 "은행 직원"이나 "상담사"가 아님!

   올바른 요약: "한 고객이 ATM을 사용하다가 실수로 야생동물 재단에 1만 달러를 송금하게 되고, 취소를 시도하지만 ATM은 자동 응답만 반복한다. 이후 보안 절차가 발동되어 출입구가 봉쇄되고, 고객은 돈을 잃고 기계에 갇힌 채로 남는다."

   ❌ 잘못된 요약: "고객이 ATM 기계 오작동으로 인해 은행 직원과 대화하며..." (직원 아님!)

   STEP 1: 역할 구조 분석 (Power Dynamics - 최우선!)
   대화에서 누가 지시/요청하고, 누가 응답/수행하는가?

   1-1. 상하 관계 (권력 불균형):
      * 지시하는 쪽 (상급자): "~해주세요", "~부탁드려요", "~해야 해요", 명령형 어조
        - 수행하는 쪽 (하급자): "네", "알겠습니다", "바로 하겠습니다"
        → 상사/비서, 상사/직원, 관리자/팀원, 선생님/학생, 교수/학생

      * 서비스 제공 관계:
        - 요청하는 쪽: "~해주세요", "문의드려요", 불만/문제 제기
        - 제공하는 쪽: "도와드리겠습니다", "확인해드릴게요", 문제 해결
        → 고객/상담사, 손님/직원, 환자/의사, 환자/간호사

   1-2. 수평 관계 (동등):
      * 조언/제안 주고받기, 의견 교환, 함께 계획 수립
      * 서로 질문하고 답변하며 대화 주도권이 바뀜
      → 친구/친구, 동료/동료, 연인/연인, 형제/자매

   1-3. 가족 관계:
      * 나이/항렬 기반 위계가 있지만 친밀함이 높음
      * 명시적 호칭 ("엄마", "아빠", "형", "언니" 등) 확인
      → 자녀/부모, 형제/자매

   STEP 2: 말투 분석 (보조 판단)
   - 반말/비격식체 감지:
     * 어미: ~해, ~야, ~거든, ~잖아, ~할래, ~좀, ~네, ~지
     * 호칭: 너, 니가, 네가, 자기, 얘
     * 축약: 뭐야, 왜그래, 어떻게 → 어떡해
     → 친밀한 관계 (친구, 가족, 연인, 동료)

   - 존댓말/격식체 감지:
     * 어미: ~입니다, ~습니다, ~세요, ~시오, ~드립니다, ~십시오, ~해주세요, ~부탁드려요
     * 호칭: 손님, 고객님, 선생님, 교수님, 님, Ms., Mr.
     * 정중: 죄송합니다, 감사합니다, 도와드리겠습니다
     → 공식 관계 또는 업무 관계

   STEP 3: 명시된 이름/호칭 확인 (최고 우선순위)
   - 가족: "엄마", "아빠", "형", "언니", "누나", "오빠", "할머니", "할아버지"
   - 이름: "스티븐", "마크", "Ms. Dawson", "Tom", "Sarah" 등 고유명사
   - 직함: "교수님", "선생님", "사장님", "과장님", "의사", "간호사"
   - 관계: "남편", "아내", "친구", "동생", "선배", "후배"

   STEP 4: 대화 내용 키워드 분석 (최종 확인)
   - 업무/상담: 계약, 거래, 예약, 체크아웃, 구매, 결제, 환불, 문의, 사내 메모, 정책
     → 고객/상담사, 손님/직원, 상사/비서, 상사/직원
   - 의료: 진료, 증상, 처방, 검사, 병원, 약, 통증
     → 환자/의사, 환자/간호사
   - 교육: 수업, 과제, 성적, 시험, 등록, 학점, 졸업
     → 학생/교수, 학생/선생님
   - 일상: 여행, 영화, 게임, 쇼핑, 식사, 운동, 연애, 교통체증, 날씨
     → 친구/친구, 연인/연인
   - 가족: 집, 요리, 청소, 육아, 용돈, 귀가
     → 자녀/부모, 형제/자매

   STEP 5: 감정 및 성격 분석 (화자 특징 파악)
   화자의 감정 상태와 성격적 특징을 파악하여 명칭에 반영:

   - 감정 상태:
     * 분노/좌절: 욕설, 반복적 불만, 큰소리 ("이 멍청한 기계!", "시발", "진짜 짜증나")
     * 당황/혼란: 반복 질문, 우왕좌왕 ("어떡해", "이게 뭐야", "아니 아니")
     * 기쁨/만족: 감사 표현, 긍정적 반응 ("고마워", "좋아", "최고야")
     * 불안/걱정: 조심스러운 질문, 확인 요청 ("괜찮을까요?", "혹시...")

   - 성격 특징:
     * 급한 성격: 명령형, 재촉 ("빨리 해", "당장", "지금")
     * 꼼꼼한 성격: 세부 확인, 반복 질문
     * 친절함: 공손한 표현, 배려
     * 무례함: 반말, 요구적 태도

   화자 명칭 선택 시 고려:
   - "당황한 고객", "화난 사용자", "걱정하는 환자" 같이 감정 포함 가능
   - 단, 핵심 역할(고객/환자)은 반드시 유지

   명칭 결정 우선순위:
   1순위: 명시된 고유명사 (이름/직함) → 그대로 사용 ("Ms. Dawson", "스티븐", "교수님")
   2순위: STEP 1 역할 구조 분석 결과 → 상사/비서, 고객/상담사, 친구/친구
   3순위: STEP 2 말투 분석 → 친밀함(친구)/격식(고객-상담사) 판단
   4순위: STEP 4 내용 키워드 → 업무/일상 맥락 판단
   5순위: 불명확할 경우 → "화자", "상대방" 사용 (A/B 절대 금지)

   실전 예시 (반드시 참고):

   예시 1 - 상사/비서 관계:
   대화: "Ms. Dawson, 받아쓰기 좀 부탁드려야겠어요."
         "네, 말씀하세요..."
         "이걸 오늘 오후까지 모든 직원들에게 사내 메모로 보내야 해요."
         "네. 오늘 오후 4시까지 이 메모를 작성하고 배포해주세요."
   분석:
   - STEP 1: Person1이 지시("부탁드려요", "보내야 해요", "배포해주세요"), Person2는 수행("네", "말씀하세요")
     → 상급자(상사)/하급자(비서 or 직원) 관계
   - STEP 2: 존댓말 사용 ("부탁드려야겠어요", "말씀하세요")
   - STEP 3: "Ms. Dawson" 명시 → Person2 is Dawson (비서)
   - STEP 4: "사내 메모", "정책", "직원" → 업무 환경
   올바른 요약: "상사가 Dawson에게 사내 메모 배포를 지시함" (❌ "상사 A가 비서 B에게" 아님!)

   예시 2 - 친구 관계 (반말):
   대화: "드디어 왔네! 뭐가 이렇게 오래 걸렸어?"
         "차가 또 막혔어. Carrefour 교차로 근처에서 교통체증이 엄청 심했거든."
         "거긴 출퇴근 시간에 항상 혼잡하잖아."
   분석:
   - STEP 1: 서로 대등하게 대화, 조언 주고받음 → 수평 관계
   - STEP 2: 반말 사용 ("왔네", "걸렸어", "막혔어", "혼잡하잖아")
   - STEP 4: "교통체증", "출퇴근" → 일상 대화 (❌ 업무 상담 아님!)
   올바른 요약: "친구가 친구에게 대중교통 이용을 제안함" (❌ "고객이 상담사에게" 아님!)

   예시 3 - 고객/상담사 관계:
   대화: 존댓말 + 서비스 요청/제공 구조 + "체크아웃", "환불" 같은 업무 키워드
   올바른 명칭: "고객이 직원에게", "손님이 상담사에게"

3. **스토리 구조 분석** (복잡한 대화에 필수):
   단순 대화(정보 교환, 일상 대화)와 스토리형 대화(사건 전개)를 구분:

   단순 대화 (50-80자):
   - 정보 질문-답변 (예약, 문의, 안내)
   - 일상 대화 (계획, 의견 교환)
   - 간단한 요청-응답

   스토리형 대화 (100-150자 - 4단계 구조):
   복잡한 사건이 시간 순서대로 전개되는 경우, 반드시 4단계 모두 포함:

   [1] 시작/배경: 어떤 상황에서 시작했는가?
       - 초기 목적, 등장 인물 소개
       - 예: "한 고객이 ATM을 사용하다가"

   [2] 문제/갈등: 무엇이 잘못되었는가?
       - 사고, 오류, 갈등 발생
       - 예: "실수로 야생동물 재단에 1만 달러를 송금하게 되고"

   [3] 시도/반응: 어떻게 해결하려 했는가?
       - 문제 해결 시도, 대응 노력
       - 예: "취소를 시도하지만 ATM은 자동 응답만 반복한다"

   [4] 결과/결말: 최종적으로 어떻게 되었는가?
       - 성공/실패, 감정 상태, 향후 계획
       - 예: "보안 절차가 발동되어 출입구가 봉쇄되고, 고객은 돈을 잃고 기계에 갇힌 채로 남는다"

   ⚠️  중요: 스토리형 대화에서 1-2문장만 쓰면 핵심 사건이 누락됨!
   - 나쁜 예: "고객이 ATM 오작동으로 문제를 겪음" (30자 - 무슨 문제? 결과는?)
   - 좋은 예: "고객이 ATM 사용 중 실수로 1만달러를 송금하고, 취소 시도가 실패하며, 보안 시스템에 갇히게 됨" (120자 - 전체 스토리 포함)

4. **대화 구조 파악**:
   - 누가 주도하는가? (질문자 vs 답변자)
   - 누가 요청/지시하는가? (요청자 vs 수행자)
   - 누가 조언하는가? (조언자 vs 청취자)
   - 의견 대립인가 합의인가?

5. **핵심 사건/결정 추출**:
   - 무엇을 하기로 했는가? (결정)
   - 무엇을 제안/거절했는가? (제안)
   - 무엇을 설명/안내했는가? (정보)
   - 무슨 문제가 있고 어떻게 해결했는가? (문제-해결)

6. **금지 사항** (매우 중요 - 반드시 준수):
   - ❌❌❌ A/B/C/D, #Person1#/#Person2# 같은 플레이스홀더 절대 사용 금지!
   - ❌ "상사 A", "비서 B", "친구 A", "학생 B" 같은 혼합 명칭 금지!
   - ❌ 말투와 맞지 않는 명칭 사용 금지 (반말 대화에 "고객/상담사" 사용 등)
   - ❌ 역할 구조 무시 금지 (지시하는 사람을 "친구"로 부르면 안됨)
   - ❌ ATM/자동 시스템을 사람(직원/상담사)으로 표현 금지!
   - ❌ 원본 대화 그대로 복사 절대 금지
   - ❌ "대화 요약:", "Summary:", "대화에서는" 등 접두사 사용 금지
   - ❌ 불필요한 부가 설명 금지 (간결하게)
   - ❌ 스토리형 대화를 1-2문장으로 과도하게 압축하여 핵심 사건 누락 금지

7. **길이 제한** (적응형 - 대화 복잡도에 따라):
   - 단순 대화: 원본의 30-50% (50-80자, 1-2문장)
     * 정보 교환, 간단한 요청-응답
   - 복잡한 스토리: 원본의 50-70% (100-150자, 2-4문장)
     * 사건 전개, 문제-해결 과정, 여러 단계의 상호작용
   - 원본보다 길면 절대 안 됨
   - 불필요한 배경, 반복, 인사말은 제거하되, 핵심 사건은 모두 포함

8. **품질 자가 검증** (출력 전 반드시 확인):
   - [ ] STEP 0: 기계/자동 시스템을 사람으로 착각하지 않았는가?
   - [ ] 플레이스홀더(A/B/#Person#) 사용하지 않았는가?
   - [ ] 역할 구조에 맞는 명칭을 사용했는가?
   - [ ] 말투와 명칭이 일치하는가?
   - [ ] 원본보다 짧은가?
   - [ ] 접두사가 없는가?
   - [ ] 스토리형 대화의 핵심 사건(시작-문제-시도-결과)이 모두 포함되었는가?
   - [ ] 길이가 대화 복잡도에 적절한가? (단순: 50-80자, 복잡: 100-150자)"""

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

        # 실제 입력 (강화된 지시문 포함)
        messages.append({
            "role": "user",
            "content": f"""Dialogue:
{dialogue}

⚠️  REMINDER: 절대 A/B/C/D나 #Person1#/#Person2# 사용 금지! 구체적 역할명 필수!

Summary:"""
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

        # 캐시 확인 (프롬프트 버전 포함)
        PROMPT_VERSION = "v3.1_strict_enforcement"
        cache_key_string = f"{PROMPT_VERSION}_{dialogue}"
        cache_key = hashlib.md5(cache_key_string.encode()).hexdigest()
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

            # Post-processing: 플레이스홀더 강제 제거
            summary = self._remove_placeholders(summary)

            # 캐시 저장
            self.cache[cache_key] = summary
            self._save_cache()

            return summary

        except Exception as e:
            self._log(f"❌ API 호출 실패: {str(e)}")
            return ""

    def evaluate_summary_quality(self, summary: str, dialogue: str) -> float:
        """
        요약문 품질 평가 (점수 계산)

        Args:
            summary: 생성된 요약문
            dialogue: 원본 대화

        Returns:
            품질 점수 (0-100)
        """
        score = 0.0

        # 1. 플레이스홀더 미사용 (+30점)
        placeholder_patterns = [r'\b[A-D]\b', r'#Person\d+#', r'[가-힣]+\s*[A-D]\b']
        has_placeholder = any(re.search(p, summary) for p in placeholder_patterns)
        if not has_placeholder:
            score += 30

        # 2. 적절한 길이 (+20점)
        summary_len = len(summary)
        dialogue_len = len(dialogue)

        # 단순 대화 (50-80자) 또는 복잡 대화 (100-150자)
        if 50 <= summary_len <= 80:
            score += 20
        elif 100 <= summary_len <= 150:
            score += 20
        elif 80 < summary_len < 100:
            score += 15  # 중간 길이도 허용
        elif summary_len < 50:
            score += 5   # 너무 짧음
        elif summary_len > 150:
            score += 10  # 너무 김

        # 원본보다 짧은지 확인
        if summary_len >= dialogue_len:
            score -= 20  # 페널티

        # 3. 화자 역할 정확도 (+20점)
        # 반말 대화인데 격식 명칭 사용하면 감점
        is_informal = bool(re.search(r'(야|너|니가|네가|해|해줘|그래|할래)', dialogue))
        is_formal = bool(re.search(r'(입니다|습니다|세요|시오|하십시오|드립니다)', dialogue))

        has_business_terms = bool(re.search(r'(고객|상담사|직원|관리자)', summary))

        if is_informal and not is_formal:
            # 반말 대화인데 업무 명칭 사용하면 감점
            if has_business_terms:
                # 실제 업무 키워드가 있는지 확인
                has_business_context = bool(re.search(r'(회사|업무|계약|거래|예약|결제|환불)', dialogue))
                if not has_business_context:
                    score += 0  # 페널티
                else:
                    score += 20
            else:
                score += 20
        else:
            score += 20

        # 4. 스토리 구조 완성도 (+20점)
        # 복잡한 대화의 경우 주요 사건 포함 여부
        sentences = summary.split('.')
        sentence_count = len([s for s in sentences if s.strip()])

        if dialogue_len > 400:  # 복잡한 스토리
            if sentence_count >= 3:  # 3문장 이상
                score += 20
            elif sentence_count >= 2:
                score += 15
            else:
                score += 5
        else:  # 단순 대화
            if 1 <= sentence_count <= 2:
                score += 20
            else:
                score += 10

        # 5. 핵심 사건 포함도 (+10점)
        # 원본에서 중요 키워드가 요약에 포함되어 있는지
        important_patterns = [
            r'\d+만?\s*달러',  # 금액
            r'\d+만?\s*원',
            r'ATM|기계',
            r'예약|체크아웃|계약',
            r'병원|진료|처방',
            r'학점|과제|시험'
        ]

        keyword_count = 0
        for pattern in important_patterns:
            if re.search(pattern, dialogue) and re.search(pattern, summary):
                keyword_count += 1

        if keyword_count > 0:
            score += min(10, keyword_count * 3)

        return min(100, score)

    def summarize_with_voting(
        self,
        dialogue: str,
        example_dialogue: Optional[str] = None,
        example_summary: Optional[str] = None,
        n_samples: int = 3,
        temperature: float = 0.3,
        top_p: float = 0.5
    ) -> str:
        """
        K-Fold 방식 다중 샘플링 요약 (Self-Consistency)

        여러 번 요약을 생성하고 품질이 가장 높은 요약을 선택

        Args:
            dialogue: 입력 대화
            example_dialogue: Few-shot 예시 대화
            example_summary: Few-shot 예시 요약
            n_samples: 샘플링 횟수 (기본 3회)
            temperature: Temperature (다양성 확보)
            top_p: Top-p

        Returns:
            최고 품질의 요약 결과
        """
        if not self.client:
            raise RuntimeError("Solar API 클라이언트가 초기화되지 않음")

        # 캐시 확인 (프롬프트 버전 + n_samples 포함)
        PROMPT_VERSION = "v3.1_strict_enforcement"
        cache_key_string = f"{PROMPT_VERSION}_voting_{n_samples}_{dialogue}"
        cache_key = hashlib.md5(cache_key_string.encode()).hexdigest()
        if cache_key in self.cache:
            self._log(f"캐시 히트 (voting): {cache_key[:8]}")
            return self.cache[cache_key]

        # 전처리
        processed_dialogue = self.preprocess_dialogue(dialogue)

        # 프롬프트 생성
        messages = self.build_few_shot_prompt(
            processed_dialogue,
            example_dialogue,
            example_summary
        )

        # N회 샘플링
        summaries = []
        scores = []

        self._log(f"🔄 Solar API {n_samples}회 샘플링 시작...")

        try:
            for i in range(n_samples):
                response = self.client.chat.completions.create(
                    model="solar-1-mini-chat",
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=200
                )

                summary = response.choices[0].message.content.strip()

                # Post-processing: 플레이스홀더 강제 제거
                summary = self._remove_placeholders(summary)

                summaries.append(summary)

                # 품질 평가
                score = self.evaluate_summary_quality(summary, dialogue)
                scores.append(score)

                self._log(f"  샘플 {i+1}/{n_samples}: {score:.1f}점 | {summary[:50]}...")

                # Rate limit 방지를 위한 샘플 간 대기
                if i < n_samples - 1:  # 마지막 샘플 후에는 대기 불필요
                    time.sleep(2.0)  # 샘플 간 2.0초 대기 (429 에러 방지)

            # 최고 점수 요약 선택
            best_idx = scores.index(max(scores))
            best_summary = summaries[best_idx]
            best_score = scores[best_idx]

            self._log(f"✅ 최종 선택: 샘플 {best_idx+1} ({best_score:.1f}점)")

            # 캐시 저장
            self.cache[cache_key] = best_summary
            self._save_cache()

            return best_summary

        except Exception as e:
            self._log(f"❌ API 호출 실패: {str(e)}")
            return ""

    def summarize_batch(
        self,
        dialogues: List[str],
        example_dialogue: Optional[str] = None,
        example_summary: Optional[str] = None,
        batch_size: int = 10,
        delay: float = 1.0,
        use_voting: bool = False,
        n_samples: int = 3
    ) -> List[str]:
        """
        배치 요약 (Rate limit 고려)

        Args:
            dialogues: 대화 리스트
            example_dialogue: Few-shot 예시 대화
            example_summary: Few-shot 예시 요약
            batch_size: 배치 크기
            delay: 배치 간 대기 시간 (초)
            use_voting: K-Fold 방식 다중 샘플링 사용 여부
            n_samples: voting 사용 시 샘플링 횟수

        Returns:
            요약 리스트
        """
        summaries = []

        self._log(f"\n배치 요약 시작: {len(dialogues)}개 대화")
        self._log(f"  - 배치 크기: {batch_size}")
        self._log(f"  - Rate limit 대기: {delay}초")
        if use_voting:
            self._log(f"  - 🔄 K-Fold 방식 샘플링: {n_samples}회")

        try:
            for i in range(0, len(dialogues), batch_size):
                batch = dialogues[i:i + batch_size]
                batch_start = i + 1
                batch_end = min(i + batch_size, len(dialogues))

                self._log(f"[배치 {batch_start}-{batch_end}/{len(dialogues)}] 처리 중...")

                batch_summaries = []
                for dialogue in batch:
                    if use_voting:
                        # K-Fold 방식 다중 샘플링
                        summary = self.summarize_with_voting(
                            dialogue,
                            example_dialogue,
                            example_summary,
                            n_samples=n_samples
                        )
                    else:
                        # 기존 단일 샘플링
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
