"""
Solar API 통합 v3.9

대폭 개선된 버전:
1. 프롬프트 간소화 + 강력한 경고
2. Few-shot 예시 추가
3. Post-processing 대폭 강화
4. 이름 우선순위 강제 적용
5. 조사 처리 완전 수정
6. 고립된 조사 제거 (v3.8) - "가 Brian" → "Brian"
7. 조사 연쇄 및 문장 중간 조사 완전 제거 (v3.9 신규)
"""

import os
import re
import time
import hashlib
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SolarAPI:
    """Solar API 클라이언트 v3.9 (조사 연쇄 및 문장 중간 조사 완전 제거)"""

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

    def _extract_names_from_dialogue(self, dialogue: str) -> List[str]:
        """
        대화에서 이름 추출 (영어 이름, 한글 이름, Mr./Ms./Mrs. 포함)

        Returns:
            추출된 이름 리스트
        """
        names = []

        # 1. Mr./Ms./Mrs. + 영어 이름
        title_name_pattern = r'(Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        for match in re.finditer(title_name_pattern, dialogue):
            full_name = f"{match.group(1)} {match.group(2)}"
            names.append(full_name)

        # 2. 영어 이름 (대화 시작 또는 : 앞에 나오는 것들)
        # 예: "Tom: Hi Mary" → Tom, Mary
        speaker_pattern = r'([A-Z][a-z]+):'
        for match in re.finditer(speaker_pattern, dialogue):
            name = match.group(1)
            if name not in ['Person']:  # Person 제외
                names.append(name)

        # 3. 한글 이름 (3-4글자, 띄어쓰기로 구분)
        # korean_name_pattern = r'\b([가-힣]{2,4})\b(?=[,\s씨님])'
        # for match in re.finditer(korean_name_pattern, dialogue):
        #     name = match.group(1)
        #     # 일반 단어 제외
        #     if name not in ['친구', '상사', '비서', '고객', '손님', '직원', '학생', '선생', '교수', '환자', '의사', '간호사']:
        #         names.append(name)

        # 중복 제거 및 정렬 (긴 이름 우선)
        names = list(set(names))
        names.sort(key=len, reverse=True)

        return names

    def _validate_and_fix_summary(self, text: str, dialogue: str) -> str:
        """
        요약문 검증 및 강제 수정 (Post-processing v3.9)

        GPT 분석 기반 대폭 개선:
        1. 이름 우선순위 강제 적용
        2. "인+조사" 패턴 완전 수정
        3. "친구 A", "친구 B" 강제 제거
        4. 역할+이름 충돌 해결
        5. 플레이스홀더 완전 제거
        6. 고립된 조사 제거 (v3.8) - "가 Brian" → "Brian"
        7. 조사 연쇄 완전 제거 (v3.9 신규) - "가에게", "는에게" 등
        8. 형용사+조사 패턴 제거 (v3.9 신규) - "보이는에게" → "보이는 사람에게"
        9. 문장 중간 고립 조사 제거 (v3.9 신규) - ". 는" → ". "

        Args:
            text: Solar API 출력 텍스트
            dialogue: 원본 대화

        Returns:
            검증 및 수정된 요약문
        """
        if not text:
            return text

        original = text
        modified = text

        # ═══════════════════════════════════════════════════════════
        # STEP 1: 이름 추출 및 우선 적용 (최고 우선순위!)
        # ═══════════════════════════════════════════════════════════

        names = self._extract_names_from_dialogue(dialogue)

        # 이름이 있으면 역할 단어를 이름으로 치환
        if names:
            # "상사 Ms. Dawson" → "Ms. Dawson"
            # "친구 Tom" → "Tom"
            for name in names:
                # 역할+이름 패턴 제거 (이름만 남김)
                role_name_patterns = [
                    (rf'(친구|동료|상사|비서|직원|고객|손님|학생|선생님|교수|환자|의사|간호사|연인|부모|자녀)\s+{re.escape(name)}', name),
                ]

                for pattern, replacement in role_name_patterns:
                    modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 2: "친구 A와 친구 B" → "두 친구" (강제 변환)
        # ═══════════════════════════════════════════════════════════

        # 2-1. "친구 A와 친구 B" 패턴
        same_role_letter_patterns = [
            (r'(친구|동료|연인)\s+[A-D]\s*와\s+\1\s+[A-D]', r'두 \1'),
            (r'(상사|비서|직원|관리자)\s+[A-D]\s*와\s+\1\s+[A-D]', r'두 \1'),
            (r'(고객|손님|구매자)\s+[A-D]\s*와\s+\1\s+[A-D]', r'두 \1'),
            (r'(학생|선생님|교수)\s+[A-D]\s*와\s+\1\s+[A-D]', r'두 \1'),
        ]

        for pattern, replacement in same_role_letter_patterns:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)

        # 2-2. "친구 A가 친구 B에게" → "두 친구가" (모든 변형)
        complex_same_role_patterns = [
            (r'(친구|동료)\s+[A-D]\s*가\s+\1\s+[A-D]\s*에게', r'두 \1가'),
            (r'(친구|동료)\s+[A-D]\s*이\s+\1\s+[A-D]\s*에게', r'두 \1가'),
            (r'(친구|동료)\s+[A-D]\s*는\s+\1\s+[A-D]\s*에게', r'두 \1는'),
        ]

        for pattern, replacement in complex_same_role_patterns:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 3: 역할+이름 패턴 제거 (이름만 남기기)
        # ═══════════════════════════════════════════════════════════

        # 3-1. 역할 + 영어 이름
        role_english_name_patterns = [
            # "친구 Francis" → "Francis"
            r'(친구|동료|연인|상사|비서|직원|고객|손님)\s+([A-Z][a-z]+)(?=[가이와과에한의은는을를도께부까서]|\s|$|,|\.)',
            # "상사 Mr. Dawson" → "Mr. Dawson"
            r'(친구|동료|연인|상사|비서|직원|고객|손님)\s+(Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+)(?=[가이와과에한의은는을를도께부까서]|\s|$|,|\.)',
        ]

        for pattern in role_english_name_patterns:
            def replace_role_name(m):
                if len(m.groups()) == 2:
                    return m.group(2)  # 이름만
                else:
                    return f"{m.group(2)} {m.group(3)}"  # Mr./Ms. + 이름

            modified = re.sub(pattern, replace_role_name, modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 4: 한글 + 알파벳 조합 제거 ("친구 A" → "친구")
        # ═══════════════════════════════════════════════════════════

        korean_letter_patterns = [
            r'(친구|동료|연인|형제|자매|부모|자녀|친척)\s+[A-D](?=[가이와과에한의은는을를도께부까서]|\s|$|,|\.)',
            r'(상사|비서|직원|관리자|팀원|부하)\s+[A-D](?=[가이와과에한의은는을를도께부까서]|\s|$|,|\.)',
            r'(고객|손님|구매자|회원|민원인)\s+[A-D](?=[가이와과에한의은는을를도께부까서]|\s|$|,|\.)',
            r'(환자|의사|간호사|약사|학생|선생님|교수|강사)\s+[A-D](?=[가이와과에한의은는을를도께부까서]|\s|$|,|\.)',
            r'(사람|남자|여자|사용자|인물)\s+[A-D](?=[가이와과에한의은는을를도께부까서]|\s|$|,|\.)',
        ]

        for pattern in korean_letter_patterns:
            modified = re.sub(pattern, r'\1', modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 5: "인+조사" 패턴 수정 (GPT 제안)
        # ═══════════════════════════════════════════════════════════

        # 5-1. "상사인가" → "상사가" (조사 결합)
        particle_fix_patterns = [
            # "인가" → "가"
            (r'(친구|동료|상사|비서|직원|고객|손님|학생|선생님|교수|관리자|환자|의사|간호사|연인|형제|자매|부모|자녀|사람|남자|여자)인(가|이)', r'\1\2'),
            # "인에게" → "에게"
            (r'(친구|동료|상사|비서|직원|고객|손님|학생|선생님|교수|관리자|환자|의사|간호사|연인|형제|자매|부모|자녀|사람|남자|여자)인(에게|에서|에|한테)', r'\1\2'),
            # "인은" → "은"
            (r'(친구|동료|상사|비서|직원|고객|손님|학생|선생님|교수|관리자|환자|의사|간호사|연인|형제|자매|부모|자녀|사람|남자|여자)인(은|는|을|를|도|만)', r'\1\2'),
        ]

        for pattern, replacement in particle_fix_patterns:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 6: 구두점 뒤 플레이스홀더 처리
        # ═══════════════════════════════════════════════════════════

        # 6-1. "묻자,A는" → "묻자, "
        # 6-2. "이야기함.A는" → "이야기함. "
        punctuation_placeholder_patterns = [
            (r'([,.])\s*[A-D](?=[가이와과에한의은는을를도께부까서])', r'\1 '),
            (r'([,.])[A-D](?=[가이와과에한의은는을를도께부까서])', r'\1 '),
        ]

        for pattern, replacement in punctuation_placeholder_patterns:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 7: #Person#, 단독 알파벳 제거
        # ═══════════════════════════════════════════════════════════

        # 7-1. #Person1#, #Person2# 제거
        placeholder_exact = [
            r'#Person[1-4]#',
            r'#[PFpf]?[AEae]*?person[1-4]#',
            r'#[A-Za-z]*?erson[1-4]#',
        ]

        for pattern in placeholder_exact:
            modified = re.sub(pattern, '', modified, flags=re.IGNORECASE)

        # 7-2. 단독 알파벳 제거 (후순위!)
        modified = re.sub(r'\s+[A-D](?=[가이와과에한의은는을를도께부까서]|\s|$|,|\.)', '', modified, flags=re.IGNORECASE)
        modified = re.sub(r'^[A-D](?=[가이와과에한의은는을를도께부까서]|\s|,|\.)', '', modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 7.5: 알파벳+조사 연쇄 완전 제거 (v3.8 신규)
        # ═══════════════════════════════════════════════════════════

        # "A가 Brian" → "Brian"
        # "B는 Tom" → "Tom"
        # 알파벳+조사를 공백과 함께 완전 제거
        alphabet_particle_patterns = [
            r'\b[A-D](가|이|는|은|를|을|에게|에서|에|한테|와|과|도|만|부터|까지|께|께서|의)\s+',
            r'^[A-D](가|이|는|은|를|을|에게|에서|에|한테|와|과|도|만|부터|까지|께|께서|의)\s+',
        ]

        for pattern in alphabet_particle_patterns:
            modified = re.sub(pattern, '', modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 7.8: 조사 연쇄 제거 (v3.9 신규)
        # ═══════════════════════════════════════════════════════════

        # "가에게" → ""
        # "는에게" → ""
        # "가에게의" → ""
        # "를에게" → ""
        particle_chain_patterns = [
            # 조사+조사 조합 (2개)
            r'(가|이|는|은|를|을)(에게|에서|의|와|과|한테)\s*',
            # 조사+조사+조사 조합 (3개)
            r'(가|이|는|은|를|을)(에게|에서|한테)(의|와|과)\s*',
        ]

        for pattern in particle_chain_patterns:
            modified = re.sub(pattern, '', modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 7.9: 형용사/동사+조사 연쇄 제거 (v3.9 신규)
        # ═══════════════════════════════════════════════════════════

        # "피곤해 보이는에게" → "피곤해 보이는 사람에게"
        # "이웃인와" → "이웃과"
        adjective_particle_patterns = [
            # "~는에게" → "~는 사람에게"
            (r'([가-힣]+는)(에게|에서|한테)', r'\1 사람\2'),
            # "~인와/과" → "~과/와"
            (r'([가-힣]{2,})인(와|과)', r'\1\2'),
            # "~인가" → "~가"
            (r'([가-힣]{2,})인(가|이)(\s)', r'\1\2\3'),
        ]

        for pattern, replacement in adjective_particle_patterns:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)

        # ═══════════════════════════════════════════════════════════
        # STEP 8: 문장 시작의 고립된 조사 제거 (v3.8)
        # ═══════════════════════════════════════════════════════════

        # "가 Brian의..." → "Brian의..."
        # 주어가 완전히 사라지고 조사만 남은 경우 제거
        isolated_particle_pattern = r'^(가|이|는|은|를|을|에게|에서|에|한테|와|과|도|만|부터|까지|께|께서|의)\s+'
        modified = re.sub(isolated_particle_pattern, '', modified)

        # ═══════════════════════════════════════════════════════════
        # STEP 8.5: 문장 중간 고립 조사 제거 (v3.9 신규)
        # ═══════════════════════════════════════════════════════════

        # "논의함. 는 스트레스가" → "논의함. 스트레스가"
        # ". 가 도움을" → ". 도움을"
        middle_particle_pattern = r'([.!?])\s+(가|이|는|은|를|을|에게|에서|의|와|과|한테|도|만)\s+'
        modified = re.sub(middle_particle_pattern, r'\1 ', modified)

        # ═══════════════════════════════════════════════════════════
        # STEP 8: 문장 시작의 고립된 조사 제거 (v3.8)
        # ═══════════════════════════════════════════════════════════

        # "가 Brian의..." → "Brian의..."
        # 주어가 완전히 사라지고 조사만 남은 경우 제거
        isolated_particle_pattern = r'^(가|이|는|은|를|을|에게|에서|에|한테|와|과|도|만|부터|까지|께|께서)\s+'
        modified = re.sub(isolated_particle_pattern, '', modified)

        # 공백 정리 (연속 공백 제거)
        modified = re.sub(r'\s+', ' ', modified)
        modified = modified.strip()

        # ═══════════════════════════════════════════════════════════
        # STEP 10: 정리 (공백, 구두점)
        # ═══════════════════════════════════════════════════════════

        # 연속 공백 제거
        modified = re.sub(r'\s+', ' ', modified)
        # 앞뒤 공백 제거
        modified = modified.strip()
        # 구두점 앞 공백 제거
        modified = re.sub(r'\s+([,.])', r'\1', modified)
        # 구두점 뒤 공백 추가 (일관성)
        modified = re.sub(r'([,.])(?=[가-힣A-Za-z])', r'\1 ', modified)

        # ═══════════════════════════════════════════════════════════
        # STEP 11: 검증
        # ═══════════════════════════════════════════════════════════

        # 플레이스홀더 잔존 확인
        placeholder_check = [
            r'\b[A-D]\b',
            r'#Person\d+#',
            r'[가-힣]+\s*[A-D](?=[가이와과])',
        ]

        has_placeholder = any(re.search(p, modified, re.IGNORECASE) for p in placeholder_check)

        if has_placeholder:
            self._log(f"⚠️  플레이스홀더 잔존: {modified[:80]}...")

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
        """대화 전처리"""
        dialogue = ' '.join(dialogue.split())
        dialogue = dialogue.replace('#Person1#:', 'A:')
        dialogue = dialogue.replace('#Person2#:', 'B:')
        dialogue = dialogue.replace('#Person3#:', 'C:')
        dialogue = dialogue.replace('#Person4#:', 'D:')
        dialogue = self.smart_truncate(dialogue, self.token_limit)
        return dialogue

    def smart_truncate(self, text: str, max_tokens: int = 512) -> str:
        """스마트 절단"""
        estimated_tokens = self.estimate_tokens(text)
        if estimated_tokens <= max_tokens:
            return text

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
        """토큰 수 추정"""
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
        Few-shot 프롬프트 생성 (v3.9 - 포괄적 조사 제거)

        Args:
            dialogue: 입력 대화
            example_dialogue: 예시 대화
            example_summary: 예시 요약

        Returns:
            메시지 리스트
        """
        # 프롬프트 버전
        PROMPT_VERSION = "v3.9_comprehensive_particle_removal"

        # 간소화된 시스템 프롬프트 (핵심만!)
        system_prompt = f"""[{PROMPT_VERSION}] 당신은 대화 요약 전문가입니다.

🚨 **절대 금지 (즉시 실격)**:
1. ❌ A, B, C, D 같은 알파벳
2. ❌ #Person1#, #Person2# 같은 플레이스홀더
3. ❌ "친구 A", "상사 B" 같은 한글+알파벳 혼합
4. ❌ "친구 A와 친구 B" - 대신 "두 친구" 사용!

✅ **필수 규칙**:
1. **이름 최우선**: 대화에 이름이 있으면 **반드시** 사용
   - "Ms. Dawson", "Tom", "Mary", "스티븐" 등
   - ❌ "상사가 비서에게" → ✅ "상사가 Ms. Dawson에게"

2. **이름 없으면 역할 사용**:
   - 업무: "상사/비서", "고객/직원"
   - 일상: "친구", "동료"
   - 두 명: "두 친구", "친구들"

3. **간결하게**: 50-150자, 핵심만

4. **말투 확인**:
   - 반말 → 친구, 동료
   - 존댓말 → 고객, 상사, 직원

예시:
- ❌ "친구 A와 친구 B가 영화를 보기로 함"
- ✅ "두 친구가 영화를 보기로 함"

- ❌ "상사가 비서에게 메모 작성을 지시함" (이름이 있는데!)
- ✅ "상사가 Ms. Dawson에게 메모 작성을 지시함"""

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Few-shot 예시 추가 (구체적 학습!)
        if not example_dialogue or not example_summary:
            # 기본 예시 제공
            example_dialogues_and_summaries = [
                # 예시 1: 이름 사용
                {
                    "dialogue": "A: Ms. Dawson, 받아쓰기 좀 부탁드려야겠어요. B: 네, 말씀하세요. A: 이걸 오늘 오후까지 모든 직원들에게 사내 메모로 보내야 해요.",
                    "summary": "상사가 Ms. Dawson에게 사내 메모 배포를 지시함."
                },
                # 예시 2: 두 친구
                {
                    "dialogue": "A: 오늘 영화 볼래? B: 좋아! 몇 시에? A: 7시 어때? B: 완벽해!",
                    "summary": "두 친구가 저녁 7시에 영화 보기로 약속함."
                },
                # 예시 3: 이름이 있을 때
                {
                    "dialogue": "Tom: Hi Mary, 오늘 저녁에 시간 있어? Mary: 응, 있어. 왜? Tom: 같이 저녁 먹을래?",
                    "summary": "Tom이 Mary에게 저녁 식사를 제안함."
                },
            ]

            for ex in example_dialogues_and_summaries:
                messages.append({
                    "role": "user",
                    "content": f"Dialogue:\n{ex['dialogue']}\n\nSummary:"
                })
                messages.append({
                    "role": "assistant",
                    "content": ex['summary']
                })
        else:
            # 사용자 제공 예시
            messages.append({
                "role": "user",
                "content": f"Dialogue:\n{example_dialogue}\n\nSummary:"
            })
            messages.append({
                "role": "assistant",
                "content": example_summary
            })

        # 실제 입력
        messages.append({
            "role": "user",
            "content": f"""Dialogue:
{dialogue}

Summary:"""
        })

        return messages

    def summarize(
        self,
        dialogue: str,
        example_dialogue: Optional[str] = None,
        example_summary: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.3,
        max_tokens: int = 200
    ) -> str:
        """단일 대화 요약"""
        if not self.client:
            raise RuntimeError("Solar API 클라이언트가 초기화되지 않음")

        # 캐시 확인
        PROMPT_VERSION = "v3.9_comprehensive_particle_removal"
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
                max_tokens=max_tokens
            )

            summary = response.choices[0].message.content.strip()

            # Post-processing: 강제 수정
            summary = self._validate_and_fix_summary(summary, dialogue)

            # 캐시 저장
            self.cache[cache_key] = summary
            self._save_cache()

            return summary

        except Exception as e:
            self._log(f"❌ API 호출 실패: {str(e)}")
            return ""

    def evaluate_summary_quality(self, summary: str, dialogue: str) -> float:
        """요약문 품질 평가"""
        score = 0.0

        # 1. 플레이스홀더 미사용 (+30점)
        placeholder_patterns = [r'\b[A-D]\b', r'#Person\d+#', r'[가-힣]+\s*[A-D]\b']
        has_placeholder = any(re.search(p, summary) for p in placeholder_patterns)
        if not has_placeholder:
            score += 30

        # 2. 적절한 길이 (+20점)
        summary_len = len(summary)
        if 50 <= summary_len <= 150:
            score += 20
        elif summary_len < 50:
            score += 5
        else:
            score += 10

        # 3. 이름 사용 확인 (+20점)
        names = self._extract_names_from_dialogue(dialogue)
        if names:
            # 이름이 요약에 포함되어 있는지 확인
            name_used = any(name in summary for name in names)
            if name_used:
                score += 20
        else:
            score += 20  # 이름 없으면 기본 점수

        # 4. "친구 A", "친구 B" 미사용 (+30점)
        forbidden_patterns = [r'친구\s+[A-D]', r'상사\s+[A-D]', r'비서\s+[A-D]']
        has_forbidden = any(re.search(p, summary) for p in forbidden_patterns)
        if not has_forbidden:
            score += 30

        return min(100, score)

    def summarize_with_voting(
        self,
        dialogue: str,
        example_dialogue: Optional[str] = None,
        example_summary: Optional[str] = None,
        n_samples: int = 3,
        temperature: float = 0.1,
        top_p: float = 0.3,
        max_tokens: int = 200
    ) -> str:
        """K-Fold 방식 다중 샘플링 요약"""
        if not self.client:
            raise RuntimeError("Solar API 클라이언트가 초기화되지 않음")

        # 캐시 확인
        PROMPT_VERSION = "v3.9_comprehensive_particle_removal"
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

        max_retries = 3

        try:
            for i in range(n_samples):
                success = False
                attempt = 0

                while not success and attempt < max_retries:
                    try:
                        response = self.client.chat.completions.create(
                            model="solar-1-mini-chat",
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens
                        )

                        summary = response.choices[0].message.content.strip()

                        # Post-processing: 강제 수정
                        summary = self._validate_and_fix_summary(summary, dialogue)

                        summaries.append(summary)

                        # 품질 평가
                        score = self.evaluate_summary_quality(summary, dialogue)
                        scores.append(score)

                        self._log(f"  샘플 {i+1}/{n_samples}: {score:.1f}점 | {summary[:50]}...")

                        success = True

                    except Exception as e:
                        error_msg = str(e)

                        if "429" in error_msg or "rate limit" in error_msg.lower():
                            attempt += 1

                            if attempt < max_retries:
                                wait_time = 5 * (2 ** (attempt - 1))
                                self._log(f"  ⚠️  Rate Limit - {wait_time}초 대기 ({attempt}/{max_retries})...")
                                time.sleep(wait_time)
                            else:
                                if summaries:
                                    self._log(f"  ⚠️  최대 재시도 초과 - 이전 샘플 재사용")
                                    summaries.append(summaries[-1])
                                    scores.append(scores[-1])
                                    success = True
                                else:
                                    raise
                        else:
                            raise

                # Rate limit 방지
                if i < n_samples - 1:
                    time.sleep(4.0)

            # 최고 점수 선택
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
        n_samples: int = 3,
        max_tokens: int = 200
    ) -> List[str]:
        """배치 요약"""
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
                for idx, dialogue in enumerate(batch):
                    dialogue_idx = i + idx + 1

                    # 대화문 출력 (첫 100자)
                    dialogue_preview = dialogue[:100].replace('\n', ' ')
                    self._log(f"\n[{dialogue_idx}/{len(dialogues)}] 대화: {dialogue_preview}...")

                    if use_voting:
                        summary = self.summarize_with_voting(
                            dialogue,
                            example_dialogue,
                            example_summary,
                            n_samples=n_samples,
                            max_tokens=max_tokens
                        )
                    else:
                        summary = self.summarize(
                            dialogue,
                            example_dialogue,
                            example_summary,
                            max_tokens=max_tokens
                        )
                    batch_summaries.append(summary)

                    # 요약 결과 출력
                    self._log(f"[{dialogue_idx}/{len(dialogues)}] 요약: {summary}")

                summaries.extend(batch_summaries)
                self._log(f"\n  ✅ 배치 완료 (누적: {len(summaries)}/{len(dialogues)})")

                if i + batch_size < len(dialogues):
                    time.sleep(delay)

        except Exception as e:
            self._log(f"\n❌ 배치 요약 중 오류: {str(e)}")
            self._log(f"  진행 상황: {len(summaries)}/{len(dialogues)}개 완료")
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
    """Solar API 생성"""
    return SolarAPI(api_key, token_limit, cache_dir, logger)
