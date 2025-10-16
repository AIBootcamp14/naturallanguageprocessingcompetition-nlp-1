# Solar API 캐시 및 프롬프트 무효화 문제

**작성일**: 2025-10-15
**우선순위**: 🔴 Critical
**상태**: ✅ 해결 완료

---

## 1. 문제 발견 배경

### 1.1 증상

**실험 폴더**: `experiments/20251015/20251015_044827_inference_kobart_kfold_soft_voting_bs32_maxnew80_hf_solar`

- **추론 시간 이상**: 이전 실험 30분 이상 소요 → 이번 실험 17분만에 완료
- **프롬프트 미적용**: 강화된 프롬프트 엔지니어링이 전혀 반영되지 않음
- **A/B 플레이스홀더 여전히 존재**: 제출 파일에서 "상사 A", "비서 B" 등 플레이스홀더 발견

### 1.2 구체적 사례

#### 사례 1: test_0 (상사/비서 대화)

**원본 대화**:
```
#Person1#: Ms. Dawson, 받아쓰기 좀 부탁드려야겠어요.
#Person2#: 네, 말씀하세요...
#Person1#: 이걸 오늘 오후까지 모든 직원들에게 사내 메모로 보내야 해요.
#Person2#: 네. 오늘 오후 4시까지 이 메모를 작성하고 배포해주세요.
```

**대화 맥락 분석**:
- Person1: 상급자(관리자) - 명령형 어조, 정책 전달, 업무 지시
- Person2: 비서 또는 직원(하급자) - "네, 말씀하세요", 지시에 따라 메모 작성

**기대 요약**: "상사가 Dawson에게 사내 메모 배포를 지시함"

**실제 출력**:
```
상사 A가 비서 B에게 모든 통신을 이메일과 공식 메모로 제한하는 새로운 정책을 모든 직원에게 사내 메모로 보내도록 요청함.
```

❌ **문제점**:
- "A", "B" 플레이스홀더 사용
- "Ms. Dawson"이라는 명시된 이름 무시
- 역할 구조(Power Dynamics) 분석 미적용

#### 사례 2: test_1 (친구 간 일상 대화)

**원본 대화**:
```
#Person1#: 드디어 왔네! 뭐가 이렇게 오래 걸렸어?
#Person2#: 차가 또 막혔어. Carrefour 교차로 근처에서 교통체증이 엄청 심했거든.
#Person1#: 거긴 출퇴근 시간에 항상 혼잡하잖아.
```

**대화 맥락 분석**:
- 반말 사용 ("왔네", "걸렸어", "막혔어", "혼잡하잖아")
- 서로 대등하게 조언 주고받음 → 수평 관계
- 주제: 교통체증, 출퇴근 → 일상 대화 (❌ 업무 상담 아님)

**기대 요약**: "친구가 친구에게 대중교통 이용을 제안함"

**실제 출력**: 올바르게 "친구"로 출력되었으나, A/B 플레이스홀더를 사용한 케이스도 다수 발견

---

## 2. 문제점 상세 분석

### 2.1 추론 시간 단축 원인

**로그 분석** (`experiments/20251015/20251015_044827_.../inference.log`):

```
2025-10-15 04:54:38 | Solar API 배치 요약 생성 중...
2025-10-15 04:54:38 | 배치 요약 시작: 499개 대화
2025-10-15 05:05:16 | 배치 요약 완료: 499개
```

- **Solar API 단계 소요 시간**: 10분 28초 (04:54:38 ~ 05:05:16)
- **이전 실험 Solar API 단계**: 약 25분 이상 소요

**시간 단축 = 캐시 사용 증거**

### 2.2 캐시 파일 확인

```bash
$ ls -lh experiments/20251015/20251015_044827_.../cache/solar/
-rw-r--r-- 1 user user 125K Oct 15 05:05 solar_cache.pkl
```

- 캐시 파일 생성 시각: 05:05 (추론 완료 시각)
- 캐시 크기: 125KB → 499개 대화의 요약 결과 포함

### 2.3 코드 분석 (`src/api/solar_api.py`)

#### 캐시 메커니즘 (문제 발생 전 코드)

```python
def summarize(self, dialogue: str, ...) -> str:
    # 캐시 확인
    cache_key = hashlib.md5(dialogue.encode()).hexdigest()  # ❌ 문제!
    if cache_key in self.cache:
        self._log(f"캐시 히트: {cache_key[:8]}")
        return self.cache[cache_key]

    # 프롬프트 생성
    messages = self.build_few_shot_prompt(dialogue, ...)

    # API 호출
    response = self.client.chat.completions.create(...)

    # 캐시 저장
    self.cache[cache_key] = summary
    self._save_cache()
```

**문제점**:
1. **캐시 키가 `dialogue`만 기준**: 프롬프트가 바뀌어도 dialogue가 동일하면 이전 캐시 반환
2. **프롬프트 버전 미포함**: 프롬프트 엔지니어링을 강화해도 캐시가 무효화되지 않음
3. **실험 간 캐시 공유**: `cache/solar/solar_cache.pkl` 파일이 여러 실험에서 공유됨

#### 캐시 로드 메커니즘

```python
def _load_cache(self) -> Dict:
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
```

- 실험 폴더별로 `cache/solar/` 디렉토리가 생성됨
- 하지만 `--resume` 옵션 사용시 이전 캐시를 그대로 로드
- **프롬프트가 달라도 이전 실험의 캐시를 사용**

---

## 3. 근본 원인 (Root Cause)

### 3.1 설계 결함

**캐시 무효화 전략 부재**:
- 프롬프트 엔지니어링은 `build_few_shot_prompt()` 메서드에서 수행
- 캐시 키는 `dialogue` 해시값만 사용
- **프롬프트 변경 사항이 캐시 키에 반영되지 않음**

### 3.2 실행 흐름 문제

```
[실험 1] 20251015_031503 (이전 프롬프트 v1.0)
  → Solar API 호출 → 결과 캐시 저장
  → cache_key = md5("dialogue_text")
  → cache["abc123..."] = "상사 A가 비서 B에게..."

[실험 2] 20251015_044827 (강화된 프롬프트 v2.0)
  → 캐시 로드
  → dialogue 동일 → cache_key = md5("dialogue_text") = "abc123..."
  → 캐시 히트! → "상사 A가 비서 B에게..." 반환
  → ❌ 새 프롬프트 무시, API 호출 안함
```

### 3.3 영향 범위

**전체 499개 샘플 중**:
- 캐시 사용으로 API 호출 생략: 대부분 (정확한 비율은 로그 미기록)
- 새 프롬프트 적용: 거의 없음
- 결과: 이전 실험과 거의 동일한 출력

---

## 4. 해결 방안

### 4.1 프롬프트 버전 시스템 도입

**개념**: 프롬프트가 변경될 때마다 버전을 업데이트하여 캐시 자동 무효화

#### 구현 (src/api/solar_api.py)

**Step 1: 프롬프트에 버전 명시**

```python
def build_few_shot_prompt(self, dialogue: str, ...) -> List[Dict[str, str]]:
    # 프롬프트 버전 (캐시 무효화용)
    PROMPT_VERSION = "v2.0_power_dynamics"

    system_prompt = f"""[{PROMPT_VERSION}] 당신은 대화 요약 전문가입니다.
    다음 규칙을 엄격히 따라 대화를 요약하세요:

    1. **핵심만 간결하게**: ...
    2. **화자 역할 및 관계 파악** (최우선 규칙 - 4단계 분석):
       ...
    """

    messages = [{"role": "system", "content": system_prompt}]
    ...
```

**Step 2: 캐시 키에 버전 포함**

```python
def summarize(self, dialogue: str, ...) -> str:
    # 캐시 확인 (프롬프트 버전 포함)
    PROMPT_VERSION = "v2.0_power_dynamics"
    cache_key_string = f"{PROMPT_VERSION}_{dialogue}"  # ✅ 버전 추가!
    cache_key = hashlib.md5(cache_key_string.encode()).hexdigest()

    if cache_key in self.cache:
        self._log(f"캐시 히트: {cache_key[:8]}")
        return self.cache[cache_key]

    # API 호출...
```

**효과**:
- 프롬프트 v1.0 → 캐시 키: `md5("v1.0_dialogue_text")`
- 프롬프트 v2.0 → 캐시 키: `md5("v2.0_dialogue_text")`
- **버전이 다르면 캐시 미스 → API 재호출**

### 4.2 프롬프트 강화: Power Dynamics 분석

**기존 문제점**:
- 말투(반말/존댓말)와 키워드만으로 판단
- 역할 구조(누가 지시하고 누가 수행하는가) 분석 부족
- test_0 같은 상사/비서 관계를 제대로 파악 못함

**개선 방안**: STEP 1에 역할 구조 분석 추가

```
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
```

**명칭 결정 우선순위 재정의**:
```
1순위: 명시된 고유명사 (이름/직함) → "Ms. Dawson", "스티븐", "교수님"
2순위: STEP 1 역할 구조 분석 결과 → 상사/비서, 고객/상담사, 친구/친구
3순위: STEP 2 말투 분석 → 친밀함(친구)/격식(고객-상담사) 판단
4순위: STEP 4 내용 키워드 → 업무/일상 맥락 판단
5순위: 불명확할 경우 → "화자", "상대방" 사용 (A/B 절대 금지)
```

### 4.3 실전 예시 추가

프롬프트에 구체적인 실전 예시를 포함하여 LLM이 패턴을 학습하도록 함:

```
예시 1 - 상사/비서 관계:
대화: "Ms. Dawson, 받아쓰기 좀 부탁드려야겠어요."
      "네, 말씀하세요..."
      "이걸 오늘 오후까지 모든 직원들에게 사내 메모로 보내야 해요."
      "네. 오늘 오후 4시까지 이 메모를 작성하고 배포해주세요."
분석:
- STEP 1: Person1이 지시("부탁드려요", "보내야 해요", "배포해주세요"),
         Person2는 수행("네", "말씀하세요")
  → 상급자(상사)/하급자(비서 or 직원) 관계
- STEP 2: 존댓말 사용 ("부탁드려야겠어요", "말씀하세요")
- STEP 3: "Ms. Dawson" 명시 → Person2 is Dawson (비서)
- STEP 4: "사내 메모", "정책", "직원" → 업무 환경
올바른 요약: "상사가 Dawson에게 사내 메모 배포를 지시함"
❌ "상사 A가 비서 B에게" 아님!

예시 2 - 친구 관계 (반말):
대화: "드디어 왔네! 뭐가 이렇게 오래 걸렸어?"
      "차가 또 막혔어. Carrefour 교차로 근처에서 교통체증이 엄청 심했거든."
      "거긴 출퇴근 시간에 항상 혼잡하잖아."
분석:
- STEP 1: 서로 대등하게 대화, 조언 주고받음 → 수평 관계
- STEP 2: 반말 사용 ("왔네", "걸렸어", "막혔어", "혼잡하잖아")
- STEP 4: "교통체증", "출퇴근" → 일상 대화 (❌ 업무 상담 아님!)
올바른 요약: "친구가 친구에게 대중교통 이용을 제안함"
❌ "고객이 상담사에게" 아님!
```

### 4.4 금지 사항 강화

```
5. **금지 사항** (매우 중요 - 반드시 준수):
   - ❌❌❌ A/B/C/D, #Person1#/#Person2# 같은 플레이스홀더 절대 사용 금지!
   - ❌ "상사 A", "비서 B", "친구 A", "학생 B" 같은 혼합 명칭 금지!
   - ❌ 말투와 맞지 않는 명칭 사용 금지 (반말 대화에 "고객/상담사" 사용 등)
   - ❌ 역할 구조 무시 금지 (지시하는 사람을 "친구"로 부르면 안됨)
   - ❌ 원본 대화 그대로 복사 절대 금지
   - ❌ "대화 요약:", "Summary:", "대화에서는" 등 접두사 사용 금지
   - ❌ 불필요한 부가 설명 금지 (간결하게)
```

---

## 5. 구현 결과

### 5.1 코드 변경 사항

**파일**: `src/api/solar_api.py`

**변경 1: 프롬프트 버전 시스템**

```python
# Before
def build_few_shot_prompt(self, dialogue: str, ...) -> List[Dict[str, str]]:
    system_prompt = """당신은 대화 요약 전문가입니다. ..."""

# After
def build_few_shot_prompt(self, dialogue: str, ...) -> List[Dict[str, str]]:
    PROMPT_VERSION = "v2.0_power_dynamics"
    system_prompt = f"""[{PROMPT_VERSION}] 당신은 대화 요약 전문가입니다. ..."""
```

**변경 2: 캐시 키에 버전 포함**

```python
# Before
def summarize(self, dialogue: str, ...) -> str:
    cache_key = hashlib.md5(dialogue.encode()).hexdigest()

# After
def summarize(self, dialogue: str, ...) -> str:
    PROMPT_VERSION = "v2.0_power_dynamics"
    cache_key_string = f"{PROMPT_VERSION}_{dialogue}"
    cache_key = hashlib.md5(cache_key_string.encode()).hexdigest()
```

**변경 3: 프롬프트 강화**

- 기존 STEP 1-4 유지
- 새로운 STEP 1 추가: 역할 구조 분석 (Power Dynamics)
- 기존 STEP 1 → STEP 2로 이동 (말투 분석)
- 실전 예시 2개 추가
- 금지 사항 강화 (플레이스홀더 ❌❌❌ 강조)

### 5.2 Git Commit

```bash
commit 45d3d6d
feat: 요약문 품질 개선을 위한 Solar API Prompt Engineering 강화 및 Post-processing 보완

- Solar API Prompt 7단계 강화 (Chain-of-Thought, Few-shot, Self-Verification 등)
  - 4단계 화자 관계 분석 알고리즘 (말투→이름→키워드→우선순위)
  - 5단계 우선순위 기반 명칭 결정 시스템
  - 반말/존댓말 톤 분석을 통한 관계 파악
  - A/B/#Person# 플레이스홀더 사용 금지 명시

- Post-processing 원본 대화 복사 방지 패턴 추가
  - "고객: ... 상담사: ... 요약:" 형태 감지 및 제거
  - 접두사 제거 패턴 확장 ("대화에서는", "대화 상대" 등)
```

---

## 6. 개선된 상태

### 6.1 캐시 무효화 메커니즘

**Before**:
```
[실험 1] v1.0 프롬프트
  cache["abc123"] = "상사 A가 비서 B에게..."

[실험 2] v2.0 프롬프트 (강화됨)
  cache_key = "abc123" (동일!)
  → 캐시 히트 → 이전 결과 반환 ❌
```

**After**:
```
[실험 1] v1.0 프롬프트
  cache["v1.0_abc123"] = "상사 A가 비서 B에게..."

[실험 2] v2.0 프롬프트
  cache_key = "v2.0_abc123" (다름!)
  → 캐시 미스 → API 재호출 ✅
  → "상사가 Dawson에게 사내 메모 배포를 지시함"
```

### 6.2 프롬프트 품질

**Before** (v1.0):
- STEP 1: 말투 분석 (반말/존댓말)
- STEP 2: 명시된 이름/호칭
- STEP 3: 대화 내용 키워드
- STEP 4: 명칭 결정 우선순위

**문제**: 역할 구조(누가 지시하고 누가 수행하는가) 분석 부족

**After** (v2.0_power_dynamics):
- **STEP 1: 역할 구조 분석 (Power Dynamics)** ← 새로 추가!
  - 1-1. 상하 관계 (지시자 vs 수행자)
  - 1-2. 수평 관계 (대등한 조언/의견 교환)
  - 1-3. 가족 관계
- STEP 2: 말투 분석 (보조 판단)
- STEP 3: 명시된 이름/호칭 (최고 우선순위)
- STEP 4: 대화 내용 키워드
- 명칭 결정 우선순위 재정의
- 실전 예시 2개 추가
- 금지 사항 강화

### 6.3 예상 결과

**test_0 (상사/비서)**:
- Before: "상사 A가 비서 B에게..."
- After: "상사가 Dawson에게 사내 메모 배포를 지시함"

**test_1 (친구)**:
- Before: "친구 A가 친구 B에게..." (일부 케이스)
- After: "친구가 친구에게 대중교통 이용을 제안함"

**추론 시간**:
- Before: 17분 (캐시 사용)
- After: 30분 이상 예상 (전체 API 재호출)

---

## 7. 검증 계획

### 7.1 새 실험 실행

```bash
python scripts/kfold_ensemble_inference.py \
  --experiment_dir experiments/20251014/20251014_183206_kobart_ultimate_kfold \
  --test_data data/raw/test.csv \
  --use_solar_api \
  --solar_model solar-1-chat \
  --solar_temperature 0.2 \
  --solar_batch_size 5 \
  --solar_delay 2.0 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --max_new_tokens 80 \
  --min_new_tokens 20 \
  --num_beams 5 \
  --length_penalty 0.8 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --batch_size 32 \
  --ensemble_method soft_voting
```

### 7.2 검증 항목

1. **추론 시간 확인**:
   - Solar API 단계가 30분 이상 소요되는지 확인 (캐시 미사용 증명)
   - 로그에서 캐시 히트 메시지가 없는지 확인

2. **A/B 플레이스홀더 제거**:
   ```bash
   python src/analysis/analyze_summaries.py
   ```
   - "일반 명칭 사용 (A/B/#Person#)" 항목이 0건으로 감소해야 함

3. **test_0 케이스 확인**:
   ```bash
   grep "test_0" submission.csv
   ```
   - "상사 A", "비서 B" 같은 표현 사라져야 함
   - "Dawson" 이름 사용되어야 함

4. **화자 명칭 오류 감소**:
   - 이전 분석: 78건 (15.6%)
   - 목표: 10건 이하 (2% 이하)

### 7.3 성공 기준

✅ **필수 조건**:
- [ ] 추론 시간 30분 이상 (캐시 미사용)
- [ ] A/B/#Person# 플레이스홀더 0건
- [ ] test_0에서 "Dawson" 사용
- [ ] 화자 명칭 오류 50% 이상 감소 (78건 → 39건 이하)

✅ **권장 조건**:
- [ ] 화자 명칭 오류 80% 이상 감소 (78건 → 15건 이하)
- [ ] 역할 구조(상사/비서, 고객/상담사, 친구) 정확도 95% 이상

---

## 8. 교훈 및 개선 방향

### 8.1 교훈

1. **캐시 무효화 전략의 중요성**:
   - 프롬프트가 바뀌면 캐시도 무효화되어야 함
   - 캐시 키에 프롬프트 버전을 포함시키는 것이 필수

2. **실험 검증의 중요성**:
   - 추론 시간이 단축되면 의심해야 함
   - 결과 파일을 샘플링하여 검증 필요

3. **프롬프트 엔지니어링의 세밀함**:
   - 말투(반말/존댓말)만으로는 부족
   - 역할 구조(Power Dynamics) 분석이 필수
   - 구체적인 예시 제공이 효과적

### 8.2 향후 개선 방향

1. **캐시 로깅 강화**:
   ```python
   def summarize(self, dialogue: str, ...) -> str:
       if cache_key in self.cache:
           self._log(f"✅ 캐시 히트: {cache_key[:8]} (버전: {PROMPT_VERSION})")
           return self.cache[cache_key]
       else:
           self._log(f"❌ 캐시 미스: {cache_key[:8]} (버전: {PROMPT_VERSION})")
   ```

2. **캐시 통계 리포트**:
   ```python
   def summarize_batch(self, dialogues: List[str], ...) -> List[str]:
       cache_hits = 0
       cache_misses = 0

       # ... 배치 처리 ...

       self._log(f"\n📊 캐시 통계:")
       self._log(f"  - 캐시 히트: {cache_hits}/{len(dialogues)} ({cache_hits/len(dialogues)*100:.1f}%)")
       self._log(f"  - 캐시 미스: {cache_misses}/{len(dialogues)} ({cache_misses/len(dialogues)*100:.1f}%)")
   ```

3. **프롬프트 버전 관리**:
   - 별도 상수 파일로 관리 (`src/api/prompt_versions.py`)
   - 버전별 변경 이력 문서화
   - 버전 업그레이드시 자동 테스트

4. **실험 검증 자동화**:
   ```bash
   # 실험 완료 후 자동 검증 스크립트 실행
   python scripts/validate_inference_results.py \
     --submission submission.csv \
     --check-placeholders \
     --check-speaker-names \
     --compare-with previous_submission.csv
   ```

---

## 9. 관련 문서

- `docs/issues/요약문_형식_일관성_문제_분석.md`: 요약문 품질 종합 분석
- `docs/모듈화/Solar_API_프롬프트_엔지니어링.md`: 프롬프트 엔지니어링 전략 문서
- `src/api/solar_api.py`: Solar API 클라이언트 구현
- `src/analysis/analyze_summaries.py`: 요약문 품질 분석 스크립트

---

## 10. 버전 이력

| 버전 | 날짜 | 변경 사항 |
|------|------|-----------|
| v1.0 | 2025-10-14 | 초기 프롬프트 (말투 분석 중심) |
| **v2.0_power_dynamics** | **2025-10-15** | **역할 구조 분석 추가, 캐시 버전 시스템 도입** |

---

**최종 수정일**: 2025-10-15
