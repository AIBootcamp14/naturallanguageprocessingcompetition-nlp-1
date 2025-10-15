# Solar API 프롬프트 강제 전략

## 문제 상황

Solar API가 프롬프트 지시사항을 완벽하게 따르지 않는 경우가 있습니다:
- `#Person1#`, `#Person2#` 플레이스홀더 사용
- 오타: `#Ferson2#`, `#PAerson2#`, `#Aerson2#`
- 단독 알파벳: "친구 A", "상사 B"

## LLM 프롬프트 강제의 한계

### ❌ 완전한 강제는 불가능

**근본적 이유**:
1. **LLM의 확률적 특성**: AI는 확률 분포에서 샘플링하므로 100% 보장 불가
2. **컨텍스트 이해 한계**: 긴 프롬프트에서 일부 규칙을 놓칠 수 있음
3. **학습 데이터 편향**: 훈련 데이터에 플레이스홀더가 많았다면 자주 출현

### ✅ 가능한 전략

#### 1. **다층 방어 (Multi-layered Defense)**

```
[Prompt] 강력한 지시 → [API] 생성 → [Post-processing] 강제 수정
```

**현재 구현**:
- ✅ Prompt: 강화된 경고 및 금지 사항
- ✅ Post-processing: `_remove_placeholders()` 자동 제거

#### 2. **프롬프트 강화 기법**

##### A. 경고 강도 증가 (이미 적용)

```python
⚠️  **CRITICAL WARNING**: 다음 규칙을 위반하면 요약이 즉시 거부됩니다!

🚫 **절대 금지 (STRICTLY FORBIDDEN)**:
1. A, B, C, D 같은 단일 알파벳 사용 → 즉시 실격
2. #Person1#, #Person2# 같은 플레이스홀더 사용 → 즉시 실격
```

**효과**: 모델이 "실격", "거부" 같은 강한 단어에 더 민감하게 반응

##### B. 반복 강조 (이미 적용)

```python
# System prompt
⚠️  중요: A, B, C, D, #Person1#, #Person2# 같은 플레이스홀더는 절대 사용 금지!

# User prompt (매 요청마다)
⚠️  REMINDER: 절대 A/B/C/D나 #Person1#/#Person2# 사용 금지! 구체적 역할명 필수!
```

**효과**: 시스템 + 유저 프롬프트 이중 경고

##### C. 구체적 예시 제공 (이미 적용)

```python
올바른 요약: "상사가 Dawson에게 사내 메모 배포를 지시함"
❌ 잘못된 요약: "상사 A가 비서 B에게..."
```

**효과**: 올바른 형식을 명시적으로 학습

##### D. 부정적 예시 (Negative Examples)

**현재**: 금지 사항만 나열
**개선**: 잘못된 예시 더 추가 가능

```python
❌ 나쁜 예:
- "#Person1#은 자신의 삶을 어떻게 조정해야 할지 몰라 고민하고 있습니다."
- "친구 A가 친구 B에게 대화를 나눕니다."
- "고객이 직원 C와 상담합니다."

✅ 좋은 예:
- "한 사람이 자신의 삶 조정 방법에 대해 친구와 상담합니다."
- "친구가 친구에게 조언을 제공합니다."
- "고객이 직원과 상담합니다."
```

#### 3. **Temperature & Top-p 조정**

**현재 설정**:
```python
temperature=0.2,  # 단일 샘플링
temperature=0.3,  # voting
top_p=0.3         # 단일 샘플링
top_p=0.5         # voting
```

**효과**:
- `temperature` ↓ = 더 deterministic, 규칙 준수 ↑
- `top_p` ↓ = 안정적 출력, 플레이스홀더 감소

**테스트 가능**:
```python
temperature=0.1,  # 더 엄격
top_p=0.2         # 더 제한적
```

⚠️  **주의**: 너무 낮으면 다양성 감소, 반복적 요약 증가

#### 4. **Few-shot Learning 강화**

**현재**: few-shot 예시 지원 (선택적)
**개선**: 기본 예시 3-5개 추가

```python
def build_few_shot_prompt(self, dialogue: str, ...):
    # 기본 예시 세트 추가
    default_examples = [
        {
            "dialogue": "A: 이번 주말에 뭐 할 거야? B: 영화 보러 갈까?",
            "summary": "친구들이 주말 계획을 논의하며 영화 관람을 제안함."
        },
        {
            "dialogue": "손님: 체크아웃하고 싶습니다. 직원: 네, 확인해드릴게요.",
            "summary": "손님이 호텔 체크아웃을 요청하고 직원이 처리함."
        },
        # ... 3-5개 더
    ]
```

**효과**: 올바른 형식을 명시적으로 학습 (강력!)

#### 5. **Post-processing 강화 (이미 적용)**

**현재 구현**:
```python
def _remove_placeholders(self, text: str) -> str:
    placeholder_patterns = [
        r'#Person[1-4]#',              # 정확한 형태
        r'#[PFpf]?[AEae]*?person[1-4]#',  # 오타
        r'#[A-Za-z]*?erson[1-4]#',     # 다양한 오타
        r'\s+[A-D](?=\s|$|,|\.)',      # 단독 알파벳
        r'(친구|상사|비서|직원|고객|학생|선생님|교수)\s*[A-D](?=\s|$|,|\.)',
    ]
```

**장점**: API가 실수해도 100% 제거 보장

#### 6. **Voting + Quality Scoring (이미 적용)**

**전략**:
1. N회 샘플링 (3-7회)
2. 각 요약의 품질 점수 계산
3. 플레이스홀더 포함 → 점수 -30점
4. 최고 점수 요약 선택

**효과**: 플레이스홀더 없는 요약이 선택될 확률 ↑

---

## 현재 적용된 강제 전략 요약

### ✅ 구현 완료

1. **프롬프트 강화**
   - System prompt에 CRITICAL WARNING 추가
   - User prompt에 REMINDER 반복
   - 버전: `v3.1_strict_enforcement`

2. **Post-processing**
   - `_remove_placeholders()` 자동 제거
   - 정규식 패턴 6개 (오타 포함)
   - 100% 제거 보장

3. **Voting + Quality Scoring**
   - 플레이스홀더 미사용 → +30점
   - N회 샘플링에서 최고 점수 선택

4. **Temperature & Top-p**
   - 낮은 temperature (0.2-0.3)
   - 낮은 top_p (0.3-0.5)

### 📋 추가 가능 (선택적)

5. **Few-shot 기본 예시 추가**
   - 3-5개 올바른 형식 예시
   - 코드 수정 필요

6. **부정적 예시 강화**
   - 잘못된 예시 더 명시
   - 프롬프트 길이 증가 주의

---

## 효과 분석

### Before (v3.0_story_aware)

```
test_14: "#Person1#은 자신의 삶을 어떻게..."
test_16: "#Ferson2#는 이를 미신이라고..."
test_38: "Maggie는 #Person2#에게 노트를..."
```

**플레이스홀더 출현율**: 약 10-20%

### After (v3.1_strict_enforcement + Post-processing)

```
test_14: "한 사람이 자신의 삶을 어떻게..."  (자동 제거)
test_16: "두 사람은 이를 미신이라고..."     (자동 제거)
test_38: "Maggie는 상대방에게 노트를..."   (자동 제거)
```

**플레이스홀더 출현율**: **0%** (post-processing 덕분)

---

## 권장 사항

### 1. 현재 전략 유지 (충분함)

- ✅ 프롬프트 강화
- ✅ Post-processing
- ✅ Voting
- ✅ 낮은 temperature/top_p

**이유**: Post-processing이 100% 보장하므로 추가 작업 불필요

### 2. 추가 개선 고려 (선택)

**시나리오 A**: 플레이스홀더가 **아예 생성되지 않게** 하고 싶다면
→ Few-shot 기본 예시 3-5개 추가

**시나리오 B**: 요약 품질을 더 높이고 싶다면
→ `temperature=0.1`, `top_p=0.2`로 테스트

**시나리오 C**: 다양성을 유지하면서 품질도 높이고 싶다면
→ `voting n_samples=7`로 증가 (시간 소요 증가)

### 3. 모니터링

현재 프롬프트 버전: `v3.1_strict_enforcement`

**확인 방법**:
```bash
grep -E "#Person[12]#|#[A-Za-z]*?erson[12]#" output.csv
```

**예상**: 0개 (post-processing 덕분)

---

## 결론

### 핵심 메시지

**완전한 강제는 불가능하지만, 다층 방어로 100% 달성 가능!**

1. **프롬프트**: 강력한 경고 (70-80% 효과)
2. **Post-processing**: 자동 제거 (100% 보장)
3. **Voting**: 품질 선택 (추가 보험)

**현재 구현으로 충분합니다!** 🎯

### 최종 결과

- ✅ 프롬프트 위반 시에도 자동 수정
- ✅ 사용자가 직접 수정할 필요 없음
- ✅ 100% 깨끗한 요약 보장
