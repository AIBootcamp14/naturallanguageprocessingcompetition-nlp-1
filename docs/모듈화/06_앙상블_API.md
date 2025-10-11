# 앙상블 및 API 시스템 가이드

> **통합 문서:** 앙상블 시스템 + Solar API + 프롬프트 엔지니어링

## 📋 목차

### Part 1: 앙상블 시스템
- [개요](#part-1-앙상블-시스템)
- [가중치 앙상블](#가중치-앙상블)
- [투표 앙상블](#투표-앙상블)
- [모델 매니저](#모델-매니저)
- [사용 방법](#앙상블-사용-방법)

### Part 2: Solar API 시스템
- [개요](#part-2-solar-api-시스템)
- [SolarAPI 클래스](#solarapi-클래스)
- [토큰 최적화](#토큰-최적화)
- [Few-shot Learning](#few-shot-learning)
- [사용 방법](#solar-api-사용-방법)
- [실행 명령어](#solar-api-실행-명령어)

### Part 3: 프롬프트 엔지니어링
- [개요](#part-3-프롬프트-엔지니어링)
- [PromptTemplate 클래스](#prompttemplate-클래스)
- [PromptLibrary 클래스](#promptlibrary-클래스)
- [PromptSelector 클래스](#promptselector-클래스)
- [프롬프트 템플릿 종류](#프롬프트-템플릿-종류)
- [사용 방법](#프롬프트-사용-방법)

---

# 📌 Part 1: 앙상블 시스템

## 📝 개요

### 목적
- 여러 모델의 예측을 결합하여 성능 향상
- 가중치 기반 앙상블
- 투표 기반 앙상블

### 핵심 기능
- ✅ Weighted Ensemble (가중 평균)
- ✅ Voting Ensemble (Hard/Soft Voting)
- ✅ ModelManager (모델 관리)

---

## ⚖️ 가중치 앙상블

### 파일 위치
```
src/ensemble/weighted.py
```

### 클래스 구조

```python
class WeightedEnsemble:
    def __init__(models, tokenizers, weights=None)
    def predict(dialogues, max_length, num_beams, batch_size)
```

### 원리

각 모델의 예측에 가중치를 부여하여 최종 예측 선택:

```
최종 예측 = argmax(w1 * 모델1_예측 + w2 * 모델2_예측 + ...)
```

**가중치 설정 전략:**
1. **균등 가중치**: 모든 모델에 동일한 가중치 (1/N)
2. **성능 기반 가중치**: 검증 ROUGE 점수에 비례
3. **수동 가중치**: 도메인 지식 기반

### 사용 예시

```python
from src.ensemble import WeightedEnsemble

# 모델 로드 (이미 로드된 모델 가정)
models = [model1, model2, model3]
tokenizers = [tokenizer1, tokenizer2, tokenizer3]

# 가중치 설정 (ROUGE 점수 기반)
weights = [0.5, 0.3, 0.2]  # 모델1이 가장 높은 성능

# 앙상블 생성
ensemble = WeightedEnsemble(models, tokenizers, weights)

# 예측
predictions = ensemble.predict(
    dialogues=test_dialogues,
    max_length=200,
    num_beams=4,
    batch_size=8
)
```

### 균등 가중치 사용

```python
# 가중치 없이 초기화 → 자동으로 균등 가중치
ensemble = WeightedEnsemble(models, tokenizers)
# weights = [0.333, 0.333, 0.333]
```

---

## 🗳️ 투표 앙상블

### 파일 위치
```
src/ensemble/voting.py
```

### 클래스 구조

```python
class VotingEnsemble:
    def __init__(models, tokenizers, voting="hard")
    def predict(dialogues, max_length, num_beams, batch_size)
```

### Hard Voting (다수결)

**원리:**
- 각 모델의 예측 중 가장 많이 나온 것 선택
- 동일한 표를 받은 경우 첫 번째 선택

**예시:**
```
입력: "두 사람이 저녁 약속을 잡았다"

모델1 예측: "저녁 약속 잡음"
모델2 예측: "저녁 약속 잡음"
모델3 예측: "저녁 식사 계획"

→ 최종: "저녁 약속 잡음" (2표)
```

### 사용 예시

```python
from src.ensemble import VotingEnsemble

models = [model1, model2, model3]
tokenizers = [tokenizer1, tokenizer2, tokenizer3]

# Hard Voting 앙상블
ensemble = VotingEnsemble(models, tokenizers, voting="hard")

# 예측
predictions = ensemble.predict(
    dialogues=test_dialogues,
    max_length=200,
    num_beams=4,
    batch_size=8
)
```

---

## 🎛️ 모델 매니저

### 파일 위치
```
src/ensemble/manager.py
```

### 클래스 구조

```python
class ModelManager:
    def __init__()
    def load_model(model_path, model_name)
    def load_models(model_paths, model_names)
    def create_ensemble(ensemble_type, weights, voting)
    def get_info()
```

### 주요 기능

#### 1. 모델 로드

```python
from src.ensemble import ModelManager

manager = ModelManager()

# 단일 모델 로드
manager.load_model(
    model_path="outputs/baseline_kobart/final_model",
    model_name="KoBART"
)

# 여러 모델 로드
manager.load_models(
    model_paths=[
        "outputs/baseline_kobart/final_model",
        "outputs/kobart_v2/final_model",
        "outputs/kobart_v3/final_model"
    ],
    model_names=["KoBART_v1", "KoBART_v2", "KoBART_v3"]
)
```

#### 2. 앙상블 생성

**가중치 앙상블:**
```python
ensemble = manager.create_ensemble(
    ensemble_type="weighted",
    weights=[0.5, 0.3, 0.2]
)
```

**투표 앙상블:**
```python
ensemble = manager.create_ensemble(
    ensemble_type="voting",
    voting="hard"
)
```

#### 3. 정보 조회

```python
info = manager.get_info()
print(f"모델 수: {info['num_models']}")
print(f"모델 이름: {info['model_names']}")
```

---

## 💻 앙상블 사용 방법

### 전체 파이프라인 예시

```python
from src.ensemble import ModelManager
import pandas as pd

# 1. 모델 매니저 생성
manager = ModelManager()

# 2. 여러 모델 로드
model_paths = [
    "outputs/baseline_kobart/final_model",
    "outputs/kobart_fold1/final_model",
    "outputs/kobart_fold2/final_model"
]

manager.load_models(model_paths)

# 3. 가중치 앙상블 생성
# ROUGE 점수 기반 가중치
weights = [0.45, 0.30, 0.25]  # 검증 성능에 비례

ensemble = manager.create_ensemble(
    ensemble_type="weighted",
    weights=weights
)

# 4. 테스트 데이터 로드
test_df = pd.read_csv("data/raw/test.csv")
dialogues = test_df['dialogue'].tolist()

# 5. 예측
predictions = ensemble.predict(
    dialogues=dialogues,
    max_length=200,
    num_beams=4,
    batch_size=8
)

# 6. 결과 저장
output_df = pd.DataFrame({
    'fname': test_df['fname'],
    'summary': predictions
})
output_df.to_csv("submissions/ensemble_submission.csv", index=False)

print(f"앙상블 예측 완료: {len(predictions)}개")
```

---

### K-Fold 모델 앙상블

```python
from src.ensemble import ModelManager

manager = ModelManager()

# K-Fold로 학습된 모델들 로드
fold_paths = [
    f"outputs/baseline_kobart_fold{i}/final_model"
    for i in range(1, 6)  # 5-Fold
]

manager.load_models(fold_paths)

# 균등 가중치 앙상블 (K-Fold는 보통 균등)
ensemble = manager.create_ensemble(ensemble_type="weighted")

# 예측
predictions = ensemble.predict(dialogues)
```

---

# 📌 Part 2: Solar API 시스템

## 📝 개요

### 목적
- Upstage Solar API를 활용한 대화 요약
- 토큰 사용량 최적화 (70-75% 절약)
- Few-shot Learning 지원
- 캐싱으로 비용 절감

### 핵심 기능
- ✅ Few-shot 프롬프트 생성
- ✅ 토큰 최적화 (대화 전처리, 스마트 절단)
- ✅ 배치 처리 (Rate limit 고려)
- ✅ 캐싱 (중복 요청 방지)

---

## 🔌 SolarAPI 클래스

### 파일 위치
```
src/api/solar_api.py
```

### 클래스 구조

```python
class SolarAPI:
    def __init__(api_key, token_limit, cache_dir)
    def preprocess_dialogue(dialogue)
    def smart_truncate(text, max_tokens)
    def estimate_tokens(text)
    def build_few_shot_prompt(dialogue, example_dialogue, example_summary)
    def summarize(dialogue, ...)
    def summarize_batch(dialogues, ...)
```

### 초기화

```python
from src.api import SolarAPI

api = SolarAPI(
    api_key="your_api_key",  # 또는 환경 변수 SOLAR_API_KEY
    token_limit=512,          # 대화당 최대 토큰
    cache_dir="cache/solar"   # 캐시 디렉토리
)
```

---

## ⚡ 토큰 최적화

### 1. 대화 전처리

**목적:** 불필요한 토큰 제거

```python
def preprocess_dialogue(dialogue):
    # 1. 공백 제거
    dialogue = ' '.join(dialogue.split())

    # 2. Person 태그 간소화
    #    #Person1#: → A:
    #    #Person2#: → B:
    dialogue = dialogue.replace('#Person1#:', 'A:')
    dialogue = dialogue.replace('#Person2#:', 'B:')

    # 3. 스마트 절단
    dialogue = smart_truncate(dialogue, 512)

    return dialogue
```

**효과:**
```
원본: "#Person1#: 안녕하세요. 오늘 날씨가 좋네요. #Person2#: 네, 정말 좋아요."
전처리: "A: 안녕하세요. 오늘 날씨가 좋네요. B: 네, 정말 좋아요."

토큰 절약: 약 15-20%
```

---

### 2. 스마트 절단

**목적:** 문장 단위로 토큰 제한

```python
def smart_truncate(text, max_tokens=512):
    # 토큰 수 추정
    estimated = estimate_tokens(text)

    if estimated <= max_tokens:
        return text

    # 문장 단위로 자르기 (마침표 기준)
    sentences = text.split('.')
    truncated = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens:
            break

        truncated.append(sentence)
        current_tokens += sentence_tokens

    return '.'.join(truncated) + '.'
```

**특징:**
- 문장 중간에서 자르지 않음
- 의미 보존
- 정확한 토큰 제한

---

### 3. 토큰 추정

**공식:**
```python
def estimate_tokens(text):
    # 한글: 2.5자 = 1토큰
    korean_chars = len(re.findall(r'[가-힣]', text))
    korean_tokens = korean_chars / 2.5

    # 영어: 4자 = 1토큰
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    english_tokens = english_chars / 4

    # 기타: 3자 = 1토큰
    other_chars = len(text) - korean_chars - english_chars
    other_tokens = other_chars / 3

    return int(korean_tokens + english_tokens + other_tokens)
```

**정확도:** ±5% 내외

---

## 📚 Few-shot Learning

### 프롬프트 구조

```python
messages = [
    # 1. System 프롬프트
    {
        "role": "system",
        "content": "You are an expert in dialogue summarization..."
    },

    # 2. User 예시 (Few-shot)
    {
        "role": "user",
        "content": "Dialogue:\nA: 점심 뭐 먹을까? B: 김치찌개\nSummary:"
    },

    # 3. Assistant 답변 (Few-shot)
    {
        "role": "assistant",
        "content": "점심 메뉴 상의"
    },

    # 4. 실제 입력
    {
        "role": "user",
        "content": f"Dialogue:\n{dialogue}\nSummary:"
    }
]
```

### Few-shot 예시 선택 전략

**1. 대표 샘플:**
```python
# 평균 길이, 일반적인 주제
example_dialogue = "A: 오늘 회의 시간 정했어? B: 3시로 하자"
example_summary = "회의 시간 결정"
```

**2. 다양한 예시 (3-shot):**
```python
examples = [
    ("짧은 대화", "짧은 요약"),
    ("중간 대화", "중간 요약"),
    ("긴 대화", "긴 요약")
]
```

---

## 💻 Solar API 사용 방법

### 1. 환경 변수 설정

```bash
export SOLAR_API_KEY="your_api_key_here"
```

또는 `.env` 파일:
```
SOLAR_API_KEY=your_api_key_here
```

---

### 2. 단일 대화 요약

```python
from src.api import SolarAPI

# API 초기화
api = SolarAPI()

# 대화 요약
dialogue = "A: 안녕하세요 B: 안녕하세요 A: 오늘 날씨 좋네요 B: 네, 정말 좋아요"

summary = api.summarize(
    dialogue=dialogue,
    temperature=0.2,  # 낮을수록 일관성 ↑
    top_p=0.3         # 낮을수록 일관성 ↑
)

print(f"요약: {summary}")
```

---

### 3. Few-shot 예시 사용

```python
# Few-shot 예시 준비
example_dialogue = "A: 점심 뭐 먹을까? B: 김치찌개 어때?"
example_summary = "점심 메뉴 상의"

# Few-shot 요약
summary = api.summarize(
    dialogue=dialogue,
    example_dialogue=example_dialogue,
    example_summary=example_summary
)
```

---

### 4. 배치 요약

```python
import pandas as pd

# 테스트 데이터 로드
test_df = pd.read_csv("data/raw/test.csv")
dialogues = test_df['dialogue'].tolist()

# Few-shot 예시 (학습 데이터에서 선택)
train_df = pd.read_csv("data/raw/train.csv")
example_dialogue = train_df['dialogue'].iloc[0]
example_summary = train_df['summary'].iloc[0]

# 배치 요약
summaries = api.summarize_batch(
    dialogues=dialogues,
    example_dialogue=example_dialogue,
    example_summary=example_summary,
    batch_size=10,     # 배치당 10개
    delay=1.0          # 1초 대기 (Rate limit)
)

# 결과 저장
output_df = pd.DataFrame({
    'fname': test_df['fname'],
    'summary': summaries
})
output_df.to_csv("submissions/solar_submission.csv", index=False)
```

---

## 🔧 Solar API 실행 명령어

### Solar API 추론 스크립트 (예시)

**파일:** `scripts/inference_solar.py`

```python
import argparse
import pandas as pd
from src.api import SolarAPI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="data/raw/test.csv")
    parser.add_argument("--train_data", default="data/raw/train.csv")
    parser.add_argument("--output", default="submissions/solar.csv")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--token_limit", type=int, default=512)
    args = parser.parse_args()

    # API 초기화
    api = SolarAPI(token_limit=args.token_limit)

    # 데이터 로드
    test_df = pd.read_csv(args.test_data)
    train_df = pd.read_csv(args.train_data)

    # Few-shot 예시 선택
    example_dialogue = train_df['dialogue'].iloc[0]
    example_summary = train_df['summary'].iloc[0]

    # 배치 요약
    summaries = api.summarize_batch(
        dialogues=test_df['dialogue'].tolist(),
        example_dialogue=example_dialogue,
        example_summary=example_summary,
        batch_size=args.batch_size
    )

    # 저장
    output_df = pd.DataFrame({
        'fname': test_df['fname'],
        'summary': summaries
    })
    output_df.to_csv(args.output, index=False)

    print(f"Solar API 추론 완료: {args.output}")

if __name__ == "__main__":
    main()
```

**실행:**
```bash
python scripts/inference_solar.py \
    --test_data data/raw/test.csv \
    --output submissions/solar.csv \
    --batch_size 10 \
    --token_limit 512
```

---

# 📌 Part 3: 프롬프트 엔지니어링

## 📝 개요

### 목적
- Solar API 및 LLM의 성능 극대화
- 대화 특성별 최적 프롬프트 자동 선택
- 토큰 사용량 최소화하면서 품질 유지
- 일관된 출력 형식 보장

### 핵심 기능
- ✅ 16개 사전 정의 프롬프트 템플릿
- ✅ 대화 길이/참여자 수/토큰 예산 기반 동적 선택
- ✅ Zero-shot, Few-shot, Chain-of-Thought 지원
- ✅ 토큰 추정 및 압축 템플릿
- ✅ 커스텀 템플릿 추가 지원

---

## 🎯 PromptTemplate 클래스

### 파일 위치
```
src/prompts/template.py
```

### 클래스 구조

```python
@dataclass
class PromptTemplate:
    name: str              # 템플릿 이름
    template: str          # 프롬프트 문자열
    description: str       # 설명
    category: str          # 카테고리
    variables: List[str]   # 필수 변수 목록

    def format(**kwargs) -> str  # 템플릿 포맷팅
```

### 사용 예시

```python
from src.prompts import PromptTemplate

# 템플릿 생성
template = PromptTemplate(
    name="custom_summary",
    template="""다음 대화를 요약해주세요:

{dialogue}

요약 ({style} 스타일):""",
    description="스타일 지정 가능한 템플릿",
    category="custom",
    variables=["dialogue", "style"]
)

# 템플릿 포맷팅
prompt = template.format(
    dialogue="A: 안녕 B: 안녕",
    style="간결한"
)

print(prompt)
# 출력:
# 다음 대화를 요약해주세요:
#
# A: 안녕 B: 안녕
#
# 요약 (간결한 스타일):
```

---

## 📚 PromptLibrary 클래스

### 기능

- 16개 기본 프롬프트 템플릿 관리
- 템플릿 조회, 추가, 분류
- 토큰 수 추정

### 주요 메서드

```python
class PromptLibrary:
    def get_template(name: str) -> PromptTemplate
    def add_template(template: PromptTemplate)
    def list_templates(category: Optional[str]) -> List[str]
    def get_templates_by_category(category: str) -> List[PromptTemplate]
    def estimate_tokens(template_name: str, **kwargs) -> int
```

### 사용 예시

```python
from src.prompts import PromptLibrary

# 라이브러리 생성
library = PromptLibrary()

# 템플릿 조회
template = library.get_template('zero_shot_basic')

# 카테고리별 목록
zero_shot_templates = library.list_templates(category='zero_shot')
print(zero_shot_templates)
# ['zero_shot_basic', 'zero_shot_detailed', 'zero_shot_structured']

# 템플릿 포맷팅
dialogue = "A: 안녕하세요 B: 안녕하세요"
prompt = template.format(dialogue=dialogue)

# 토큰 추정
tokens = library.estimate_tokens('zero_shot_basic', dialogue=dialogue)
print(f"예상 토큰: {tokens}")
# 예상 토큰: 14
```

---

## 🔍 PromptSelector 클래스

### 기능

대화 특성을 분석하여 최적 프롬프트를 자동 선택

### 선택 전략

1. **길이 기반 선택** - 단어 수에 따라
2. **참여자 수 기반 선택** - 2인/소그룹/대규모
3. **토큰 예산 기반 선택** - 압축 필요 여부
4. **카테고리 기반 선택** - Zero-shot/Few-shot/CoT
5. **적응형 선택** - 종합적 분석

### 주요 메서드

```python
class PromptSelector:
    def select_by_length(dialogue: str) -> PromptTemplate
    def select_by_speakers(dialogue: str) -> PromptTemplate
    def select_by_token_budget(dialogue: str, token_budget: int) -> PromptTemplate
    def select_by_category(category: str, dialogue: str, **kwargs) -> PromptTemplate
    def select_adaptive(dialogue: str, token_budget: int, prefer_category: str) -> PromptTemplate
    def get_selection_info(dialogue: str) -> Dict[str, Any]
```

### 사용 예시

```python
from src.prompts import PromptSelector

selector = PromptSelector()

dialogue = "#Person1#: 안녕하세요 #Person2#: 안녕하세요"

# 1. 길이 기반 선택
template = selector.select_by_length(dialogue)
print(f"길이 기반: {template.name}")
# 길이 기반: short_dialogue

# 2. 참여자 수 기반 선택
template = selector.select_by_speakers(dialogue)
print(f"참여자 수 기반: {template.name}")
# 참여자 수 기반: two_speakers

# 3. 토큰 예산 기반 선택
template = selector.select_by_token_budget(dialogue, token_budget=100)
print(f"토큰 예산 기반: {template.name}")
# 토큰 예산 기반: zero_shot_detailed

# 4. 적응형 선택 (자동 최적화)
template = selector.select_adaptive(
    dialogue=dialogue,
    token_budget=512,
    prefer_category="zero_shot"
)
print(f"적응형: {template.name}")
# 적응형: zero_shot_basic
```

---

## 📝 프롬프트 템플릿 종류

### 1. Zero-shot 템플릿 (3개)

예시 없이 직접 요약하는 템플릿

| 템플릿 이름 | 설명 | 사용 시기 |
|-----------|------|---------|
| zero_shot_basic | 기본 템플릿 | 짧은 대화, 빠른 처리 |
| zero_shot_detailed | 상세 템플릿 | 긴 대화, 품질 중시 |
| zero_shot_structured | 구조화 템플릿 | 일관된 형식 필요 |

**예시: zero_shot_basic**
```
다음 대화를 요약해주세요:

{dialogue}

요약:
```

---

### 2. Few-shot 템플릿 (3개)

예시를 제공하여 학습시키는 템플릿

| 템플릿 이름 | 예시 개수 | 사용 시기 |
|-----------|---------|---------|
| few_shot_1shot | 1개 | 단순 패턴 |
| few_shot_2shot | 2개 | 일반적 상황 |
| few_shot_3shot | 3개 | 복잡한 패턴 |

**예시: few_shot_2shot**
```
대화 요약 예시를 참고하여 마지막 대화를 요약해주세요.

예시 1:
대화: {example1_dialogue}
요약: {example1_summary}

예시 2:
대화: {example2_dialogue}
요약: {example2_summary}

이제 다음 대화를 요약해주세요:
대화: {dialogue}
요약:
```

---

### 3. Chain-of-Thought (CoT) 템플릿 (2개)

단계별 사고 과정을 유도하는 템플릿

| 템플릿 이름 | 설명 | 사용 시기 |
|-----------|------|---------|
| cot_step_by_step | 단계별 템플릿 | 복잡한 대화 |
| cot_analytical | 분석적 템플릿 | 긴 대화 (300+ 단어) |

**예시: cot_step_by_step**
```
다음 대화를 단계별로 분석하여 요약해주세요.

단계 1: 대화의 주요 주제 파악
단계 2: 핵심 정보와 결정사항 추출
단계 3: 부수적 정보 제거
단계 4: 간결한 문장으로 정리

대화:
{dialogue}

최종 요약:
```

---

### 4. 대화 길이별 템플릿 (3개)

단어 수에 따라 최적화된 템플릿

| 템플릿 이름 | 길이 범위 | 특징 |
|-----------|---------|------|
| short_dialogue | < 200 단어 | 핵심만 간단히 |
| medium_dialogue | 200-500 단어 | 3-4문장 요약 |
| long_dialogue | > 500 단어 | 주제별 구조화 |

---

### 5. 참여자 수별 템플릿 (3개)

참여자 수에 따라 최적화된 템플릿

| 템플릿 이름 | 참여자 수 | 초점 |
|-----------|---------|------|
| two_speakers | 2명 | 각자의 입장과 합의점 |
| group_small | 3-4명 | 주요 의견과 결론 |
| group_large | 5명 이상 | 핵심 주제와 결정사항 |

---

### 6. 압축 템플릿 (2개)

토큰 절약을 위한 최소화 템플릿

| 템플릿 이름 | 압축률 | 사용 시기 |
|-----------|--------|---------|
| compressed_minimal | 최대 | 토큰 80% 이상 사용 |
| compressed_concise | 높음 | 토큰 60-80% 사용 |

**예시: compressed_minimal**
```
{dialogue}

요약:
```

---

## 💻 프롬프트 사용 방법

### 1. 기본 사용 (자동 선택)

```python
from src.prompts import create_prompt_library, create_prompt_selector

# 초기화
library = create_prompt_library()
selector = create_prompt_selector(library)

# 대화 준비
dialogue = "#Person1#: 안녕하세요. 오늘 회의 시간을 정하려고 합니다. #Person2#: 3시는 어떠세요?"

# 자동 선택 및 포맷팅
template = selector.select_adaptive(dialogue)
prompt = template.format(dialogue=dialogue)

print(f"선택된 템플릿: {template.name}")
print(f"프롬프트:\n{prompt}")
```

---

### 2. Solar API와 통합

```python
from src.api import SolarAPI
from src.prompts import create_prompt_selector

# 초기화
api = SolarAPI()
selector = create_prompt_selector()

# 대화 준비
dialogue = "#Person1#: 안녕하세요 #Person2#: 안녕하세요"

# 프롬프트 자동 선택
template = selector.select_adaptive(
    dialogue=dialogue,
    token_budget=512,
    prefer_category="zero_shot"
)

# 프롬프트 생성
prompt = template.format(dialogue=dialogue)

# API 호출
summary = api.summarize(
    dialogue=dialogue,
    custom_prompt=prompt  # 커스텀 프롬프트 사용
)

print(f"요약: {summary}")
```

---

## 🔗 관련 파일

**소스 코드:**
- `src/ensemble/weighted.py` - 가중치 앙상블
- `src/ensemble/voting.py` - 투표 앙상블
- `src/ensemble/manager.py` - 모델 매니저
- `src/ensemble/__init__.py` - 패키지 초기화
- `src/api/solar_api.py` - Solar API 클라이언트
- `src/api/__init__.py` - 패키지 초기화
- `src/prompts/template.py` - PromptTemplate 및 PromptLibrary
- `src/prompts/selector.py` - PromptSelector
- `src/prompts/__init__.py` - 패키지 초기화

**테스트:**
- `src/tests/test_ensemble.py` - 앙상블 테스트
- `src/tests/test_solar_api.py` - Solar API 테스트
- `src/tests/test_prompts.py` - 프롬프트 시스템 테스트

**문서:**
- `docs/모듈화/00_전체_시스템_개요.md` - 시스템 개요
- `docs/모듈화/실행_명령어_총정리.md` - 실행 명령어
- `docs/PRD/09_Solar_API_최적화.md` - PRD 문서
- `docs/PRD/12_다중_모델_앙상블_전략.md` - PRD 문서
- `docs/PRD/15_프롬프트_엔지니어링_전략.md` - PRD 문서
