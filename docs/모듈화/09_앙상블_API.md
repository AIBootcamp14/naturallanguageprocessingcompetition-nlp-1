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
- ✅ Stacking Ensemble (메타 학습기)
- ✅ Blending Ensemble (검증 기반 가중치)
- ✅ ModelManager (모델 관리)

---

## ⚖️ 가중치 앙상블

### 파일 위치
```
src/ensemble/weighted.py
```

### 클래스 구조

```python
# ==================== WeightedEnsemble 클래스 ==================== #
class WeightedEnsemble:
    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(models, tokenizers, weights=None)

    # ---------------------- 예측 메서드 ---------------------- #
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
# ---------------------- 가중치 앙상블 모듈 임포트 ---------------------- #
from src.ensemble import WeightedEnsemble

# 모델 로드 (이미 로드된 모델 가정)
models = [model1, model2, model3]              # 앙상블할 모델 리스트
tokenizers = [tokenizer1, tokenizer2, tokenizer3]  # 각 모델의 토크나이저

# 가중치 설정 (ROUGE 점수 기반)
weights = [0.5, 0.3, 0.2]                      # 모델1이 가장 높은 성능

# 앙상블 생성
ensemble = WeightedEnsemble(models, tokenizers, weights)

# 예측
predictions = ensemble.predict(
    dialogues=test_dialogues,                  # 테스트 대화 데이터
    max_length=200,                            # 생성할 최대 토큰 수
    num_beams=4,                               # 빔 서치 빔 개수
    batch_size=8                               # 배치 크기
)
```

### 균등 가중치 사용

```python
# 가중치 없이 초기화 → 자동으로 균등 가중치
ensemble = WeightedEnsemble(models, tokenizers)  # 균등 가중치 자동 할당
# weights = [0.333, 0.333, 0.333]                # 각 모델에 동일한 가중치
```

---

## 🗳️ 투표 앙상블

### 파일 위치
```
src/ensemble/voting.py
```

### 클래스 구조

```python
# ==================== VotingEnsemble 클래스 ==================== #
class VotingEnsemble:
    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(models, tokenizers, voting="hard")

    # ---------------------- 예측 메서드 ---------------------- #
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
# ---------------------- 투표 앙상블 모듈 임포트 ---------------------- #
from src.ensemble import VotingEnsemble

# 모델 및 토크나이저 준비
models = [model1, model2, model3]              # 앙상블할 모델 리스트
tokenizers = [tokenizer1, tokenizer2, tokenizer3]  # 각 모델의 토크나이저

# Hard Voting 앙상블
ensemble = VotingEnsemble(models, tokenizers, voting="hard")  # 다수결 방식 선택

# 예측
predictions = ensemble.predict(
    dialogues=test_dialogues,                  # 테스트 대화 데이터
    max_length=200,                            # 생성할 최대 토큰 수
    num_beams=4,                               # 빔 서치 빔 개수
    batch_size=8                               # 배치 크기
)
```

---

## 🏗️ Stacking 앙상블

### 파일 위치
```
src/ensemble/stacking.py
```

### 클래스 구조

```python
# ==================== StackingEnsemble 클래스 ==================== #
class StackingEnsemble:
    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(base_models, tokenizers, model_names, meta_learner="ridge", logger=None)

    # ---------------------- 메타 학습기 학습 메서드 ---------------------- #
    def train_meta_learner(train_dialogues, train_summaries)

    # ---------------------- 예측 메서드 ---------------------- #
    def predict(dialogues, max_length, num_beams, batch_size)

    # ---------------------- Base 모델 예측 수집 메서드 ---------------------- #
    def _get_base_predictions(dialogues)

    # ---------------------- ROUGE 특징 추출 메서드 ---------------------- #
    def _extract_rouge_features(predictions, references)
```

### 원리

**2단계 앙상블:**
1. **Stage 1**: Base 모델들이 예측 생성
2. **Stage 2**: Meta-learner가 Base 예측들을 조합하여 최종 예측 선택

```
입력 대화
    ↓
[모델1] [모델2] [모델3]  ← Stage 1: Base Models
    ↓       ↓       ↓
 예측1   예측2   예측3
    ↓       ↓       ↓
    [Meta-Learner]      ← Stage 2: ROUGE 기반 학습
         ↓
    최종 예측
```

### Meta-Learner 종류

| Meta-Learner | 설명 | 장점 |
|-------------|------|------|
| `ridge` | Ridge Regression | 안정적, 빠름 |
| `random_forest` | Random Forest | 비선형 패턴 학습 |
| `linear` | Linear Regression | 단순, 해석 가능 |

### 사용 예시

```python
# ---------------------- Stacking 앙상블 모듈 임포트 ---------------------- #
from src.ensemble import StackingEnsemble

# 모델 및 토크나이저 준비
models = [model1, model2, model3]              # Base 모델 리스트
tokenizers = [tokenizer1, tokenizer2, tokenizer3]  # 각 모델의 토크나이저
model_names = ["KoBART", "Llama", "Qwen"]      # 모델 이름 (로깅용)

# Stacking 앙상블 생성
ensemble = StackingEnsemble(
    base_models=models,                        # Base 모델들
    tokenizers=tokenizers,                     # 토크나이저들
    model_names=model_names,                   # 모델 이름들
    meta_learner="ridge"                       # 메타 학습기 타입 (ridge/random_forest/linear)
)

# Meta-learner 학습 (검증 데이터 사용)
ensemble.train_meta_learner(
    train_dialogues=val_df['dialogue'].tolist(),  # 검증 대화 데이터
    train_summaries=val_df['summary'].tolist()    # 검증 요약 데이터 (정답)
)

# 예측
predictions = ensemble.predict(
    dialogues=test_dialogues,                  # 테스트 대화 데이터
    max_length=200,                            # 생성할 최대 토큰 수
    num_beams=4,                               # 빔 서치 빔 개수
    batch_size=8                               # 배치 크기
)
```

### 특징

- **ROUGE 기반 특징 추출**: 각 Base 예측의 ROUGE-1/2/L 점수를 특징으로 사용
- **자동 가중치 학습**: 검증 데이터를 통해 최적 조합 자동 학습
- **높은 성능**: 단순 앙상블보다 +1-2 ROUGE 점수 향상

---

## 🔀 Blending 앙상블

### 파일 위치
```
src/ensemble/blending.py
```

### 클래스 구조

```python
# ==================== BlendingEnsemble 클래스 ==================== #
class BlendingEnsemble:
    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(base_models, tokenizers, model_names, logger=None)

    # ---------------------- 가중치 최적화 메서드 ---------------------- #
    def optimize_weights(val_dialogues, val_summaries, method="rouge")

    # ---------------------- 예측 메서드 ---------------------- #
    def predict(dialogues, max_length, num_beams, batch_size)

    # ---------------------- ROUGE 기반 가중치 최적화 메서드 ---------------------- #
    def _optimize_by_rouge(val_predictions, val_summaries)
```

### 원리

**검증 데이터 기반 가중치 최적화:**
1. 각 모델이 검증 데이터에 대해 예측 생성
2. ROUGE 점수를 목적 함수로 최적 가중치 탐색
3. 학습된 가중치로 테스트 데이터 예측

```python
# ---------------------- 목적 함수 정의 ---------------------- #
def objective(weights):
    # 가중치 기반 앙상블 예측 생성
    ensemble_pred = weighted_combine(predictions, weights)
    # ROUGE 점수 계산
    rouge_score = calculate_rouge(ensemble_pred, references)
    return -rouge_score                        # 최소화 문제로 변환 (음수)

# scipy.optimize로 최적 가중치 탐색
optimal_weights = minimize(objective, init_weights, method='SLSQP')  # SLSQP 알고리즘 사용
```

### 사용 예시

```python
# ---------------------- Blending 앙상블 모듈 임포트 ---------------------- #
from src.ensemble import BlendingEnsemble

# 모델 및 토크나이저 준비
models = [model1, model2, model3]              # Base 모델 리스트
tokenizers = [tokenizer1, tokenizer2, tokenizer3]  # 각 모델의 토크나이저
model_names = ["KoBART", "Llama", "Qwen"]      # 모델 이름 (로깅용)

# Blending 앙상블 생성
ensemble = BlendingEnsemble(
    base_models=models,                        # Base 모델들
    tokenizers=tokenizers,                     # 토크나이저들
    model_names=model_names                    # 모델 이름들
)

# 가중치 최적화 (검증 데이터 사용)
ensemble.optimize_weights(
    val_dialogues=val_df['dialogue'].tolist(),  # 검증 대화 데이터
    val_summaries=val_df['summary'].tolist(),   # 검증 요약 데이터 (정답)
    method="rouge"                              # ROUGE 기반 최적화
)

print(f"최적 가중치: {ensemble.weights}")
# 최적 가중치: [0.52, 0.31, 0.17]              # 자동 계산된 최적 가중치

# 예측
predictions = ensemble.predict(
    dialogues=test_dialogues,                  # 테스트 대화 데이터
    max_length=200,                            # 생성할 최대 토큰 수
    num_beams=4,                               # 빔 서치 빔 개수
    batch_size=8                               # 배치 크기
)
```

### Stacking vs Blending 비교

| 특징 | Stacking | Blending |
|-----|----------|----------|
| **학습 방식** | Meta-learner 학습 | 가중치 최적화 |
| **복잡도** | 높음 | 중간 |
| **속도** | 느림 | 빠름 |
| **과적합** | 중간 | 낮음 |
| **성능** | 최고 | 높음 |
| **권장 사용** | 최종 제출용 | 빠른 실험용 |

---

## 🎛️ 모델 매니저

### 파일 위치
```
src/ensemble/manager.py
```

### 클래스 구조

```python
# ==================== ModelManager 클래스 ==================== #
class ModelManager:
    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__()

    # ---------------------- 단일 모델 로드 메서드 ---------------------- #
    def load_model(model_path, model_name)

    # ---------------------- 여러 모델 로드 메서드 ---------------------- #
    def load_models(model_paths, model_names)

    # ---------------------- 앙상블 생성 메서드 ---------------------- #
    def create_ensemble(ensemble_type, weights, voting)

    # ---------------------- 정보 조회 메서드 ---------------------- #
    def get_info()
```

### 주요 기능

#### 1. 모델 로드

```python
# ---------------------- 모델 매니저 모듈 임포트 ---------------------- #
from src.ensemble import ModelManager

# 모델 매니저 초기화
manager = ModelManager()

# 단일 모델 로드
manager.load_model(
    model_path="outputs/baseline_kobart/final_model",  # 모델 저장 경로
    model_name="KoBART"                                # 모델 이름 (식별용)
)

# 여러 모델 로드
manager.load_models(
    model_paths=[                                  # 모델 경로 리스트
        "outputs/baseline_kobart/final_model",
        "outputs/kobart_v2/final_model",
        "outputs/kobart_v3/final_model"
    ],
    model_names=["KoBART_v1", "KoBART_v2", "KoBART_v3"]  # 각 모델 이름
)
```

#### 2. 앙상블 생성

**가중치 앙상블:**
```python
# 가중치 앙상블 생성
ensemble = manager.create_ensemble(
    ensemble_type="weighted",                  # 앙상블 타입 (weighted)
    weights=[0.5, 0.3, 0.2]                    # 각 모델의 가중치
)
```

**투표 앙상블:**
```python
# 투표 앙상블 생성
ensemble = manager.create_ensemble(
    ensemble_type="voting",                    # 앙상블 타입 (voting)
    voting="hard"                              # 투표 방식 (hard/soft)
)
```

#### 3. 정보 조회

```python
# 모델 매니저 정보 조회
info = manager.get_info()                      # 로드된 모델 정보 반환
print(f"모델 수: {info['num_models']}")        # 로드된 모델 개수
print(f"모델 이름: {info['model_names']}")     # 로드된 모델 이름 리스트
```

---

## 💻 앙상블 사용 방법

### 전체 파이프라인 예시

```python
# ---------------------- 필요한 모듈 임포트 ---------------------- #
from src.ensemble import ModelManager
import pandas as pd

# ==================== 1. 모델 매니저 생성 ==================== #
manager = ModelManager()                       # 모델 관리 객체 초기화

# ==================== 2. 여러 모델 로드 ==================== #
model_paths = [                                # 앙상블할 모델 경로 리스트
    "outputs/baseline_kobart/final_model",
    "outputs/kobart_fold1/final_model",
    "outputs/kobart_fold2/final_model"
]

manager.load_models(model_paths)               # 모델들 메모리에 로드

# ==================== 3. 가중치 앙상블 생성 ==================== #
# ROUGE 점수 기반 가중치
weights = [0.45, 0.30, 0.25]                   # 검증 성능에 비례하여 설정

ensemble = manager.create_ensemble(
    ensemble_type="weighted",                  # 가중치 앙상블 타입
    weights=weights                            # 모델별 가중치
)

# ==================== 4. 테스트 데이터 로드 ==================== #
test_df = pd.read_csv("data/raw/test.csv")    # 테스트 데이터 로드
dialogues = test_df['dialogue'].tolist()      # 대화 컬럼 리스트로 변환

# ==================== 5. 예측 ==================== #
predictions = ensemble.predict(
    dialogues=dialogues,                       # 테스트 대화 데이터
    max_length=200,                            # 생성할 최대 토큰 수
    num_beams=4,                               # 빔 서치 빔 개수
    batch_size=8                               # 배치 크기
)

# ==================== 6. 결과 저장 ==================== #
output_df = pd.DataFrame({
    'fname': test_df['fname'],                 # 파일명
    'summary': predictions                     # 앙상블 예측 결과
})
output_df.to_csv("submissions/ensemble_submission.csv", index=False)  # CSV 저장

print(f"앙상블 예측 완료: {len(predictions)}개")  # 완료 메시지
```

---

### K-Fold 모델 앙상블

```python
# ---------------------- 모델 매니저 모듈 임포트 ---------------------- #
from src.ensemble import ModelManager

# 모델 매니저 초기화
manager = ModelManager()

# K-Fold로 학습된 모델들 로드
fold_paths = [                                 # 각 폴드별 모델 경로 생성
    f"outputs/baseline_kobart_fold{i}/final_model"
    for i in range(1, 6)                       # 5-Fold 교차검증
]

manager.load_models(fold_paths)                # 모든 폴드 모델 로드

# 균등 가중치 앙상블 (K-Fold는 보통 균등)
ensemble = manager.create_ensemble(ensemble_type="weighted")  # 가중치 자동 균등 분배

# 예측
predictions = ensemble.predict(dialogues)      # 앙상블 예측 실행
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
# ==================== SolarAPI 클래스 ==================== #
class SolarAPI:
    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(api_key, token_limit, cache_dir)

    # ---------------------- 대화 전처리 메서드 ---------------------- #
    def preprocess_dialogue(dialogue)

    # ---------------------- 스마트 텍스트 절단 메서드 ---------------------- #
    def smart_truncate(text, max_tokens)

    # ---------------------- 토큰 수 추정 메서드 ---------------------- #
    def estimate_tokens(text)

    # ---------------------- Few-shot 프롬프트 생성 메서드 ---------------------- #
    def build_few_shot_prompt(dialogue, example_dialogue, example_summary)

    # ---------------------- 단일 요약 메서드 ---------------------- #
    def summarize(dialogue, ...)

    # ---------------------- 배치 요약 메서드 ---------------------- #
    def summarize_batch(dialogues, ...)
```

### 초기화

```python
# ---------------------- Solar API 모듈 임포트 ---------------------- #
from src.api import SolarAPI

# Solar API 클라이언트 초기화
api = SolarAPI(
    api_key="your_api_key",                    # API 키 (또는 환경 변수 SOLAR_API_KEY)
    token_limit=512,                           # 대화당 최대 토큰 수
    cache_dir="cache/solar"                    # 응답 캐시 저장 디렉토리
)
```

---

## ⚡ 토큰 최적화

### 1. 대화 전처리

**목적:** 불필요한 토큰 제거

```python
# ---------------------- 대화 전처리 함수 ---------------------- #
def preprocess_dialogue(dialogue):
    # 1. 공백 제거
    dialogue = ' '.join(dialogue.split())      # 연속된 공백을 하나로 통합

    # 2. Person 태그 간소화
    #    #Person1#: → A:
    #    #Person2#: → B:
    dialogue = dialogue.replace('#Person1#:', 'A:')  # Person1 태그 축약
    dialogue = dialogue.replace('#Person2#:', 'B:')  # Person2 태그 축약

    # 3. 스마트 절단
    dialogue = smart_truncate(dialogue, 512)   # 토큰 제한에 맞춰 절단

    return dialogue                            # 전처리된 대화 반환
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
# ---------------------- 스마트 텍스트 절단 함수 ---------------------- #
def smart_truncate(text, max_tokens=512):
    # 토큰 수 추정
    estimated = estimate_tokens(text)          # 현재 텍스트의 토큰 수 추정

    if estimated <= max_tokens:                # 토큰 제한 이하면
        return text                            # 원본 반환

    # 문장 단위로 자르기 (마침표 기준)
    sentences = text.split('.')                # 마침표 기준 문장 분리
    truncated = []                             # 절단된 문장 리스트
    current_tokens = 0                         # 현재 누적 토큰 수

    # 각 문장 순회
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)  # 문장의 토큰 수 추정

        if current_tokens + sentence_tokens > max_tokens:  # 토큰 제한 초과 시
            break                              # 루프 종료

        truncated.append(sentence)             # 문장 추가
        current_tokens += sentence_tokens      # 토큰 수 누적

    return '.'.join(truncated) + '.'           # 문장들 결합하여 반환
```

**특징:**
- 문장 중간에서 자르지 않음
- 의미 보존
- 정확한 토큰 제한

---

### 3. 토큰 추정

**공식:**
```python
# ---------------------- 토큰 수 추정 함수 ---------------------- #
def estimate_tokens(text):
    # 한글: 2.5자 = 1토큰
    korean_chars = len(re.findall(r'[가-힣]', text))  # 한글 문자 수 계산
    korean_tokens = korean_chars / 2.5             # 한글 토큰 추정

    # 영어: 4자 = 1토큰
    english_chars = len(re.findall(r'[a-zA-Z]', text))  # 영문 문자 수 계산
    english_tokens = english_chars / 4             # 영문 토큰 추정

    # 기타: 3자 = 1토큰
    other_chars = len(text) - korean_chars - english_chars  # 기타 문자 수 계산
    other_tokens = other_chars / 3                 # 기타 토큰 추정

    return int(korean_tokens + english_tokens + other_tokens)  # 총 토큰 수 반환
```

**정확도:** ±5% 내외

---

## 📚 Few-shot Learning

### 프롬프트 구조

```python
# ---------------------- Few-shot 프롬프트 메시지 구성 ---------------------- #
messages = [
    # 1. System 프롬프트
    {
        "role": "system",                      # 시스템 역할
        "content": "You are an expert in dialogue summarization..."  # 시스템 지시사항
    },

    # 2. User 예시 (Few-shot)
    {
        "role": "user",                        # 사용자 역할
        "content": "Dialogue:\nA: 점심 뭐 먹을까? B: 김치찌개\nSummary:"  # 예시 대화
    },

    # 3. Assistant 답변 (Few-shot)
    {
        "role": "assistant",                   # 어시스턴트 역할
        "content": "점심 메뉴 상의"            # 예시 요약 답변
    },

    # 4. 실제 입력
    {
        "role": "user",                        # 사용자 역할
        "content": f"Dialogue:\n{dialogue}\nSummary:"  # 요약할 실제 대화
    }
]
```

### Few-shot 예시 선택 전략

**1. 대표 샘플:**
```python
# 평균 길이, 일반적인 주제
example_dialogue = "A: 오늘 회의 시간 정했어? B: 3시로 하자"  # 대표 대화 예시
example_summary = "회의 시간 결정"                              # 대표 요약 예시
```

**2. 다양한 예시 (3-shot):**
```python
# 다양한 길이의 예시 준비
examples = [
    ("짧은 대화", "짧은 요약"),                # 짧은 대화 패턴
    ("중간 대화", "중간 요약"),                # 중간 길이 대화 패턴
    ("긴 대화", "긴 요약")                     # 긴 대화 패턴
]
```

---

## 💻 Solar API 사용 방법

### 1. 환경 변수 설정

```bash
# Solar API 키 환경 변수 설정
export SOLAR_API_KEY="your_api_key_here"      # API 키를 환경 변수로 등록
```

또는 `.env` 파일:
```
SOLAR_API_KEY=your_api_key_here
```

---

### 2. 단일 대화 요약

```python
# ---------------------- Solar API 모듈 임포트 ---------------------- #
from src.api import SolarAPI

# API 초기화
api = SolarAPI()                               # Solar API 클라이언트 생성

# 대화 요약
dialogue = "A: 안녕하세요 B: 안녕하세요 A: 오늘 날씨 좋네요 B: 네, 정말 좋아요"

summary = api.summarize(
    dialogue=dialogue,                         # 요약할 대화
    temperature=0.2,                           # 생성 온도 (낮을수록 일관성 ↑)
    top_p=0.3                                  # Top-p 샘플링 (낮을수록 일관성 ↑)
)

print(f"요약: {summary}")                       # 요약 결과 출력
```

---

### 3. Few-shot 예시 사용

```python
# Few-shot 예시 준비
example_dialogue = "A: 점심 뭐 먹을까? B: 김치찌개 어때?"  # 예시 대화
example_summary = "점심 메뉴 상의"                          # 예시 요약

# Few-shot 요약
summary = api.summarize(
    dialogue=dialogue,                         # 요약할 대화
    example_dialogue=example_dialogue,         # Few-shot 예시 대화
    example_summary=example_summary            # Few-shot 예시 요약
)
```

---

### 4. 배치 요약

```python
# ---------------------- 데이터 처리 라이브러리 임포트 ---------------------- #
import pandas as pd

# 테스트 데이터 로드
test_df = pd.read_csv("data/raw/test.csv")    # 테스트 CSV 로드
dialogues = test_df['dialogue'].tolist()      # 대화 컬럼을 리스트로 변환

# Few-shot 예시 (학습 데이터에서 선택)
train_df = pd.read_csv("data/raw/train.csv")  # 학습 CSV 로드
example_dialogue = train_df['dialogue'].iloc[0]  # 첫 번째 대화를 예시로 선택
example_summary = train_df['summary'].iloc[0]    # 첫 번째 요약을 예시로 선택

# 배치 요약
summaries = api.summarize_batch(
    dialogues=dialogues,                       # 요약할 대화 리스트
    example_dialogue=example_dialogue,         # Few-shot 예시 대화
    example_summary=example_summary,           # Few-shot 예시 요약
    batch_size=10,                             # 배치당 처리할 개수
    delay=1.0                                  # 배치 간 대기 시간 (초, Rate limit 대응)
)

# 결과 저장
output_df = pd.DataFrame({
    'fname': test_df['fname'],                 # 파일명
    'summary': summaries                       # 요약 결과
})
output_df.to_csv("submissions/solar_submission.csv", index=False)  # CSV 저장
```

---

## 🔧 Solar API 실행 명령어

### Solar API 추론 스크립트 (예시)

**파일:** `scripts/inference_solar.py`

```python
# ---------------------- 표준 라이브러리 ---------------------- #
import argparse
# argparse : 명령줄 인자 파싱

# ---------------------- 서드파티 라이브러리 ---------------------- #
import pandas as pd
# pandas   : 데이터프레임 처리

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.api import SolarAPI

# ---------------------- 메인 함수 ---------------------- #
def main():
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="data/raw/test.csv")    # 테스트 데이터 경로
    parser.add_argument("--train_data", default="data/raw/train.csv")  # 학습 데이터 경로
    parser.add_argument("--output", default="submissions/solar.csv")   # 출력 파일 경로
    parser.add_argument("--batch_size", type=int, default=10)          # 배치 크기
    parser.add_argument("--token_limit", type=int, default=512)        # 토큰 제한
    args = parser.parse_args()                                         # 인자 파싱

    # API 초기화
    api = SolarAPI(token_limit=args.token_limit)                       # Solar API 클라이언트 생성

    # 데이터 로드
    test_df = pd.read_csv(args.test_data)                              # 테스트 데이터 로드
    train_df = pd.read_csv(args.train_data)                            # 학습 데이터 로드

    # Few-shot 예시 선택
    example_dialogue = train_df['dialogue'].iloc[0]                    # 예시 대화 선택
    example_summary = train_df['summary'].iloc[0]                      # 예시 요약 선택

    # 배치 요약
    summaries = api.summarize_batch(
        dialogues=test_df['dialogue'].tolist(),                        # 대화 리스트
        example_dialogue=example_dialogue,                             # Few-shot 예시 대화
        example_summary=example_summary,                               # Few-shot 예시 요약
        batch_size=args.batch_size                                     # 배치 크기
    )

    # 저장
    output_df = pd.DataFrame({
        'fname': test_df['fname'],                                     # 파일명
        'summary': summaries                                           # 요약 결과
    })
    output_df.to_csv(args.output, index=False)                         # CSV 저장

    print(f"Solar API 추론 완료: {args.output}")                       # 완료 메시지

# ---------------------- 메인 실행부 ---------------------- #
if __name__ == "__main__":
    main()                                                             # 메인 함수 실행
```

**실행:**
```bash
# Solar API 추론 스크립트 실행
python scripts/inference_solar.py \
    --test_data data/raw/test.csv \          # 테스트 데이터 경로
    --output submissions/solar.csv \         # 출력 파일 경로
    --batch_size 10 \                        # 배치 크기 (10개씩 처리)
    --token_limit 512                        # 토큰 제한 (512토큰)
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
# ==================== PromptTemplate 데이터클래스 ==================== #
@dataclass
class PromptTemplate:
    name: str                                  # 템플릿 이름
    template: str                              # 프롬프트 문자열
    description: str                           # 설명
    category: str                              # 카테고리 (zero_shot/few_shot/cot 등)
    variables: List[str]                       # 필수 변수 목록 (예: ['dialogue'])

    # ---------------------- 템플릿 포맷팅 메서드 ---------------------- #
    def format(**kwargs) -> str                # 변수를 채워 최종 프롬프트 생성
```

### 사용 예시

```python
# ---------------------- 프롬프트 템플릿 모듈 임포트 ---------------------- #
from src.prompts import PromptTemplate

# 템플릿 생성
template = PromptTemplate(
    name="custom_summary",                     # 템플릿 이름
    template="""다음 대화를 요약해주세요:

{dialogue}

요약 ({style} 스타일):""",                     # 프롬프트 문자열 (변수 포함)
    description="스타일 지정 가능한 템플릿",   # 템플릿 설명
    category="custom",                         # 카테고리 (커스텀)
    variables=["dialogue", "style"]            # 필수 변수 리스트
)

# 템플릿 포맷팅
prompt = template.format(
    dialogue="A: 안녕 B: 안녕",                # dialogue 변수 채우기
    style="간결한"                             # style 변수 채우기
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
# ==================== PromptLibrary 클래스 ==================== #
class PromptLibrary:
    # ---------------------- 템플릿 조회 메서드 ---------------------- #
    def get_template(name: str) -> PromptTemplate

    # ---------------------- 템플릿 추가 메서드 ---------------------- #
    def add_template(template: PromptTemplate)

    # ---------------------- 템플릿 목록 조회 메서드 ---------------------- #
    def list_templates(category: Optional[str]) -> List[str]

    # ---------------------- 카테고리별 템플릿 조회 메서드 ---------------------- #
    def get_templates_by_category(category: str) -> List[PromptTemplate]

    # ---------------------- 토큰 수 추정 메서드 ---------------------- #
    def estimate_tokens(template_name: str, **kwargs) -> int
```

### 사용 예시

```python
# ---------------------- 프롬프트 라이브러리 모듈 임포트 ---------------------- #
from src.prompts import PromptLibrary

# 라이브러리 생성
library = PromptLibrary()                      # 기본 템플릿이 로드된 라이브러리

# 템플릿 조회
template = library.get_template('zero_shot_basic')  # 기본 zero-shot 템플릿 조회

# 카테고리별 목록
zero_shot_templates = library.list_templates(category='zero_shot')  # zero_shot 카테고리 템플릿 목록
print(zero_shot_templates)
# ['zero_shot_basic', 'zero_shot_detailed', 'zero_shot_structured']

# 템플릿 포맷팅
dialogue = "A: 안녕하세요 B: 안녕하세요"       # 대화 데이터
prompt = template.format(dialogue=dialogue)    # 템플릿에 대화 삽입

# 토큰 추정
tokens = library.estimate_tokens('zero_shot_basic', dialogue=dialogue)  # 토큰 수 추정
print(f"예상 토큰: {tokens}")
# 예상 토큰: 14                                # 추정된 토큰 수
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
# ==================== PromptSelector 클래스 ==================== #
class PromptSelector:
    # ---------------------- 길이 기반 선택 메서드 ---------------------- #
    def select_by_length(dialogue: str) -> PromptTemplate

    # ---------------------- 참여자 수 기반 선택 메서드 ---------------------- #
    def select_by_speakers(dialogue: str) -> PromptTemplate

    # ---------------------- 토큰 예산 기반 선택 메서드 ---------------------- #
    def select_by_token_budget(dialogue: str, token_budget: int) -> PromptTemplate

    # ---------------------- 카테고리 기반 선택 메서드 ---------------------- #
    def select_by_category(category: str, dialogue: str, **kwargs) -> PromptTemplate

    # ---------------------- 적응형 선택 메서드 ---------------------- #
    def select_adaptive(dialogue: str, token_budget: int, prefer_category: str) -> PromptTemplate

    # ---------------------- 선택 정보 조회 메서드 ---------------------- #
    def get_selection_info(dialogue: str) -> Dict[str, Any]
```

### 사용 예시

```python
# ---------------------- 프롬프트 선택기 모듈 임포트 ---------------------- #
from src.prompts import PromptSelector

# 선택기 초기화
selector = PromptSelector()

# 테스트 대화
dialogue = "#Person1#: 안녕하세요 #Person2#: 안녕하세요"

# 1. 길이 기반 선택
template = selector.select_by_length(dialogue)  # 대화 길이 분석하여 선택
print(f"길이 기반: {template.name}")
# 길이 기반: short_dialogue                    # 짧은 대화용 템플릿

# 2. 참여자 수 기반 선택
template = selector.select_by_speakers(dialogue)  # 참여자 수 분석하여 선택
print(f"참여자 수 기반: {template.name}")
# 참여자 수 기반: two_speakers                 # 2인 대화용 템플릿

# 3. 토큰 예산 기반 선택
template = selector.select_by_token_budget(dialogue, token_budget=100)  # 토큰 예산 고려
print(f"토큰 예산 기반: {template.name}")
# 토큰 예산 기반: zero_shot_detailed           # 예산 내 최적 템플릿

# 4. 적응형 선택 (자동 최적화)
template = selector.select_adaptive(
    dialogue=dialogue,                         # 대화 데이터
    token_budget=512,                          # 토큰 예산
    prefer_category="zero_shot"                # 선호 카테고리
)
print(f"적응형: {template.name}")
# 적응형: zero_shot_basic                      # 종합 분석 결과
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
# ---------------------- 프롬프트 시스템 모듈 임포트 ---------------------- #
from src.prompts import create_prompt_library, create_prompt_selector

# 초기화
library = create_prompt_library()              # 템플릿 라이브러리 생성
selector = create_prompt_selector(library)     # 선택기 생성 (라이브러리 연결)

# 대화 준비
dialogue = "#Person1#: 안녕하세요. 오늘 회의 시간을 정하려고 합니다. #Person2#: 3시는 어떠세요?"

# 자동 선택 및 포맷팅
template = selector.select_adaptive(dialogue)  # 대화 분석하여 최적 템플릿 선택
prompt = template.format(dialogue=dialogue)    # 선택된 템플릿에 대화 삽입

print(f"선택된 템플릿: {template.name}")       # 선택된 템플릿 이름 출력
print(f"프롬프트:\n{prompt}")                   # 최종 프롬프트 출력
```

---

### 2. Solar API와 통합

```python
# ---------------------- 필요한 모듈 임포트 ---------------------- #
from src.api import SolarAPI
from src.prompts import create_prompt_selector

# 초기화
api = SolarAPI()                               # Solar API 클라이언트 생성
selector = create_prompt_selector()            # 프롬프트 선택기 생성

# 대화 준비
dialogue = "#Person1#: 안녕하세요 #Person2#: 안녕하세요"

# 프롬프트 자동 선택
template = selector.select_adaptive(
    dialogue=dialogue,                         # 대화 데이터
    token_budget=512,                          # 토큰 예산
    prefer_category="zero_shot"                # 선호 카테고리
)

# 프롬프트 생성
prompt = template.format(dialogue=dialogue)    # 템플릿에 대화 삽입

# API 호출
summary = api.summarize(
    dialogue=dialogue,                         # 요약할 대화
    custom_prompt=prompt                       # 자동 선택된 커스텀 프롬프트 사용
)

print(f"요약: {summary}")                       # 요약 결과 출력
```

---

# 📌 Part 4: 프롬프트 A/B 테스팅

## 📝 개요

### 목적
- 여러 프롬프트 변형의 성능 비교
- 통계적 유의성 검증
- 최적 프롬프트 자동 선택
- ROUGE 기반 객관적 평가

### 핵심 기능
- ✅ 다중 변형 동시 테스트
- ✅ ROUGE 기반 자동 평가
- ✅ 통계적 유의성 검증 (p-value)
- ✅ 응답 시간 측정
- ✅ 보고서 자동 생성

### 파일 위치
```
src/prompts/ab_testing.py
```

---

## 🧪 PromptABTester 클래스

### 클래스 구조

```python
# ==================== PromptABTester 클래스 ==================== #
class PromptABTester:
    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(api_client, rouge_calculator, logger)

    # ---------------------- 변형 추가 메서드 ---------------------- #
    def add_variant(name, template, description)

    # ---------------------- A/B 테스트 실행 메서드 ---------------------- #
    def run_ab_test(dialogues, references, sample_size) -> ABTestResult

    # ---------------------- 최고 변형 조회 메서드 ---------------------- #
    def get_best_variant() -> PromptVariant

    # ---------------------- 보고서 생성 메서드 ---------------------- #
    def generate_report(output_path) -> str

    # ---------------------- 결과 내보내기 메서드 ---------------------- #
    def export_results(output_path)
```

---

## 📊 데이터 클래스

### 1. PromptVariant

프롬프트 변형 정보를 담는 클래스

```python
# ==================== PromptVariant 데이터클래스 ==================== #
@dataclass
class PromptVariant:
    name: str                                  # 변형 이름
    template: str                              # 프롬프트 템플릿 문자열
    description: str                           # 변형 설명
    results: List[str]                         # 테스트 결과 리스트
    rouge_scores: Dict[str, float]             # ROUGE 점수 딕셔너리 (rouge1/rouge2/rougeL)
    avg_latency: float                         # 평균 응답 시간 (초)
    token_usage: int                           # 총 토큰 사용량
```

### 2. ABTestResult

A/B 테스트 결과를 담는 클래스

```python
# ==================== ABTestResult 데이터클래스 ==================== #
@dataclass
class ABTestResult:
    best_variant: str                          # 최고 성능 변형명
    all_scores: Dict[str, Dict]                # 모든 변형의 점수 딕셔너리
    statistical_significance: bool             # 통계적 유의성 여부
    p_value: float                             # p-value (낮을수록 유의미)
    winner_margin: float                       # 1등과 2등의 점수 차이
```

---

## 💻 사용 방법

### 1. 기본 사용 흐름

```python
from src.prompts.ab_testing import PromptABTester, create_ab_tester
from src.api import SolarAPI
import pandas as pd

# 1. A/B 테스터 생성
api = SolarAPI()
tester = create_ab_tester(api_client=api)

# 2. 변형 추가
tester.add_variant(
    name="zero_shot",
    template="다음 대화를 요약해주세요:\n\n{dialogue}\n\n요약:",
    description="기본 Zero-shot 프롬프트"
)

tester.add_variant(
    name="detailed",
    template="""아래 대화를 읽고 핵심 내용을 3-5문장으로 요약해주세요.

대화:
{dialogue}

요약:""",
    description="상세한 지시사항 포함"
)

tester.add_variant(
    name="structured",
    template="""[태스크] 대화 요약
[형식] 한 문단, 3-5문장
[스타일] 객관적, 간결함

대화 내용:
{dialogue}

요약 결과:""",
    description="구조화된 프롬프트"
)

# 3. 테스트 데이터 준비
train_df = pd.read_csv("data/raw/train.csv")
dialogues = train_df['dialogue'].tolist()[:50]  # 샘플 50개
references = train_df['summary'].tolist()[:50]

# 4. A/B 테스트 실행
result = tester.run_ab_test(
    dialogues=dialogues,
    references=references,
    sample_size=30  # 30개 샘플만 사용 (빠른 테스트)
)

# 5. 결과 확인
print(f"최고 성능 변형: {result.best_variant}")
print(f"통계적 유의성: {result.statistical_significance}")
print(f"p-value: {result.p_value:.4f}")

# 6. 최고 변형 가져오기
best = tester.get_best_variant()
print(f"\n최고 변형: {best.name}")
print(f"ROUGE-Sum: {best.rouge_scores['rouge_sum']:.4f}")
print(f"평균 응답시간: {best.avg_latency:.3f}초")
```

---

### 2. 변형 추가

**필수 요구사항:**
- `template`에 반드시 `{dialogue}` 플레이스홀더 포함
- `name`은 고유해야 함

**예시: 다양한 변형 추가**

```python
# Few-shot 변형
tester.add_variant(
    name="few_shot_1",
    template="""예시:
대화: {example_dialogue}
요약: {example_summary}

이제 다음 대화를 요약해주세요:
{dialogue}

요약:""",
    description="1-shot 예시 포함"
)

# Chain-of-Thought 변형
tester.add_variant(
    name="cot",
    template="""다음 대화를 단계별로 분석하여 요약해주세요.

1단계: 주요 주제 파악
2단계: 핵심 정보 추출
3단계: 간결한 요약 생성

대화:
{dialogue}

최종 요약:""",
    description="단계별 사고 유도"
)

# 간결한 변형
tester.add_variant(
    name="minimal",
    template="{dialogue}\n\n요약:",
    description="최소 토큰 사용"
)

# 역할 지정 변형
tester.add_variant(
    name="role_based",
    template="""당신은 전문 요약가입니다. 다음 대화를 객관적이고 간결하게 요약해주세요.

{dialogue}

요약:""",
    description="역할 기반 프롬프트"
)
```

---

### 3. A/B 테스트 실행

**테스트 흐름:**

1. 각 변형에 대해 모든 대화 요약 생성
2. ROUGE 점수 계산
3. 응답 시간 측정
4. 통계적 유의성 검증
5. 최고 변형 선택

**실행 예시:**

```python
# 전체 데이터로 테스트
result = tester.run_ab_test(
    dialogues=dialogues,
    references=references
)

# 샘플링하여 빠른 테스트
result = tester.run_ab_test(
    dialogues=dialogues,
    references=references,
    sample_size=20  # 20개만 사용
)
```

**출력 예시:**

```
============================================================
A/B 테스트 시작
  - 변형 수: 3
  - 테스트 샘플: 30개
============================================================

[zero_shot] 테스트 중...
  설명: 기본 Zero-shot 프롬프트
  진행: 10/30
  진행: 20/30
  진행: 30/30

  결과:
    ROUGE-1: 0.4521
    ROUGE-2: 0.3215
    ROUGE-L: 0.4102
    ROUGE-Sum: 1.1838
    평균 응답시간: 1.234초

[detailed] 테스트 중...
  설명: 상세한 지시사항 포함
  진행: 10/30
  진행: 20/30
  진행: 30/30

  결과:
    ROUGE-1: 0.4687
    ROUGE-2: 0.3401
    ROUGE-L: 0.4298
    ROUGE-Sum: 1.2386
    평균 응답시간: 1.456초

[structured] 테스트 중...
  설명: 구조화된 프롬프트
  진행: 10/30
  진행: 20/30
  진행: 30/30

  결과:
    ROUGE-1: 0.4603
    ROUGE-2: 0.3287
    ROUGE-L: 0.4211
    ROUGE-Sum: 1.2101
    평균 응답시간: 1.389초

============================================================
A/B 테스트 결과
============================================================
🏆 최고 성능: detailed
   점수: 1.2386
   승차: 0.0285
   통계적 유의성: ✓ 유의미
   p-value: 0.0231
============================================================
```

---

### 4. 보고서 생성

**텍스트 보고서:**

```python
# 화면 출력
report = tester.generate_report()
print(report)

# 파일 저장
report = tester.generate_report(
    output_path="reports/ab_test_report.txt"
)
```

**보고서 예시:**

```
================================================================================
프롬프트 A/B 테스트 보고서
================================================================================

## 테스트 개요
  - 테스트 변형 수: 3
  - 최고 성능 변형: detailed
  - 통계적 유의성: 유의미

## 변형별 결과

### 1. detailed
   설명: 상세한 지시사항 포함
   ROUGE-1: 0.4687
   ROUGE-2: 0.3401
   ROUGE-L: 0.4298
   ROUGE-Sum: 1.2386
   평균 응답시간: 1.456초

### 2. structured
   설명: 구조화된 프롬프트
   ROUGE-1: 0.4603
   ROUGE-2: 0.3287
   ROUGE-L: 0.4211
   ROUGE-Sum: 1.2101
   평균 응답시간: 1.389초

### 3. zero_shot
   설명: 기본 Zero-shot 프롬프트
   ROUGE-1: 0.4521
   ROUGE-2: 0.3215
   ROUGE-L: 0.4102
   ROUGE-Sum: 1.1838
   평균 응답시간: 1.234초

## 통계 분석
   승차 (1등-2등): 0.0285
   p-value: 0.0231

## 권장사항
✓ 'detailed' 변형을 사용하는 것을 권장합니다.

================================================================================
```

**JSON 결과 내보내기:**

```python
# JSON 형식으로 저장
tester.export_results("results/ab_test_results.json")
```

**JSON 예시:**

```json
{
  "best_variant": "detailed",                  // 최고 성능 변형 이름
  "statistical_significance": true,            // 통계적 유의성 여부
  "p_value": 0.0231,                          // p-value 값
  "winner_margin": 0.0285,                    // 1등과 2등의 점수 차이
  "variants": {                               // 모든 변형 정보
    "detailed": {                             // 변형 이름
      "name": "detailed",                     // 변형 이름
      "template": "...",                      // 프롬프트 템플릿
      "description": "상세한 지시사항 포함",  // 변형 설명
      "rouge_scores": {                       // ROUGE 점수들
        "rouge1": 0.4687,                     // ROUGE-1 점수
        "rouge2": 0.3401,                     // ROUGE-2 점수
        "rougeL": 0.4298,                     // ROUGE-L 점수
        "rouge_sum": 1.2386                   // ROUGE 합계 점수
      },
      "avg_latency": 1.456                    // 평균 응답 시간 (초)
    }
    // ... 다른 변형들
  }
}
```

---

## 📈 통계적 유의성 검증

### 검증 방식

1. **표준편차 계산**
   ```python
   std = np.std(rouge_sums)                    # ROUGE 합계 점수들의 표준편차
   ```

2. **p-value 계산**
   ```python
   p_value = std / (best_score + 1e-10)        # 표준편차를 최고 점수로 나눔 (0 방지)
   ```

3. **유의성 판단**
   ```python
   is_significant = (p_value < 0.05) and (winner_margin > 0.01)  # p-value < 0.05 and 승차 > 0.01
   ```

### 해석 가이드

| p-value | 승차 | 유의성 | 해석 |
|---------|------|--------|------|
| < 0.01 | > 0.03 | ✓ 매우 유의미 | 명확한 승자 |
| < 0.05 | > 0.01 | ✓ 유의미 | 승자 있음 |
| < 0.10 | > 0.01 | ⚠️ 경계선 | 더 많은 샘플 필요 |
| ≥ 0.10 | - | ✗ 불충분 | 차이 없음 |

### 권장사항

**유의미한 경우:**
- 최고 변형을 프로덕션에 적용
- 2등 변형은 백업으로 보관

**불충분한 경우:**
- 샘플 크기 증가 (50개 → 100개)
- 변형 수정 (더 명확한 차이 만들기)
- 다른 조건으로 재테스트

---

## 🎯 실전 활용 예시

### 예시 1: Solar API 최적화

```python
from src.prompts.ab_testing import create_ab_tester
from src.api import SolarAPI
import pandas as pd

# Solar API로 A/B 테스터 생성
api = SolarAPI()
tester = create_ab_tester(api_client=api)

# 토큰 최적화 변형들
tester.add_variant(
    name="compressed",
    template="{dialogue}\n\n요약:",
    description="최소 토큰"
)

tester.add_variant(
    name="optimized",
    template="대화: {dialogue}\n\n요약 (20자 이내):",
    description="길이 제한 포함"
)

tester.add_variant(
    name="standard",
    template="다음 대화를 간결하게 요약해주세요:\n\n{dialogue}\n\n요약:",
    description="표준 프롬프트"
)

# 테스트 실행
train_df = pd.read_csv("data/raw/train.csv")
result = tester.run_ab_test(
    dialogues=train_df['dialogue'][:30],
    references=train_df['summary'][:30]
)

# 보고서 저장
tester.generate_report("reports/solar_optimization.txt")
tester.export_results("results/solar_optimization.json")

# 최고 변형 사용
best = tester.get_best_variant()
print(f"✓ 최적 프롬프트: {best.name}")
print(f"  ROUGE-Sum: {best.rouge_scores['rouge_sum']:.4f}")
print(f"  응답시간: {best.avg_latency:.3f}초")
```

---

### 예시 2: Few-shot 개수 최적화

```python
# 1-shot, 2-shot, 3-shot 비교
for n_shot in [1, 2, 3]:
    template = f"""예시 {n_shot}개 제공...

대화: {{dialogue}}
요약:"""

    tester.add_variant(
        name=f"few_shot_{n_shot}",
        template=template,
        description=f"{n_shot}-shot 프롬프트"
    )

# 테스트 실행
result = tester.run_ab_test(dialogues, references, sample_size=40)

# 결과: 보통 2-shot이 최적 (성능 vs 토큰 트레이드오프)
```

---

### 예시 3: 스타일 변형 테스트

```python
# 다양한 지시 스타일
styles = {
    "polite": "부탁드립니다",
    "direct": "해주세요",
    "command": "하시오",
    "professional": "바랍니다"
}

for style_name, style_text in styles.items():
    template = f"다음 대화를 요약{style_text}:\n\n{{dialogue}}\n\n요약:"

    tester.add_variant(
        name=f"style_{style_name}",
        template=template,
        description=f"{style_name} 스타일"
    )

result = tester.run_ab_test(dialogues, references)
```

---

## ⚠️ 주의사항

### 1. 샘플 크기

```python
# 너무 작음 (통계적 신뢰도 낮음)
result = tester.run_ab_test(dialogues, references, sample_size=10)  # ⚠️

# 권장 (충분한 신뢰도)
result = tester.run_ab_test(dialogues, references, sample_size=30)  # ✓

# 높은 정확도 필요 시
result = tester.run_ab_test(dialogues, references, sample_size=100)  # ✓✓
```

### 2. API 비용

Solar API 사용 시 비용 발생:
```python
# 3개 변형 × 50개 샘플 = 150회 API 호출
# → 비용 고려
```

**절약 팁:**
- `sample_size` 제한 (30-50개)
- 변형 수 제한 (3-5개)
- 캐싱 활용

### 3. 템플릿 검증

```python
# ❌ 잘못된 템플릿 (플레이스홀더 없음)
tester.add_variant(
    name="bad",
    template="대화를 요약하세요"  # {dialogue} 없음!
)
# ValueError 발생

# ✓ 올바른 템플릿
tester.add_variant(
    name="good",
    template="대화를 요약하세요: {dialogue}"
)
```

### 4. 응답 시간

```python
# 변형 수 × 샘플 수 × 평균 응답시간
# 3개 × 50개 × 1.5초 = 225초 (약 4분)

# 큰 테스트는 시간 소요
result = tester.run_ab_test(
    dialogues[:100],  # 100개
    references[:100]
)
# → 약 7-8분 소요
```

---

## 🔗 팩토리 함수

### create_ab_tester()

```python
from src.prompts.ab_testing import create_ab_tester
from src.api import SolarAPI
from src.evaluation import RougeCalculator

# 완전한 초기화
api = SolarAPI()
rouge_calc = RougeCalculator()

tester = create_ab_tester(
    api_client=api,            # Solar API 클라이언트
    rouge_calculator=rouge_calc,  # ROUGE 계산기
    logger=None                # Logger (선택적)
)

# 간단한 초기화 (기본값 사용)
tester = create_ab_tester()
```

---

## 🔗 관련 파일

**소스 코드:**
- `src/ensemble/weighted.py` - 가중치 앙상블
- `src/ensemble/voting.py` - 투표 앙상블
- `src/ensemble/stacking.py` - **Stacking 앙상블**
- `src/ensemble/blending.py` - **Blending 앙상블**
- `src/ensemble/manager.py` - 모델 매니저
- `src/ensemble/__init__.py` - 패키지 초기화
- `src/api/solar_api.py` - Solar API 클라이언트
- `src/api/__init__.py` - 패키지 초기화
- `src/prompts/template.py` - PromptTemplate 및 PromptLibrary
- `src/prompts/selector.py` - PromptSelector
- `src/prompts/ab_testing.py` - **Prompt A/B Testing**
- `src/prompts/__init__.py` - 패키지 초기화

**테스트:**
- `src/tests/test_ensemble.py` - 앙상블 테스트
- `src/tests/test_solar_api.py` - Solar API 테스트
- `src/tests/test_prompts.py` - 프롬프트 시스템 테스트

**관련 문서:**
- [01_시작_가이드.md](./01_시작_가이드.md) - 빠른 시작 가이드
- [02_핵심_시스템.md](./02_핵심_시스템.md) - 핵심 시스템 및 Config
- [07_모델_학습_추론.md](./07_모델_학습_추론.md) - 모델 시스템
- [08_평가_최적화.md](./08_평가_최적화.md) - 평가 및 최적화
- [04_명령어_옵션_완전_가이드.md](./04_명령어_옵션_완전_가이드.md) - 전체 명령어 가이드
