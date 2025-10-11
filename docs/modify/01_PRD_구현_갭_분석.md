# 🔍 PRD 구현 갭 분석 (Gap Analysis)

**작성일**: 2025-10-11
**분석 대상**: `/docs/PRD` 19개 문서 vs 현재 모듈화 코드

---

## 📊 Executive Summary

### 전체 구현률
- **구현된 기능**: 25% (기본 학습/추론만)
- **미구현 기능**: 75% (고급 기능 전부)

### 핵심 문제
**현재 `run_pipeline.py`와 `train.py`는 PRD 14번 "실행 옵션 시스템"에 계획된 기능의 5% 미만만 구현**

```python
# 현재 구현 (run_pipeline.py)
if not args.skip_training:
    train_cmd = [...] # train.py 호출
result = subprocess.run(inference_cmd)  # inference.py 호출
```

**PRD 14번에서 요구한 것**:
```bash
python train.py \
    --mode full \  # ❌ 미구현
    --models all \  # ❌ 미구현
    --k_folds 5 \  # ❌ 미구현
    --ensemble_strategy stacking \  # ❌ 미구현
    --use_tta \  # ❌ 미구현
    --optuna_trials 100 \  # ❌ 미구현
    --use_wandb \  # ⚠️ 부분 구현
    --save_visualizations  # ⚠️ 부분 구현
```

---

## 📋 PRD별 상세 분석

### ✅ PRD 01: 프로젝트 개요
**구현 상태**: 100% (문서 작성 완료)
- 대회 정보, 평가 기준, 데이터셋 구성 모두 문서화됨
- **조치 불필요**

### ✅ PRD 02: 프로젝트 구조
**구현 상태**: 90%
- 디렉토리 구조 대부분 구현됨
- `src/` 모듈화 완료: config, data, models, training, inference, evaluation, logging, utils

**미구현**:
- ❌ `src/api/` (Solar API 래퍼)
- ❌ `src/ensemble/` (앙상블 시스템)
- ❌ `src/optimization/` (Optuna)
- ❌ `src/prompts/` (프롬프트 관리)
- ❌ `src/validation/` (교차검증)

**조치 필요**:
1. 5개 모듈 디렉토리 생성
2. 각 모듈의 `__init__.py` 및 핵심 클래스 구현

### ✅ PRD 03: 브랜치 전략
**구현 상태**: 100% (Git 구조 완료)
- **조치 불필요**

### ⚠️ PRD 04: 성능 개선 전략
**구현 상태**: 10%

**구현된 것**:
- ✅ 기본 데이터 전처리 (노이즈 제거)

**미구현된 것**:
- ❌ LLM 파인튜닝 통합 (train_llm.py는 있지만 train.py와 분리)
- ❌ Solar API 최적화
- ❌ 교차 검증 시스템
- ❌ 데이터 증강 (백트랜슬레이션, 패러프레이징)
- ❌ 앙상블
- ❌ 후처리 최적화

**조치 필요**:
```
우선순위 1 (긴급):
1. train.py에 --mode 옵션 추가
2. LLM 파인튜닝을 train.py에 통합

우선순위 2 (중요):
3. Solar API 클라이언트 구현 (src/api/)
4. K-Fold 시스템 구현 (src/validation/)

우선순위 3 (필요):
5. 데이터 증강 파이프라인 (src/data/augmentation.py)
6. 앙상블 시스템 (src/ensemble/)
```

### ✅ PRD 05: 실험 추적 관리
**구현 상태**: 70%
- ✅ WandB 통합 (부분적)
- ✅ 로깅 시스템 구축
- ❌ MLflow 미구현
- ❌ 실험 명명 규칙 자동화 미흡

**조치 필요**:
1. WandB 로깅 확장 (모든 하이퍼파라미터 자동 기록)
2. 실험명 자동 생성 로직 강화

### ✅ PRD 06: 기술 요구사항
**구현 상태**: 100%
- Python 3.11.9, 필수 라이브러리 모두 설치됨
- **조치 불필요**

### ✅ PRD 07: 리스크 관리
**구현 상태**: 100% (문서 작성 완료)
- **조치 불필요**

---

## 🚨 치명적 미구현: PRD 08-15

### ❌ PRD 08: LLM 파인튜닝 전략
**구현 상태**: 0% (통합 관점)

**현재 상황**:
- `scripts/train_llm.py` 파일은 존재하지만 **완전히 독립적**
- `train.py`에서 호출 불가능
- Encoder-Decoder(KoBART)와 Causal LM(LLM)이 별도 스크립트

**PRD 요구사항**:
```bash
python train.py --mode single --models llama-3.2-korean-3b
python train.py --mode single --models kobart  # 동일한 인터페이스!
```

**조치 필요** (🔥 최우선):
```python
# src/models/model_loader.py 수정 필요
def load_model_and_tokenizer(config, logger=None):
    model_type = config.model.type  # "encoder_decoder" or "causal_lm"

    if model_type == "encoder_decoder":
        return _load_encoder_decoder(config)
    elif model_type == "causal_lm":
        return _load_causal_lm_with_qlora(config)  # ❌ 미구현!
    else:
        raise ValueError(...)
```

**수정 사항**:
1. `src/models/llm_loader.py` 생성 (train_llm.py 코드 이전)
2. `load_model_and_tokenizer()` 함수에 LLM 로딩 로직 통합
3. Config에 `model.type` 필드 추가
4. LLM용 TrainingArguments 설정 분리

**예상 작업량**: 4-6시간

---

### ❌ PRD 09: Solar API 최적화
**구현 상태**: 0%

**미구현 항목**:
- ❌ `src/api/solar_client.py`
- ❌ Few-shot 프롬프트 빌더
- ❌ 토큰 절약 전처리
- ❌ 배치 처리 시스템
- ❌ 캐싱 메커니즘

**조치 필요**:
```python
# src/api/solar_client.py (신규 생성)
class SolarAPIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )

    def build_few_shot_prompt(self, dialogue: str) -> list:
        # PRD 09 참고하여 구현
        pass

    def preprocess_dialogue(self, dialogue: str) -> str:
        # 토큰 70% 절약 전처리
        pass

    def generate_summary(self, dialogue: str) -> str:
        # API 호출
        pass
```

**예상 작업량**: 3-4시간

---

### ❌ PRD 10: 교차 검증 시스템
**구현 상태**: 0%

**미구현 항목**:
- ❌ K-Fold 분할 로직
- ❌ 듀얼 생성 시스템 (모델 + API)
- ❌ 품질 평가기
- ❌ 최적 요약 선택 알고리즘

**조치 필요**:
```python
# src/validation/cross_validator.py (신규 생성)
class KFoldValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.fold_results = []

    def split_data(self, df):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return kf.split(df)

    def train_fold(self, fold_idx, train_data, val_data):
        # 각 폴드 학습
        pass

# src/validation/dual_generator.py (신규 생성)
class DualSummarizationSystem:
    def __init__(self, model, solar_api):
        self.model = model
        self.api = solar_api

    def generate_summaries(self, dialogue):
        model_summary = self.model.generate(dialogue)
        api_summary = self.api.generate_summary(dialogue)
        return self.select_best(model_summary, api_summary)
```

**예상 작업량**: 5-7시간

---

### ⚠️ PRD 11: 로깅 및 모니터링 시스템
**구현 상태**: 60%

**구현된 것**:
- ✅ `src/logging/logger.py`
- ✅ GPU 체크 (`team_gpu_check.py`)
- ✅ 자동 배치 크기 (`auto_batch_size.py`)
- ✅ 기본 시각화 (`src/utils/visualizations/`)

**미구현**:
- ❌ WandB Logger 전용 클래스 (`src/logging/wandb_logger.py`)
- ❌ Notebook Logger (`src/logging/notebook_logger.py`)
- ❌ 7종 시각화 완전 통합

**조치 필요**:
1. `wandb_logger.py` 생성 (현재는 trainer.py에 직접 통합)
2. 시각화 자동 생성 로직 추가

**예상 작업량**: 2-3시간

---

### ❌ PRD 12: 다중 모델 앙상블 전략
**구현 상태**: 0%

**미구현 항목**:
- ❌ `src/ensemble/ensemble_manager.py`
- ❌ Weighted Voting
- ❌ Stacking
- ❌ TTA (Text Test Augmentation)

**조치 필요**:
```python
# src/ensemble/ensemble_manager.py (신규 생성)
class MultiModelEnsemble:
    def __init__(self, model_configs, strategy='weighted_avg'):
        self.models = self._load_models(model_configs)
        self.strategy = strategy

    def predict(self, dialogue, use_tta=False):
        predictions = []
        for model in self.models:
            pred = model.generate(dialogue)
            predictions.append(pred)

        if self.strategy == 'weighted_avg':
            return self._weighted_average(predictions)
        elif self.strategy == 'stacking':
            return self._stacking(predictions)
```

**예상 작업량**: 6-8시간

---

### ❌ PRD 13: Optuna 하이퍼파라미터 최적화
**구현 상태**: 0%

**미구현 항목**:
- ❌ `src/optimization/optuna_tuner.py`
- ❌ 목적 함수
- ❌ 탐색 공간 정의
- ❌ Pruning 전략

**조치 필요**:
```python
# src/optimization/optuna_tuner.py (신규 생성)
import optuna

class OptunaHyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.study = None

    def objective(self, trial):
        # 하이퍼파라미터 샘플링
        lr = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])

        # 모델 학습 및 평가
        score = self._train_and_evaluate(lr, batch_size)
        return score

    def optimize(self, n_trials=100):
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=n_trials)
        return self.study.best_params
```

**예상 작업량**: 5-6시간

---

### ❌ PRD 14: 실행 옵션 시스템 (🔥 최우선 과제)
**구현 상태**: 5%

**현재 train.py**:
```python
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--debug", action="store_true")
```

**PRD 요구사항 (590줄 분량)**:
```python
parser.add_argument('--mode', choices=['single', 'kfold', 'multi_model', 'optuna', 'full'])
parser.add_argument('--models', nargs='+', choices=['solar-10.7b', 'polyglot-ko', 'kullm-v2', ...])
parser.add_argument('--k_folds', type=int, default=5)
parser.add_argument('--ensemble_strategy', choices=['weighted_avg', 'stacking', ...])
parser.add_argument('--use_tta', action='store_true')
parser.add_argument('--optuna_trials', type=int, default=100)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--num_beams', type=int, default=4)
# ... 50개 이상의 옵션
```

**조치 필요** (🔥 가장 중요):
1. `scripts/train.py` 완전 재작성 (PRD 14번 참고)
2. 5가지 모드 구현:
   - `single`: 단일 모델 학습
   - `kfold`: K-Fold 교차 검증
   - `multi_model`: 다중 모델 앙상블
   - `optuna`: 하이퍼파라미터 최적화
   - `full`: 모든 기능 통합
3. Trainer 클래스 분리:
   - `SingleModelTrainer`
   - `KFoldTrainer`
   - `MultiModelEnsembleTrainer`
   - `OptunaOptimizer`
   - `FullPipelineTrainer`

**예상 작업량**: 12-16시간 (가장 큰 작업)

---

### ❌ PRD 15: 프롬프트 엔지니어링 전략
**구현 상태**: 0%

**미구현 항목**:
- ❌ `src/prompts/prompt_manager.py`
- ❌ Few-shot 템플릿 라이브러리
- ❌ Zero-shot 템플릿
- ❌ Chain-of-Thought 프롬프트
- ❌ 동적 프롬프트 선택기

**조치 필요**:
```python
# src/prompts/prompt_manager.py (신규 생성)
class PromptManager:
    def __init__(self):
        self.templates = self._load_templates()

    def get_prompt(self, dialogue, strategy='few_shot'):
        if strategy == 'few_shot':
            return self._build_few_shot(dialogue)
        elif strategy == 'zero_shot':
            return self._build_zero_shot(dialogue)
        elif strategy == 'cot':
            return self._build_chain_of_thought(dialogue)

    def _build_few_shot(self, dialogue):
        # PRD 15 템플릿 구현
        pass
```

**예상 작업량**: 4-5시간

---

### ❌ PRD 16: 데이터 품질 검증 시스템
**구현 상태**: 0%

**미구현 항목**:
- ❌ 구조적 검증
- ❌ 의미적 검증
- ❌ 통계적 검증
- ❌ 라벨 일관성 검증

**조치 필요**:
```python
# src/validation/data_quality.py (신규 생성)
class DataQualityValidator:
    def validate_structure(self, df):
        # 필드 체크, 널 값 체크
        pass

    def validate_semantic(self, df):
        # 대화-요약 일치도 체크
        pass

    def detect_outliers(self, df):
        # 이상치 탐지
        pass
```

**예상 작업량**: 3-4시간

---

### ❌ PRD 17: 추론 최적화 전략
**구현 상태**: 0%

**미구현 항목**:
- ❌ ONNX 변환
- ❌ TensorRT 최적화
- ❌ 양자화 (INT8/INT4)
- ❌ 배치 추론 최적화

**조치 필요**:
이 부분은 **선택적 최적화**이므로 우선순위 낮음

**예상 작업량**: 8-10시간 (나중에)

---

### ⚠️ PRD 18: 베이스라인 검증 전략
**구현 상태**: 70%

**구현된 것**:
- ✅ 베이스라인 config 분석 완료
- ✅ 핵심 설정값 적용 (learning_rate=1e-5, batch_size=50 등)

**미구현**:
- ⚠️ 토큰 제거 방식 확인 필요 (공백 치환 vs 삭제)
- ⚠️ no_repeat_ngram_size=2 확인

**조치 필요**:
1. `src/data/preprocessor.py`에서 토큰 제거 방식 점검
2. Config 파일에 `no_repeat_ngram_size=2` 명시

**예상 작업량**: 1시간

---

### ⚠️ PRD 19: Config 설정 전략
**구현 상태**: 40%

**구현된 것**:
- ✅ 기본 config 시스템 (`src/config/`)
- ✅ `load_config()` 함수

**미구현**:
- ❌ 계층적 config 시스템
- ❌ `configs/base/`, `configs/models/`, `configs/strategies/` 구조
- ❌ Config 병합 메커니즘
- ❌ OmegaConf 활용

**현재 구조**:
```
configs/
└── train_config.yaml  # 단일 파일
```

**PRD 요구사항**:
```
configs/
├── base/
│   ├── default.yaml
│   ├── encoder_decoder.yaml
│   └── causal_lm.yaml
├── models/
│   ├── kobart.yaml
│   ├── llama_3.2_3b.yaml
│   └── qwen3_4b.yaml
├── strategies/
│   ├── data_augmentation.yaml
│   ├── ensemble.yaml
│   └── optuna.yaml
└── experiments/
    ├── baseline_kobart.yaml
    └── full_pipeline.yaml
```

**조치 필요**:
1. Config 디렉토리 재구조화
2. `src/config/loader.py` 수정 (OmegaConf 병합 로직)
3. 각 모델별 config 파일 작성

**예상 작업량**: 4-5시간

---

## 📊 구현 우선순위 및 작업량 요약

### 🔥 우선순위 1 (즉시 필요, 24-30시간)
1. **PRD 14: 실행 옵션 시스템** (12-16h)
   - `train.py` 완전 재작성
   - 5가지 모드 구현
   - Trainer 클래스 분리

2. **PRD 08: LLM 파인튜닝 통합** (4-6h)
   - `src/models/llm_loader.py` 생성
   - `load_model_and_tokenizer()` 확장

3. **PRD 10: K-Fold 교차 검증** (5-7h)
   - `src/validation/cross_validator.py` 구현

4. **PRD 19: Config 재구조화** (4-5h)
   - 계층적 config 시스템

### ⚠️ 우선순위 2 (중요, 20-24시간)
5. **PRD 09: Solar API 최적화** (3-4h)
   - `src/api/solar_client.py` 구현

6. **PRD 12: 앙상블 시스템** (6-8h)
   - `src/ensemble/` 모듈 구현

7. **PRD 13: Optuna 최적화** (5-6h)
   - `src/optimization/optuna_tuner.py` 구현

8. **PRD 15: 프롬프트 엔지니어링** (4-5h)
   - `src/prompts/prompt_manager.py` 구현

9. **PRD 11: 로깅 확장** (2-3h)
   - WandB Logger, Notebook Logger

### 📌 우선순위 3 (선택적, 12-15시간)
10. **PRD 16: 데이터 품질 검증** (3-4h)
11. **PRD 18: 베이스라인 검증** (1h)
12. **PRD 17: 추론 최적화** (8-10h) - 나중에

---

## 🎯 전체 구현 로드맵

### Phase 1: 핵심 인프라 구축 (1-2일, 24-30시간)
```
Day 1 (12시간):
- PRD 14: train.py 재작성 (5가지 모드)
- PRD 19: Config 재구조화

Day 2 (12시간):
- PRD 08: LLM 통합
- PRD 10: K-Fold 구현
```

### Phase 2: 고급 기능 통합 (2-3일, 20-24시간)
```
Day 3-4:
- PRD 09: Solar API
- PRD 12: 앙상블
- PRD 13: Optuna
```

### Phase 3: 완성 및 검증 (1일, 12-15시간)
```
Day 5:
- PRD 15: 프롬프트
- PRD 11: 로깅 확장
- PRD 16, 18: 검증
```

---

## 📝 즉시 조치 사항 (오늘)

### 1. 디렉토리 구조 생성
```bash
mkdir -p src/api
mkdir -p src/ensemble
mkdir -p src/optimization
mkdir -p src/prompts
mkdir -p src/validation

touch src/api/__init__.py
touch src/api/solar_client.py
touch src/ensemble/__init__.py
touch src/ensemble/ensemble_manager.py
touch src/optimization/__init__.py
touch src/optimization/optuna_tuner.py
touch src/prompts/__init__.py
touch src/prompts/prompt_manager.py
touch src/validation/__init__.py
touch src/validation/cross_validator.py
touch src/validation/dual_generator.py
touch src/validation/data_quality.py
```

### 2. Config 재구조화
```bash
mkdir -p configs/base
mkdir -p configs/models
mkdir -p configs/strategies
mkdir -p configs/experiments

# 현재 train_config.yaml을 base/encoder_decoder.yaml로 이전
mv configs/train_config.yaml configs/base/encoder_decoder.yaml
```

### 3. train.py 백업 및 재작성 시작
```bash
cp scripts/train.py scripts/train_backup.py
# PRD 14번 참고하여 train.py 재작성
```

---

## 🚨 결론

**현재 모듈화는 PRD의 25%만 구현되어 있습니다.**

가장 큰 문제는:
1. **실행 옵션 시스템이 없음** → 사용자가 원하는 대로 기능 선택 불가
2. **LLM 파인튜닝이 분리됨** → Encoder-Decoder와 Causal LM을 통합 인터페이스로 사용 불가
3. **고급 기능(앙상블, 교차검증, Optuna)이 전무** → 성능 극대화 불가

**권장 조치**:
- 다음 1-2주 동안 우선순위 1과 2의 작업을 집중적으로 수행
- PRD 문서를 참고하여 코드 작성
- 각 기능 구현 후 즉시 테스트

---

**다음 문서**: `02_실행_옵션_시스템_구현_가이드.md`
