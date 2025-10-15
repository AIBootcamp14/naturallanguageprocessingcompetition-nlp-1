# 일상 대화 요약 모델 성능 개선

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HS6nBbT4)

**대회**: NIKLuge 2024 - 일상 대화 요약
**현재 Best Score**: 47.47점 (Phase 1, LP=0.5)
**최근 실험**: 47.41점 (Exp #7-A, 증강 데이터)
**목표**: 50점 이상 달성

---

## 프로젝트 소개

한국어 일상 대화를 입력받아 핵심 내용을 요약하는 모델을 개발하는 프로젝트입니다. KoBART 기반 Seq2Seq 모델을 baseline으로 시작하여, 점진적인 개선을 통해 성능을 향상시킵니다.

### 주요 특징

- ✅ **체계적인 실험 관리**: 한 번에 하나씩 변경, Test set으로만 검증
- ✅ **코드 모듈화**: 7개 재사용 가능한 Python 모듈 (1,745줄)
- ✅ **완벽한 재현성**: 모든 실험을 Jupyter Notebook으로 기록
- ✅ **자동화된 검증**: CSV 포맷 검증, ROUGE 계산

---

## 프로젝트 구조

```
naturallanguageprocessingcompetition-nlp-1/
├── code/
│   ├── baseline.ipynb                   # 원본 Baseline (47.12점)
│   ├── baseline_modular.ipynb          # 모듈화 Baseline ✨
│   ├── config.yaml                      # 하이퍼파라미터 설정
│   └── requirements.txt
│
├── scripts/                             # 재사용 가능한 모듈 ✨
│   ├── utils.py                        # Config, 시드, CSV 검증
│   ├── data_loader.py                  # 데이터 로딩, Preprocess
│   ├── tokenizer_utils.py              # Tokenizer 설정
│   ├── model_utils.py                  # 모델 로딩 (학습/추론)
│   ├── dataset.py                      # PyTorch Dataset 클래스
│   ├── trainer_utils.py                # Trainer, compute_metrics
│   └── inference_utils.py              # 추론 파이프라인
│
├── data/
│   ├── train.csv                       # 학습 데이터 (12,457개)
│   ├── dev.csv                         # 검증 데이터 (499개)
│   └── test.csv                        # 테스트 데이터 (499개)
│
├── docs/
│   ├── experiment_logs.md              # 실험 기록
│   ├── RESTART_GUIDE.md                # 재시작 전략
│   └── baseline_code_summary.md        # 코드 설명
│
└── tasks/
    └── tasks-prd-*.md                  # Task List
```

---

## 빠른 시작

### 1. 환경 설정

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code
pip install -r requirements.txt
```

### 2. Baseline 실행 (원본)

```bash
jupyter notebook baseline.ipynb
# 모든 셀 실행 → prediction/output.csv 생성
```

**기대 결과**: 46-47점

### 3. 모듈화 Baseline 실행 ✨ NEW

```bash
jupyter notebook baseline_modular.ipynb
# 모든 셀 실행 → prediction/output_modular.csv 생성
```

**장점**:
- 코드 재사용 (실험마다 모듈만 import)
- 명확한 구조 (각 기능이 독립적인 파일)
- 쉬운 디버깅 (모듈 단위 테스트 가능)

---

## 모듈 사용 방법

### 기본 사용

```python
# 1. 모듈 import
import sys
sys.path.append('../scripts')

from utils import load_config, get_device, set_seed
from data_loader import Preprocess
from tokenizer_utils import load_tokenizer
from model_utils import load_model_for_train
from dataset import prepare_train_dataset
from trainer_utils import get_trainer
from inference_utils import run_inference

# 2. Config 로드
config = load_config('./config.yaml')
device = get_device()
set_seed(42)

# 3. Tokenizer
tokenizer = load_tokenizer(
    config['general']['model_name'],
    config['tokenizer']['special_tokens']
)

# 4. 데이터셋
preprocessor = Preprocess(
    bos_token=config['tokenizer']['bos_token'],
    eos_token=config['tokenizer']['eos_token']
)
train_ds, val_ds = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

# 5. 모델 & 학습
model = load_model_for_train(config, tokenizer, device)
trainer = get_trainer(config, model, tokenizer, train_ds, val_ds)
trainer.train()

# 6. 추론
result = run_inference(model, tokenizer, test_loader, config, device, save_path='./output.csv')
```

### 실험 진행 방법

**Experiment #2: Learning Rate 튜닝 예시**

```python
# baseline_modular.ipynb를 복사
cp baseline_modular.ipynb exp2_lr_tuning.ipynb

# config 수정
config['training']['learning_rate'] = 5e-5  # 1e-5 → 5e-5
config['wandb']['name'] = 'kobart-lr-5e-5'

# 학습 & 추론
trainer = get_trainer(config, model, tokenizer, train_ds, val_ds)
trainer.train()

result = run_inference(model, tokenizer, test_loader, config, device,
                      save_path='./prediction/output_exp2.csv')
```

---

## 주요 기능

### 1. 실험 로그 시스템

모든 실험은 `docs/experiment_logs.md`에 자동 기록됩니다.

```markdown
## Experiment #N: 실험 이름

**날짜**: YYYY-MM-DD
**변경사항**: Learning rate 1e-5 → 5e-5

### 설정
(config.yaml 내용)

### 결과
ROUGE-1: XX.XX%
ROUGE-2: XX.XX%
ROUGE-L: XX.XX%
Final Score: XX.XX

### 판단
✅/❌ + 분석

### 다음 단계
...
```

### 2. CSV 검증

```python
from utils import validate_csv

result = validate_csv('./prediction/output.csv')

if not result['valid']:
    print("오류:", result['errors'])
else:
    print(f"✅ 유효한 CSV (샘플 수: {result['num_samples']})")
```

### 3. Config 관리

```python
from utils import load_config, save_config

# 로드
config = load_config('./config.yaml')

# 수정
config['training']['learning_rate'] = 5e-5

# 저장
save_config(config, './config_exp2.yaml')
```

---

## 실험 결과

### Completed Experiments

| Exp # | Description | ROUGE-1 | ROUGE-2 | ROUGE-L | Score | Date | Status |
|-------|-------------|---------|---------|---------|-------|------|--------|
| #0 | Baseline (Original) | 56.43% | 36.65% | 47.75% | **46.9426** | 2025-10-12 | ✅ |
| #0.1 | Baseline (Modular) | 56.28% | 36.65% | 47.93% | **46.9526** | 2025-10-13 | ✅ (+0.01) |
| #1 | Augmented Data (LLM) | 52.43% | 32.50% | 43.41% | **42.7807** | 2025-10-12 | ❌ Failed (-4.16) |
| #2 | Post-processing v2 | 56.31% | 36.65% | 48.00% | **46.9863** | 2025-10-13 | ❌ Rollback (+0.03) |
| #3 | Learning Rate 2e-5 (v1) | 56.19% | 36.32% | 47.57% | **46.6919** | 2025-10-13 | ❌ Failed (-0.26) |
| #3 | Learning Rate 2e-5 (v2) | 55.93% | 36.72% | 47.17% | **46.6089** | 2025-10-13 | ❌ Failed (-0.34) |

### Planned Experiments

| Exp # | Description | Target | Risk | Priority | Status |
|-------|-------------|--------|------|----------|--------|
| #4 | Learning Rate 3e-5 | +1~2 | ✅ Low | Day 3 | ❌ Skipped (LR 방향 잘못됨) |
| #5 | Learning Rate 5e-5 | +2~3 | ⚠️ Medium | Day 4 | ❌ Skipped (LR 방향 잘못됨) |
| #6 | Time Token | +0.5~1 | ⚠️ Medium | Day 4-5 | 📋 Planned |
| #7 | Money Token | +0.3~0.7 | ⚠️ Medium | Day 5-6 | 📋 Planned |
| #8 | Warmup Steps 50/100 | +0.5~1 | ✅ Low | Week 2 | 📋 Planned |
| #9 | Epochs 30 | +0.5~1 | ✅ Low | Week 2 | 📋 Planned |
| #10 | Data Aug (Filtered) | +0.5~1 | ⚠️ Medium | Week 2+ | 📋 Planned |
| #11 | Data Aug (LLM Style) | +1~2 | ⚠️ Medium | Week 2+ | 📋 Planned |

**참고**: 상세 계획은 `tasks/eda-findings.md` 참조

### 실패 실험 분석

#### Exp #1: 증강 데이터 학습

**원인**:
1. 증강 데이터의 스타일 불일치 (번역투 → 자연스러운 한국어)
2. 원본 데이터 스타일과 충돌
3. 모델 혼란 발생

**교훈**:
- ✅ 데이터 증강 != 무조건 좋음
- ✅ 스타일 일관성이 양보다 중요
- ✅ Dev ROUGE가 낮으면 Test도 낮음

**개선 방안** (Agent 3 권장):
- 방안 1: 필터링된 재사용 (스타일 일치 샘플만)
- 방안 2: LLM 기반 스타일 보존 증강
- 상세: `tasks/eda-findings.md` 참조

#### Exp #2: 후처리 개선 (Post-processing v2)

**원인**:
1. 모델 출력이 이미 최적화되어 있었음
2. Baseline의 단순함에는 이유가 있었음
3. Dev set 검증 없이 Test로 직행 → 예측 불가

**교훈**:
- ✅ "당연히 좋을 것"이라는 가정은 위험함
- ✅ 이론적 개선 ≠ 성능 향상 (실증 필수)
- ✅ Dev set 검증 먼저 수행하기
- ✅ Baseline의 단순함을 존중, 모델 학습 개선에 집중

#### Exp #3: Learning Rate 2e-5

**원인**:
1. **LR 2e-5가 과도함** - Baseline 1e-5가 이미 최적
2. **모든 checkpoint에서 일관된 하락** - checkpoint 선택 무관
3. **Dev/Test 괴리 심화** - Dev +0.81%p → Test -0.26점
4. **checkpoint 선택의 역설** - Best loss(ckpt-1000)가 오히려 더 나쁨 (-0.34)

**교훈**:
- ✅ LR 튜닝은 예상보다 **훨씬 민감함**
- ✅ Baseline 하이퍼파라미터에는 이유가 있음
- ✅ 잘못된 LR로는 어떤 checkpoint도 좋지 않음
- ✅ checkpoint 최적화 < LR 선택의 중요성
- ✅ Dev 점수(20%p 격차)로 Test 예측 불가능

**새로운 방향**:
- Epochs 연장 (20 → 30)
- Warmup Steps 조정
- Special Tokens 추가

---

## 트러블슈팅

### 1. Dev/Test 격차 문제

**증상**: Dev set 94점, Test set 20점

**원인**: 과적합 또는 데이터 불일치

**해결책**:
1. Baseline부터 재현 (생략 금지)
2. 한 번에 하나씩만 변경
3. Test로만 검증 (Dev는 참고용)
4. 격차 5점 이내 유지

### 2. CSV 제출 오류

**증상**: 플랫폼에서 형식 오류

**원인**: Index 컬럼 누락

**해결책**:
```python
# ❌ 잘못된 방법
df.to_csv('output.csv', index=False)

# ✅ 올바른 방법 (index 포함)
df.to_csv('output.csv', index=True)
# 또는
validate_csv('output.csv')  # 자동 검증
```

### 3. 특수 토큰 미제거

**증상**: 요약문에 `<s>`, `</s>`, `<pad>` 포함

**해결책**:
```python
from utils import remove_special_tokens

cleaned = remove_special_tokens(
    summaries,
    tokens=['<s>', '</s>', '<usr>', '<pad>']
)
```

---

## 개발 원칙

### 1. 한 번에 하나씩
- 매 실험마다 **하나의 변수만** 변경
- 여러 개 동시 변경 시 원인 파악 불가

### 2. Test로만 검증
- Dev set은 참고용
- **Test 제출로만 최종 판단** (12회/일 제한)

### 3. 모든 것을 기록
- Notebook으로 전체 흐름 보존
- `experiment_logs.md`에 결과 기록
- Git 커밋 메시지에 점수 포함

### 4. 재현 가능성
- 시드 고정 (`set_seed(42)`)
- Config 파일로 설정 관리
- Notebook으로 전체 프로세스 추적

---

## 로드맵

### ✅ Phase 1: Baseline 재현 및 인프라 구축 (완료)
- [x] Baseline 재현 (46.9426점)
- [x] 코드 모듈화 (7개 모듈, 1,745줄)
- [x] 실험 로그 시스템
- [x] Git 관리

### ✅ Phase 1.5: EDA 분석 (완료, 2025-10-13)
- [x] 5개 agents 병렬 분석
- [x] 후처리 개선 방안 도출
- [x] 하이퍼파라미터 우선순위 결정
- [x] Special Token 최적화 방안
- [x] 데이터 증강 재시도 전략
- [x] `tasks/eda-findings.md` 문서화

### 🔄 Week 1 (2025-10-13 ~ 10-19) - 50점 돌파
- [x] **Day 1 (2025-10-13)**: Exp #2 (후처리 v2) → ❌ 실패 (46.99, 변화 없음)
- [x] **Day 1 (2025-10-13)**: Exp #3 (LR 2e-5) → ❌ 실패 (46.69, -0.26)
- [ ] **Day 2**: Exp #9 (Epochs 30) 또는 Warmup 조정 → 47~48점 목표
- [ ] **Day 3-4**: Time Token 또는 다른 안전한 개선
- [ ] **Day 5-7**: 조합 최적화 → 50점 돌파 시도

**목표**: **50점 이상 달성**
**현재 Best**: 46.9526점 (Baseline Modular)
**제출 횟수**: 8/12 사용 (4회 남음)

### 📋 Week 2 (2025-10-20 ~ 10-26) - 52~54점
- [ ] Special Token 추가 (Time/Money)
- [ ] Warmup Steps 조정 (50, 100)
- [ ] Epochs 연장 (30)
- [ ] 고급 하이퍼파라미터 조합

**목표**: **52~54점 달성**

### 🚀 Week 3+ (Long-term) - 55점 이상
- [ ] 데이터 증강 재시도 (필터링/LLM 스타일 보존)
- [ ] Ensemble 기법 (선택적)
- [ ] 더 큰 모델 실험 (mBART, KoT5)

**목표**: **55점 이상**

---

## 성능 목표

- **현재**: 46.9526점 (Baseline Modular)
- **Week 1 목표**: 50점 돌파
- **Week 2 목표**: 52~54점
- **최종 목표**: 55점 이상

### 예상 성과 (EDA 분석 기반)

| Timeline | Target | Key Improvements |
|----------|--------|------------------|
| Day 1 | 48점 | 후처리 개선 |
| Day 2 | 49점 | LR 2e-5 |
| Week 1 | 50점 | LR 튜닝 완료 |
| Week 2 | 52~54점 | Special Token + Warmup |
| Week 3+ | 55점+ | 데이터 증강 재시도 |

---

## 기술 스택

- **언어**: Python 3.10
- **프레임워크**: PyTorch 2.5.1, Transformers 4.46.3
- **모델**: KoBART (digit82/kobart-summarization)
- **평가**: ROUGE (rouge 1.0.1)
- **실험 추적**: Weights & Biases (선택적)
- **버전 관리**: Git, GitHub

---

## 참고 자료

- [대회 페이지](대회 URL)
- [KoBART 모델](https://huggingface.co/digit82/kobart-summarization)
- [ROUGE 평가 방법](docs/Competition_Overview/evaluation_method.md)
- [Baseline 코드 설명](docs/baseline_code_summary.md)
- [재시작 가이드](docs/RESTART_GUIDE.md)

---

## 라이선스

MIT License

---

**마지막 업데이트**: 2025-10-13
**Current Best**: 46.94점 (Baseline)
**Git Commit**: 3ac2b65