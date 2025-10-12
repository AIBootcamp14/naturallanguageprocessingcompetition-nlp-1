# 일상 대화 요약 모델 성능 개선

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HS6nBbT4)

**대회**: NIKLuge 2024 - 일상 대화 요약
**현재 Best Score**: 46.94점 (Baseline)
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

| 실험 | ROUGE-1 | ROUGE-2 | ROUGE-L | Final | 변경사항 | 상태 |
|------|---------|---------|---------|-------|---------|------|
| **Exp #0** | 56.43% | 36.65% | 47.75% | **46.94** | Baseline 재현 | ✅ 성공 |
| **Exp #1** | 52.43% | 32.50% | 43.41% | **42.78** | 증강 데이터 (2배) | ❌ 실패 |

### Exp #1 실패 분석

**원인**:
1. 증강 데이터의 스타일 불일치 (번역투 → 자연스러운 한국어)
2. 원본 데이터 스타일과 충돌
3. 모델 혼란 발생

**교훈**:
- ✅ 데이터 증강 != 무조건 좋음
- ✅ 스타일 일관성이 양보다 중요
- ✅ Dev ROUGE가 낮으면 Test도 낮음

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
- [x] Baseline 재현 (46.94점)
- [x] 실험 로그 시스템
- [x] 코드 모듈화 (7개 모듈, 1,745줄)
- [x] Git 관리

### 🔄 Phase 2: 단계별 성능 개선 (진행 중)
- [ ] Exp #2: Learning Rate 튜닝 (1e-5 → 5e-5)
- [ ] Exp #3: Longer Training (20 → 30 epochs)
- [ ] Exp #4: Generation 파라미터 튜닝
- [ ] Exp #5: 후처리 로직

### 📋 Phase 3: 데이터 증강 (보류)
- Exp #1 실패 후 보류
- 스타일 일관성 유지 방법 연구 필요

### 🎯 Phase 4: 최종 제출
- Best 모델 선택
- 재현성 테스트 (3회 학습, 표준편차 ±0.5)
- 최종 제출 및 문서 정리

---

## 성능 목표

- **현재**: 46.94점 (Baseline)
- **Phase 2 목표**: 48-50점
- **최종 목표**: 50-55점
- **도전 목표**: 55-60점

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