# 일상 대화 요약 모델 성능 개선

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HS6nBbT4)

**대회**: NIKLuge 2024 - 일상 대화 요약
**상태**: ✅ **대회 종료** (2025-10-15)
**최고 점수**: 47.47점 (Phase 1: LP=0.5)
**개선폭**: +0.35점 (Baseline 47.12점 대비)
**제출**: 12/12 사용 완료

---

## 프로젝트 소개

한국어 일상 대화를 입력받아 핵심 내용을 요약하는 모델을 개발하는 프로젝트입니다. KoBART 기반 Seq2Seq 모델을 baseline으로 시작하여, 체계적인 실험을 통해 성능을 향상시켰습니다.

### 최종 성과

- 🏆 **최고 점수**: 47.47점 (Phase 1: LP=0.5, +0.35점)
- 📊 **총 실험**: Baseline ~ Exp #7까지 12회 제출
- 📈 **개선 효율**: 간단한 변경(LP=0.5)으로 최고 효과
- 📚 **교훈 확립**: Loss Gap 분석, WeightedSampler 함정 발견

### 주요 특징

- ✅ **체계적인 실험 관리**: 한 번에 하나씩 변경, Test set으로만 검증
- ✅ **Loss Gap 분석**: 과적합 조기 탐지 기법 확립
- ✅ **완벽한 문서화**: 재현 가능한 상세 기록 및 교훈 정리
- ✅ **재사용 가능한 프레임워크**: CLI 스크립트 및 모듈화 코드

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

### 완료된 실험 요약

| 실험명 | 설명 | Test 점수 | 변화 | 날짜 | 상태 |
|--------|------|-----------|------|------|------|
| **Baseline** | 공식 코드 재현 | 47.12 | - | 10/12 | ✅ 성공 |
| **Phase 1: LP=0.5** | Length Penalty 최적화 | **47.47** | **+0.35** 🏆 | 10/13 | ✅ **최고** |
| Phase 1: LP=0.3 | LP 추가 실험 | 47.15 | +0.03 | 10/13 | ✅ 성공 |
| Phase 1: LP=0.7 | LP 추가 실험 | 47.22 | +0.10 | 10/13 | ✅ 성공 |
| **Exp #5** | no_repeat_ngram=3 | 47.03 | -0.44 | 10/14 | ❌ 실패 |
| **Exp #6** | Learning Rate 3e-5 | - | - | - | ⏸️ 보류 |
| **Exp #7-A** | 증강 데이터 (가중치 없음) | 47.41 | -0.06 | 10/15 | ✅ 안정 |
| Exp #7-C | 가중치 max=5.0 | - | - | 10/15 | ⏭️ 스킵 |
| **Exp #7-F** | 가중치 조정 (최종) | 46.62 | -0.79 | 10/15 | ❌ **실패** |

**제출 통계**: 12/12 사용 완료 (100%)
**점수 범위**: 46.62~47.47 (0.85점 차이)
**평균 개선**: +0.35점 (Baseline 대비)

**상세 기록**: `docs/EXPERIMENT_LOG.md` 참조

### 핵심 교훈 (Lessons Learned)

#### 1. Loss Gap이 진실 ⭐⭐⭐

**정의**: `Loss Gap = Train Loss - Eval Loss`

**해석**:
- **양수 (+0.15 이상)**: 건강한 학습, 제출 고려 ✅
- **음수 (-)**: 과적합, 제출 금지 ❌

**실험 증거**:
- Exp #7-A: Loss Gap **+0.50** → Test **47.41** ✅
- Exp #7-F: Loss Gap +0.47 → Test 46.62 ❌ (분포 왜곡)

#### 2. WeightedRandomSampler의 함정 ⭐⭐⭐

**문제 패턴**:
```
증강 없는 카테고리 × 가중치 = 같은 샘플 반복
→ 모델 암기 → Test 실패
```

**Exp #7-F 실패 원인**:
- 노동/고용(135개) × 3.70배 = 500회 반복
- Dev ROUGE 36.43% ↑ (높음)
- Test Score 46.62 ↓ (낮음)
- **Dev도 학습 분포 영향 받음**

**교훈**: 증강 없으면 가중치 절대 금지!

#### 3. Dev ROUGE ≠ Test Score ⭐⭐

| 실험 | Dev ROUGE-1 | Test Score | 상관관계 |
|------|-------------|------------|----------|
| Exp #7-A | 36.18% | 47.41 | Dev 낮음, Test 높음 |
| Exp #7-F | 36.43% | 46.62 | Dev 높음, Test 낮음 ❌ |

**교훈**: Test 제출만이 유일한 진실!

#### 4. 단순함의 가치 (KISS 원칙) ⭐⭐⭐

| 접근법 | 복잡도 | 소요시간 | 점수 변화 | 효율성 |
|--------|--------|----------|-----------|--------|
| **LP=0.5** | ⭐ 낮음 | 12초 | **+0.35** | ⭐⭐⭐ 최고 |
| 데이터 증강 | ⭐⭐⭐ 높음 | 3시간 | -0.06 | ⭐ 낮음 |
| 가중치 조정 | ⭐⭐ 중간 | 3시간 | -0.79 | ❌ 역효과 |

**교훈**: 가장 간단한 변경이 가장 큰 효과!

**상세 교훈**: `docs/LESSONS_LEARNED.md` 참조

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

## 대회 결과 및 향후 방향

### ✅ 완료된 작업 (2025-10-12 ~ 10-15)

**Phase 1: Baseline 재현 및 최적화**
- [x] Baseline 재현 (47.12점)
- [x] Length Penalty 최적화 (47.47점, +0.35점) 🏆
- [x] 데이터 증강 프레임워크 구축 (1,009개)
- [x] Loss Gap 분석 기법 확립

**Phase 2: 데이터 증강 실험**
- [x] Exp #7-A: 증강 + 가중치 없음 (47.41점, 안정적)
- [x] Exp #7-C: 가중치 max=5.0 (Loss Gap으로 사전 실패 예측)
- [x] Exp #7-F: 가중치 조정 (46.62점, 실패 교훈)

**Phase 3: 문서화 완료**
- [x] 전체 실험 기록 (EXPERIMENT_LOG.md)
- [x] 최종 리포트 (COMPETITION_FINAL_REPORT.md)
- [x] 교훈 정리 (LESSONS_LEARNED.md)
- [x] 아카이브 가이드 (ARCHIVE.md)

**제출 통계**: 12/12 사용 완료 (100%)
**최고 점수**: 47.47점 (Phase 1: LP=0.5)
**개선폭**: +0.35점 (+0.74%)

### 📋 향후 개선 방향 (Future Work)

**즉시 적용 가능 (High Priority)**:
1. **균등 증강** (Balanced Augmentation) - 예상: +0.8~2.0점
   - 모든 카테고리를 300~500개로 균등화
   - WeightedSampler 없이 자연 분포 학습
2. **Learning Rate 튜닝** (1e-5 → 3e-5) - 예상: +0.5~1.5점
3. **Extended Training** (Epochs 20 → 30) - 예상: +0.3~0.8점

**장기 전략 (Lower Priority)**:
4. **Larger Models** (gogamza/kobart-base-v2, KoT5) - 예상: +1.0~3.0점
5. **Ensemble Methods** - 예상: +0.3~1.0점

**상세 계획**: `docs/NEXT_STEPS.md` 참조

---

## 최고 성능 모델 재현

### 🏆 Phase 1: LP=0.5 (47.47점)

**설정**:
```yaml
model: digit82/kobart-summarization
length_penalty: 0.5  # Baseline: 1.0
num_beams: 4
no_repeat_ngram_size: 2
learning_rate: 1e-5
batch_size: 50/32
epochs: 20
```

**재현 방법**:
```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# 1. config.yaml 수정: length_penalty=0.5
# 2. 추론 실행
python inference.py --checkpoint checkpoint-XXXX
```

### 2위: Exp #7-A (47.41점)

**설정**:
- 증강 데이터: 13,465개 (원본 12,457 + 증강 1,009)
- WeightedSampler: **사용 안 함** (핵심!)
- Loss Gap: +0.50 (안정적)

**재현 방법**:
```bash
python inference.py --experiment exp7a --checkpoint checkpoint-2068
```

**Checkpoint 경로**: `submission_exp7a/checkpoint-2068/` (1.4GB)

---

## 기술 스택

- **언어**: Python 3.10
- **프레임워크**: PyTorch 2.5.1, Transformers 4.46.3
- **모델**: KoBART (digit82/kobart-summarization)
- **평가**: ROUGE (rouge 1.0.1)
- **실험 추적**: Weights & Biases (선택적)
- **버전 관리**: Git, GitHub

---

## 참고 문서

### 대회 결과 문서
- [**COMPETITION_FINAL_REPORT.md**](docs/COMPETITION_FINAL_REPORT.md) - 대회 최종 리포트
- [**LESSONS_LEARNED.md**](docs/LESSONS_LEARNED.md) - 실험 교훈 및 Best Practices
- [**EXPERIMENT_LOG.md**](docs/EXPERIMENT_LOG.md) - 전체 실험 상세 기록
- [**ARCHIVE.md**](docs/ARCHIVE.md) - 프로젝트 아카이브 가이드

### 기술 문서
- [RESTART_GUIDE.md](docs/RESTART_GUIDE.md) - 재시작 가이드
- [NEXT_STEPS.md](docs/NEXT_STEPS.md) - 향후 개선 방향
- [code/README.md](code/README.md) - 프레임워크 사용법
- [Competition_Overview/](docs/Competition_Overview/) - 대회 규칙

### 외부 링크
- [KoBART 모델](https://huggingface.co/digit82/kobart-summarization)
- [ROUGE 평가 방법](docs/Competition_Overview/evaluation_method.md)

---

## 라이선스

MIT License

---

**프로젝트 기간**: 2025-10-12 ~ 2025-10-15 (4일)
**최고 점수**: 47.47점 (Phase 1: LP=0.5)
**제출**: 12/12 사용 완료
**상태**: ✅ 대회 종료, 문서화 완료
**마지막 업데이트**: 2025-10-15