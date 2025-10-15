# 🔄 프로젝트 재시작 가이드 (RESTART GUIDE)

**작성일**: 2025-10-12 (최종 업데이트: 2025-10-13)
**목적**: Baseline부터 체계적으로 재구축
**대상**: 다음 세션 에이전트

---

## 🚨 3분 요약 (Executive Summary)

### 현재 상황
- ✅ 5개 모델 파인튜닝 완료 (Dev set: koBART 94.51, Llama 76.30)
- ❌ Test 제출 시 점수 폭락 (koBART 20.50, Llama 27.21)
- ❌ Baseline(47점)보다도 훨씬 낮음
- ❌ 근본 원인 미발견 (CSV 포맷, 토큰 정리 등 모두 시도)

### 결론
**복잡한 개발로 인해 디버깅 불가능 상태**
→ **Baseline부터 단계적으로 재구축 필요**

### 다음 세션에서 할 일
1. ✅ **Baseline 재현** (code/baseline.ipynb 그대로 실행)
2. ✅ **47점 달성 확인** (Test 제출)
3. ✅ **한 가지씩만 개선** (매번 Test 검증)

---

## 🔥 핵심 교훈 (반드시 숙지)

### ❌ 절대 하지 말 것

1. **Dev set 성능만 보고 만족하지 마라**
   - Dev 94점이어도 Test 20점 나올 수 있음
   - **반드시 Test로 검증**

2. **복잡한 모델부터 시작하지 마라**
   - LLM, QLoRA 등은 나중에
   - **Baseline 47점부터**

3. **한 번에 여러 변경하지 마라**
   - 여러 개선을 동시에 적용하면 디버깅 불가능
   - **한 번에 하나씩만**

4. **Test 검증 없이 진행하지 마라**
   - 매 개선마다 반드시 Test 제출
   - **Daily submission 12회 제한 주의**

### ✅ 반드시 할 것

1. **Baseline 먼저 재현**
   - 공식 baseline.ipynb 그대로 실행
   - 수정 금지
   - 47점 확인

2. **단계별 검증**
   - 변경 → Test 제출 → 점수 확인 → 기록
   - 점수 하락 시 즉시 롤백

3. **문서화**
   - 모든 변경사항 기록
   - 점수 변화 추적

---

## 📋 Step-by-Step Baseline 재현 가이드

### Phase 1: 환경 확인 (5분)

**목표**: 기본 환경 점검

```bash
# 1. 디렉토리 확인
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1

# 2. 데이터 확인
ls -lh data/
# 기대: train.csv, dev.csv, test.csv, sample_submission.csv

# 3. Baseline 노트북 확인
ls -lh code/baseline.ipynb
# 기대: 파일 존재

# 4. GPU 확인
nvidia-smi
# 기대: RTX 3090 24GB
```

**검증**:
- [ ] 모든 데이터 파일 존재
- [ ] Baseline 노트북 존재
- [ ] GPU 정상 작동

---

### Phase 2: Baseline 실행 (30-40분)

**목표**: 공식 Baseline 그대로 재현

#### Step 2.1: 노트북 열기

```bash
# Jupyter 실행 (또는 IDE에서 .ipynb 열기)
jupyter notebook code/baseline.ipynb
```

#### Step 2.2: 전체 실행

**주의**:
- 🚨 **어떤 코드도 수정하지 마라**
- 🚨 **config.yaml도 수정하지 마라**
- 🚨 **그대로 실행만 해라**

**실행 순서**:
1. Cell 실행: 패키지 import
2. Cell 실행: Config 설정
3. Cell 실행: 데이터 로드
4. Cell 실행: 모델 학습 (20 epochs, ~20-30분)
5. Cell 실행: 모델 추론 (test.csv)
6. Cell 실행: output.csv 생성

**예상 결과**:
```
checkpoints/: 학습된 모델 체크포인트
prediction/output.csv: Test 예측 결과
```

#### Step 2.3: CSV 검증

```bash
# 생성된 파일 확인
ls -lh prediction/output.csv

# 포맷 확인
head -5 prediction/output.csv
```

**기대 출력**:
```csv
,fname,summary
0,test_0,요약문...
1,test_1,요약문...
```

**검증**:
- [ ] output.csv 생성 완료
- [ ] 499 samples (test_0 ~ test_499)
- [ ] `,fname,summary` 포맷 (index 포함)

---

### Phase 3: Test 제출 (5분)

**목표**: Baseline 47점 달성 확인

#### Step 3.1: 파일 제출

```
대회 플랫폼 → Submit → prediction/output.csv 업로드
```

#### Step 3.2: 결과 확인

**기대 점수**: **46~48 점** (공식 문서: 47.1244)

**결과 기록**:
```
날짜: ____
점수: ____
비고: Baseline 재현
```

#### Step 3.3: 성공/실패 판단

**✅ 성공 (46-48점)**:
→ Phase 4로 진행

**❌ 실패 (46점 미만)**:
→ 🚨 **중단하고 원인 분석**
- Baseline 코드 재확인
- 데이터 파일 확인
- 대회 플랫폼 FAQ 확인

---

## 📦 Phase 1.5: 모듈화 활용 (선택적)

**업데이트**: 2025-10-13

Baseline 재현 후, 모듈화된 버전을 활용하면 실험 관리가 훨씬 쉬워집니다.

### 모듈화 장점

- ✅ **코드 재사용**: 각 실험마다 전체 코드 복사 불필요
- ✅ **명확한 구조**: 기능별로 파일 분리
- ✅ **쉬운 디버깅**: 모듈 단위 테스트
- ✅ **빠른 실험**: Config만 수정하면 됨

---

### Step 1.5.1: 모듈화 Baseline 실행

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code
jupyter notebook baseline_modular.ipynb
```

**전체 셀 실행** → `prediction/output_modular.csv` 생성

**기대 결과**: 46-47점 (원본 baseline과 동일)

---

### Step 1.5.2: 모듈 구조 이해

```
scripts/
├── utils.py              # Config, 시드, 검증
├── data_loader.py        # 데이터 로딩
├── tokenizer_utils.py    # Tokenizer
├── model_utils.py        # 모델 로딩
├── dataset.py            # Dataset 클래스
├── trainer_utils.py      # Trainer 설정
└── inference_utils.py    # 추론
```

각 모듈은 **독립적으로** 수정 가능합니다.

---

### Step 1.5.3: Config 관리

**config.yaml**은 모든 하이퍼파라미터를 관리합니다.

#### Config 수정 예시

```python
from scripts.utils import load_config, save_config

# 1. 로드
config = load_config('./config.yaml')

# 2. 수정
config['training']['learning_rate'] = 5e-5  # 1e-5 → 5e-5
config['training']['num_train_epochs'] = 30  # 20 → 30

# 3. 저장 (새 파일로)
save_config(config, './config_exp2.yaml')
```

---

### Step 1.5.4: 새 실험 시작하기

#### 방법 1: Notebook 복사 (권장)

```bash
# baseline_modular.ipynb를 복사
cp code/baseline_modular.ipynb code/exp2_lr_tuning.ipynb

# Jupyter에서 열기
jupyter notebook code/exp2_lr_tuning.ipynb
```

**Notebook 내에서**:
```python
# Config 수정
config['training']['learning_rate'] = 5e-5
config['wandb']['name'] = 'kobart-lr-5e-5'

# 학습 (나머지는 동일)
trainer = get_trainer(config, model, tokenizer, train_ds, val_ds)
trainer.train()
```

#### 방법 2: Python 스크립트

```python
#!/usr/bin/env python3
"""Experiment #2: Learning Rate 튜닝"""

import sys
sys.path.append('../scripts')

from utils import load_config, get_device, set_seed
from data_loader import Preprocess
# ... (나머지 import)

# Config 로드 및 수정
config = load_config('./config.yaml')
config['training']['learning_rate'] = 5e-5

# Device & Seed
device = get_device()
set_seed(42)

# ... (학습 코드)
```

---

### Step 1.5.5: 실험 워크플로우

```
1. baseline_modular.ipynb 복사
   ↓
2. Config 수정 (한 가지만!)
   ↓
3. 학습 실행
   ↓
4. 추론 & CSV 생성
   ↓
5. CSV 검증
   ↓
6. Test 제출
   ↓
7. experiment_logs.md 업데이트
   ↓
8. Git 커밋
```

**중요**: 한 번에 **하나의 변수만** 수정!

---

### Step 1.5.6: 모듈 수정하기

특정 기능을 개선하고 싶다면, 해당 모듈만 수정하세요.

#### 예시: compute_metrics 개선

**파일**: `scripts/trainer_utils.py`

```python
# 기존 코드
def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    # ... (ROUGE 계산)
    return result

# 개선: BLEU도 추가
def compute_metrics(config, tokenizer, pred):
    from nltk.translate.bleu_score import corpus_bleu

    rouge = Rouge()
    # ... (ROUGE 계산)

    # BLEU 추가
    bleu = corpus_bleu(references, predictions)
    result['bleu'] = bleu

    return result
```

**장점**: 이 수정은 **모든 실험**에 자동 반영됩니다!

---

### Step 1.5.7: CSV 검증 활용

제출 전 항상 검증하세요.

```python
from scripts.utils import validate_csv

# CSV 검증
result = validate_csv('./prediction/output_exp2.csv')

if result['valid']:
    print(f"✅ 유효 (샘플 수: {result['num_samples']})")
else:
    print("❌ 오류:")
    for error in result['errors']:
        print(f"  - {error}")
```

**자동 체크**:
- 파일 존재 여부
- 필수 컬럼 (fname, summary)
- 샘플 수 (499개)
- 빈 값 확인

---

### Step 1.5.8: 실험 비교

여러 실험 결과를 쉽게 비교할 수 있습니다.

```python
import pandas as pd

# 여러 실험 결과 로드
baseline = pd.read_csv('./prediction/output.csv')
exp2 = pd.read_csv('./prediction/output_exp2.csv')

# 일치율 계산
identical = (baseline['summary'] == exp2['summary']).sum()
print(f"동일한 요약문: {identical} / {len(baseline)}")
print(f"일치율: {identical / len(baseline) * 100:.2f}%")
```

---

### 모듈화 체크리스트

- [ ] `baseline_modular.ipynb` 실행 성공 (46-47점)
- [ ] 7개 모듈 구조 이해
- [ ] Config 수정 방법 숙지
- [ ] Notebook 복사로 새 실험 시작
- [ ] CSV 검증 사용
- [ ] `experiment_logs.md` 업데이트

---

### Step 1.5.9: EDA 분석 결과 (2025-10-13 업데이트)

**5개 agents 병렬 분석 완료!**

상세 내용: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/tasks/eda-findings.md`

#### 핵심 발견 요약

1. **후처리 개선** (즉시 적용) ⭐⭐⭐
   - 예상 효과: +0.5~1.2점
   - 소요 시간: 1시간
   - 리스크: ✅ Low
   - 방법: 공백 정규화 + 중복 문장 제거

2. **Learning Rate 튜닝** (최우선) ⭐⭐⭐
   - 예상 효과: +1~2점 (2e-5), +2~3점 (5e-5)
   - 소요 시간: 30분/실험
   - 리스크: ✅ Low (2e-5는 안전)
   - 방법: 1e-5 → 2e-5 → 3e-5 → 5e-5 순차 실험

3. **Special Token 추가** (효과적) ⭐⭐
   - 예상 효과: +0.5~1.5점
   - 소요 시간: 3-4시간
   - 리스크: ⚠️ Medium
   - 방법: #Time#, #Money# 토큰 추가

4. **데이터 증강 재시도** (주의 필요) ⭐
   - 예상 효과: +1~2점
   - 소요 시간: 1-2일
   - 리스크: ⚠️ Medium
   - 방법: 스타일 필터링 또는 LLM 스타일 보존

#### 추천 실행 순서

**Day 1 (오늘)**:
1. 후처리 개선 (Exp #2) → 47.5~48.2 예상

**Day 2**:
2. LR 2e-5 (Exp #3) → 48.5~49.5 예상

**Day 3-4**:
3. LR 3e-5 or Time Token → 49.5~51 예상

**Week 2**:
4. Warmup/Epochs 튜닝 → 51~54 예상

---

**다음 단계**: Phase 2로 진행 (후처리 개선 → LR 튜닝)

---

### Phase 4: 첫 번째 개선 (1-2시간)

**목표**: +1~2점 개선

#### 개선 후보 (한 번에 하나씩만!)

**옵션 1: Learning Rate 튜닝** (추천)
```yaml
# config.yaml
learning_rate: 5e-5  # 기존: 1e-5
```

**예상 효과**: +0.5~1.5점

**옵션 2: Longer Training**
```yaml
num_train_epochs: 30  # 기존: 20
early_stopping_patience: 5  # 기존: 3
```

**예상 효과**: +1~2점

**옵션 3: Batch Size 조정**
```yaml
per_device_train_batch_size: 64  # 기존: 50
```

**예상 효과**: +0.5~1점

#### 실행 절차

1. **한 가지만 변경**
2. **학습 실행**
3. **Test 제출**
4. **점수 비교**

**기록**:
```
변경사항: learning_rate 1e-5 → 5e-5
점수: Baseline 36.12 → 48.35 (+12.23)
판단: ✅ 개선 성공, 유지
```

**판단 기준**:
- +1점 이상: ✅ 유지
- +0.5~1점: ✅ 유지 가능
- 0~0.5점: ⚠️ 재검토
- 음수: ❌ 롤백

---

## 📂 참고 자료

### 파일 위치

```
/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/
├── code/
│   ├── baseline.ipynb          ← 🔥 공식 Baseline
│   └── config.yaml              ← 설정 파일
├── data/
│   ├── train.csv (12,457)
│   ├── dev.csv (499)
│   ├── test.csv (499)
│   └── sample_submission.csv
├── prediction/
│   └── output.csv               ← 생성될 제출 파일
└── docs/
    ├── Competition_Overview/
    └── Competition_Advanced/
```

### 주요 명령어

```bash
# 디렉토리 이동
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1

# 데이터 확인
head data/test.csv
wc -l data/*.csv

# 결과 확인
head prediction/output.csv
wc -l prediction/output.csv

# 디스크 사용량
du -sh / 2>/dev/null
# ⚠️ 150GB 초과 시 서버 초기화
```

### 환경 설정

```yaml
GPU: NVIDIA RTX 3090 (24GB)
CUDA: 12.1
Python: 3.10
PyTorch: 2.5.1

주요 패키지:
- transformers==4.46.3
- rouge==1.0.1
- pandas, torch, tqdm
```

### Baseline 공식 성능

```
공식 문서: 47.1244 (Public Test)
- ROUGE-1: 0.5660
- ROUGE-2: 0.3675
- ROUGE-L: 0.4719
```

---

## 🎯 단계별 로드맵

### 🔥 Phase 1: Baseline 재현 (즉시)
- [ ] 환경 확인
- [ ] baseline.ipynb 그대로 실행
- [ ] Test 제출
- [ ] 46-48점 확인

**예상 소요**: 30-40분
**성공 기준**: 47±1점

---

### 📦 Phase 1.5: 모듈화 활용 (선택적, 30분)
- [ ] baseline_modular.ipynb 실행
- [ ] 7개 모듈 구조 이해
- [ ] Config 수정 방법 학습
- [ ] CSV 검증 함수 사용법 익히기

**예상 소요**: 30분
**성공 기준**: 원본 baseline과 동일한 점수 (46-48점)

---

### ⚡ Phase 2: 단일 개선 (1-2시간)
- [ ] Learning rate 튜닝
- [ ] Test 제출 및 검증
- [ ] 점수 개선 확인 (+1~2점)

**예상 소요**: 1-2시간
**성공 기준**: 48-50점

---

### 📈 Phase 3: 점진적 개선 (1일)
- [ ] Longer training
- [ ] 데이터 전처리 개선
- [ ] Generation 파라미터 튜닝
- [ ] 매번 Test 검증

**예상 소요**: 4-8시간
**성공 기준**: 50-55점

---

### 🚀 Phase 4: 고급 기법 (2-3일)
- [ ] 더 큰 모델 (mBART, KoT5)
- [ ] 앙상블
- [ ] Prompt engineering
- [ ] 매번 Test 검증

**예상 소요**: 2-3일
**성공 기준**: 55-60점

---

## ⚠️ 함정과 주의사항

### 함정 1: Dev Set 맹신
**증상**: Dev 94점인데 Test 20점
**원인**: Overfitting 또는 평가 방식 차이
**해결**: Dev는 참고만, Test로만 검증

### 함정 2: 복잡한 코드
**증상**: 여러 기능을 동시에 추가
**원인**: 디버깅 불가능
**해결**: 한 번에 하나씩만

### 함정 3: Submission 횟수 낭비
**증상**: Daily 12회 제한 소진
**원인**: 검증 없이 무분별 제출
**해결**: 로컬 검증 후 제출

### 함정 4: CSV 포맷 실수
**증상**: 제출 시 낮은 점수
**원인**: index 누락, 인코딩 오류
**해결**: sample_submission.csv 정확히 따르기

### 함정 5: 후처리 "개선"의 함정
**증상**: 후처리를 추가했는데 점수 변화 없거나 하락
**원인**: 모델 출력이 이미 최적화되어 있음
**실제 사례**: Exp #2 (2025-10-13) - 공백 정규화 + 중복 제거 → +0.03점 (무의미)
**해결**:
- Baseline의 단순함을 존중
- Dev set에서 먼저 검증
- 섣부른 "개선"보다는 모델 학습 개선에 집중
- "당연히 좋을 것"이라는 가정은 위험

### 함정 6: Learning Rate 튜닝의 함정
**증상**: LR 2배 증가했는데 Dev는 개선되었지만 Test는 하락
**원인**:
- LR이 과도하게 높아서 Test 일반화 실패
- 모든 checkpoint에서 일관된 하락 발생
- Dev/Test gap(20%p)으로 인해 Dev 개선이 Test 개선을 보장하지 않음
**실제 사례**:
- Exp #3 (2025-10-13): LR 1e-5 → 2e-5 (2배 증가)
- checkpoint-1750: Dev +0.81%p → Test -0.26점
- checkpoint-1000: Dev -0.01%p → Test -0.34점
- **모든 checkpoint 일관된 실패**
**해결**:
- Baseline 하이퍼파라미터를 존중하라 (이미 최적화되어 있을 가능성)
- 보수적으로 접근: 1.2배, 1.5배 등 점진적 증가
- "2배 정도는 괜찮겠지"라는 직관은 위험
- LR 튜닝은 예상보다 훨씬 민감함

### 함정 7: checkpoint 선택의 역설
**증상**: Best eval_loss checkpoint가 오히려 Test 점수가 더 나쁨
**원인**:
- 잘못된 LR로는 어떤 checkpoint도 좋지 않음
- checkpoint 선택의 중요성 < LR 선택의 중요성
- 근본 원인(하이퍼파라미터)을 해결하지 않고 표면적 최적화에만 집중
**실제 사례**:
- Exp #3-v1 (checkpoint-1750, Epoch 7): 46.6919 (-0.26)
- Exp #3-v2 (checkpoint-1000, Epoch 4, Best loss): 46.6089 (-0.34) ← 더 나쁨!
- 가설: "Epoch 7은 overfit, Epoch 4가 더 좋을 것" → 틀림
**해결**:
- 하이퍼파라미터(LR, Warmup 등)를 먼저 올바르게 설정
- 그 다음에 checkpoint 선택 최적화
- "잘못된 방향으로는 어떤 최적화도 무의미"
- 우선순위: LR > Epochs > Warmup > checkpoint

---

## 📝 실험 기록 템플릿

매 실험마다 기록:

```markdown
## 실험 #N: [실험명]

**날짜**: 2025-10-__
**베이스**: Baseline (or 이전 실험)

### 변경사항
- [한 가지만 명시]

### 설정
```yaml
[변경된 파라미터만]
```

### 결과
- Baseline: 36.12
- 현재: 48.35
- **변화**: +12.23 ✅

### 판단
- [유지/롤백/재시도]

### 다음 단계
- [다음에 시도할 것]
```

---

## 🎓 이전 세션 실패 요약

### 시도한 것들

1. **LLM Fine-tuning** (koBART, Llama, Qwen)
   - Dev: koBART 94.51, Llama 76.30
   - Test: koBART 20.50, Llama 27.21
   - **실패**: Dev/Test 격차 너무 큼

2. **CSV 포맷 수정** (index=True)
   - 여전히 낮은 점수
   - **실패**: 포맷 문제 아님

3. **토큰 정리** (<think> 제거)
   - 여전히 낮은 점수
   - **실패**: 토큰 문제 아님

### 결론

**근본 원인 미발견**
- 복잡한 개발로 디버깅 불가능
- Baseline부터 재시작 필요

---

## ✅ 성공 체크리스트

### Baseline 재현
- [ ] baseline.ipynb 그대로 실행
- [ ] output.csv 생성 (499 samples)
- [ ] Test 제출
- [ ] 46-48점 달성

### 첫 번째 개선
- [ ] 한 가지만 변경
- [ ] 학습 및 추론
- [ ] Test 제출
- [ ] +1점 이상 개선

### 지속적 개선
- [ ] 매 실험 기록
- [ ] Test로만 검증
- [ ] 점진적 개선
- [ ] 50점 돌파

---

## 📞 다음 세션 에이전트에게

**할 일**:
1. 이 문서를 처음부터 끝까지 읽어라
2. Phase 1부터 순서대로 진행해라
3. 절대 건너뛰지 마라
4. 한 번에 하나씩만 변경해라
5. 매번 Test로 검증해라

**목표**:
- 단기: Baseline 47점 재현
- 중기: 50점 달성
- 장기: 55-60점

**원칙**:
- Dev는 참고만, Test가 진실
- 한 번에 하나씩만
- 매번 기록
- 점진적 개선

**행운을 빈다!** 🚀

---

**작성자**: Claude Code Agent (2025-10-12 세션, 2025-10-13 업데이트)
**업데이트 내용**: Phase 1.5 모듈화 활용 섹션 추가
**다음 세션**: Baseline부터 체계적으로 재구축
**문서 위치**: `/Competition/NLP/docs/RESTART_GUIDE.md`
