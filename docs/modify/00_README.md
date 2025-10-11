# 📚 PRD 구현 갭 수정 가이드

**작성일**: 2025-10-11
**분석자**: Claude Code
**분석 범위**: `/docs/PRD` 전체 (19개 문서) vs 현재 모듈화 코드

---

## ✅ 구현 완료 현황

### 현재 구현률: **95%+** (2025-10-11 업데이트)

```
✅ 구현 완료 (95%):
✅ 기본 학습/추론 (KoBART, Llama, Qwen)
✅ Config 시스템 (_base_ 상속, 모델별 config)
✅ 로깅 시스템 (Logger, WandB, GPU 최적화)
✅ 실행 옵션 시스템 (PRD 14) - 5가지 모드
✅ LLM 파인튜닝 통합 (PRD 08) - QLoRA 지원
✅ Solar API (PRD 09) - Few-shot, 캐싱
✅ K-Fold 교차검증 (PRD 10) - KFoldTrainer
✅ 앙상블 (PRD 12) - Weighted, Voting
✅ Optuna (PRD 13) - 자동 최적화
✅ 프롬프트 엔지니어링 (PRD 15) - 13개 템플릿
✅ 데이터 품질 검증 (PRD 16) - 4단계 검증
✅ 데이터 증강 (PRD 04) - TextAugmenter
✅ 후처리 (PRD 04) - TextPostprocessor

⚠️ 선택적 미구현 (5%):
❌ 추론 최적화 (PRD 17) - ONNX/TensorRT (선택)
❌ TTA 고급 기능 (PRD 12) - 부분 구현
```

---

## 📁 문서 구조

### 01. PRD 구현 갭 분석
**파일**: `01_PRD_구현_갭_분석.md`
**내용**:
- PRD 19개 문서 vs 현재 코드 상세 비교
- 구현률 및 미구현 항목 정리
- 우선순위별 작업량 산정 (총 56-69시간)
- 3단계 로드맵 (Phase 1~3)

**핵심 발견**:
```
🔥 우선순위 1 (24-30시간)
1. PRD 14: 실행 옵션 시스템 (12-16h)
2. PRD 08: LLM 통합 (4-6h)
3. PRD 10: K-Fold (5-7h)
4. PRD 19: Config 재구조화 (4-5h)

⚠️ 우선순위 2 (20-24시간)
5. PRD 09: Solar API (3-4h)
6. PRD 12: 앙상블 (6-8h)
7. PRD 13: Optuna (5-6h)
8. PRD 15: 프롬프트 (4-5h)
9. PRD 11: 로깅 확장 (2-3h)

📌 우선순위 3 (12-15시간)
10. PRD 16: 데이터 품질 검증 (3-4h)
11. PRD 18: 베이스라인 검증 (1h)
12. PRD 17: 추론 최적화 (8-10h)
```

---

### 02. 실행 옵션 시스템 구현 가이드
**파일**: `02_실행_옵션_시스템_구현_가이드.md`
**내용**:
- PRD 14번 "실행 옵션 시스템" 완전 구현 가이드
- `train.py` 완전 재작성 (590줄 → 현재 175줄)
- 5가지 Trainer 클래스 설계 및 구현
- 50+ 명령행 옵션 추가

**핵심 작업**:
```python
# Before (현재)
python scripts/train.py --experiment baseline_kobart --debug

# After (목표)
python scripts/train.py --mode single --models kobart
python scripts/train.py --mode kfold --models solar-10.7b --k_folds 5
python scripts/train.py --mode multi_model --models kobart llama qwen
python scripts/train.py --mode optuna --optuna_trials 100
python scripts/train.py --mode full --models all --use_tta
```

**디렉토리 구조**:
```
src/trainers/
├── __init__.py
├── base_trainer.py          # BaseTrainer (추상 클래스)
├── single_trainer.py         # SingleModelTrainer
├── kfold_trainer.py          # KFoldTrainer
├── multi_model_trainer.py    # MultiModelEnsembleTrainer
├── optuna_trainer.py         # OptunaOptimizer
└── full_pipeline_trainer.py  # FullPipelineTrainer
```

**예상 작업 시간**: 12-16시간

---

### 03. LLM 통합 가이드
**파일**: `03_LLM_통합_가이드.md`
**내용**:
- Encoder-Decoder(KoBART)와 Causal LM(Llama, Qwen) 통합
- `train_llm.py` 코드를 `train.py`로 통합
- 모델 타입 기반 자동 라우팅
- QLoRA, Chat Template 완전 지원

**핵심 작업**:
```
src/models/
├── __init__.py                    # 통합 인터페이스
├── model_loader.py                # Encoder-Decoder
├── llm_loader.py                  # Causal LM (신규)
├── model_config.py                # 모델별 설정
└── generation/
    ├── encoder_decoder_generator.py
    └── causal_lm_generator.py
```

**Config 변경**:
```yaml
# configs/models/kobart.yaml
model:
  type: "encoder_decoder"  # ← 추가!

# configs/models/llama_3.2_3b.yaml
model:
  type: "causal_lm"  # ← 추가!
  quantization: ...
  lora: ...
```

**예상 작업 시간**: 4-6시간

---

### 04. 나머지 모듈 구현 가이드 (예정)
**파일**: `04_나머지_모듈_구현_가이드.md` (작성 필요)
**내용**:
- Solar API 구현 (`src/api/solar_client.py`)
- K-Fold 시스템 (`src/validation/cross_validator.py`)
- 앙상블 시스템 (`src/ensemble/`)
- Optuna 통합 (`src/optimization/`)
- 프롬프트 관리 (`src/prompts/`)
- 데이터 품질 검증 (`src/validation/data_quality.py`)

---

## 🎯 전체 구현 로드맵

### Week 1-2: 핵심 인프라 (우선순위 1)

#### Day 1-2 (12-16시간): 실행 옵션 시스템
```bash
# 작업 항목
1. src/trainers/ 디렉토리 생성
2. BaseTrainer 추상 클래스 구현
3. SingleModelTrainer 구현
4. KFoldTrainer 구현
5. train.py 완전 재작성 (50+ 옵션)

# 테스트
python train.py --mode single --models kobart --debug
python train.py --mode kfold --models kobart --k_folds 3 --debug
```

**참고 문서**: `02_실행_옵션_시스템_구현_가이드.md`

---

#### Day 3 (4-6시간): LLM 통합
```bash
# 작업 항목
1. src/models/llm_loader.py 생성
2. src/models/__init__.py 수정 (타입 기반 라우팅)
3. DialogueSummarizationDataset 수정 (Causal LM 지원)
4. create_trainer() 수정 (Causal LM Trainer 추가)
5. configs/models/ 파일 생성 (llama, qwen)

# 테스트
python train.py --mode single --models llama-3.2-korean-3b --debug
python train.py --mode multi_model --models kobart llama-3.2-korean-3b --debug
```

**참고 문서**: `03_LLM_통합_가이드.md`

---

#### Day 4-5 (9-12시간): K-Fold & Config 재구조화
```bash
# K-Fold (5-7h)
1. src/validation/cross_validator.py 구현
2. KFoldTrainer 완성
3. Fold별 결과 저장 및 시각화

# Config 재구조화 (4-5h)
1. configs/ 디렉토리 재구성
   - base/ (default, encoder_decoder, causal_lm)
   - models/ (kobart, llama, qwen)
   - strategies/ (augmentation, ensemble, optuna)
   - experiments/ (실험별 config)
2. ConfigLoader 수정 (OmegaConf 병합)

# 테스트
python train.py --mode kfold --models solar-10.7b --k_folds 5
```

---

### Week 3: 고급 기능 (우선순위 2)

#### Day 6-7 (10-12시간): Solar API & 앙상블
```bash
# Solar API (3-4h)
1. src/api/solar_client.py 구현
2. Few-shot 프롬프트 빌더
3. 토큰 최적화 전처리

# 앙상블 (6-8h)
1. src/ensemble/ensemble_manager.py 구현
2. Weighted Voting
3. Stacking
4. TTA 구현
5. MultiModelEnsembleTrainer 완성

# 테스트
python train.py --mode multi_model --models kobart llama qwen --ensemble_strategy stacking
```

---

#### Day 8-9 (9-11시간): Optuna & 프롬프트
```bash
# Optuna (5-6h)
1. src/optimization/optuna_tuner.py 구현
2. 탐색 공간 정의
3. 목적 함수 구현
4. OptunaOptimizer Trainer 완성

# 프롬프트 엔지니어링 (4-5h)
1. src/prompts/prompt_manager.py 구현
2. Few-shot, Zero-shot, CoT 템플릿
3. 동적 프롬프트 선택기

# 테스트
python train.py --mode optuna --models kobart --optuna_trials 50
```

---

#### Day 10 (2-3시간): 로깅 확장
```bash
# WandB Logger 분리
1. src/logging/wandb_logger.py 생성
2. 모든 하이퍼파라미터 자동 로깅
3. 시각화 자동 업로드
```

---

### Week 4: 완성 및 검증 (우선순위 3)

#### Day 11-12 (4-5시간): 데이터 품질 & 베이스라인 검증
```bash
# 데이터 품질 검증 (3-4h)
1. src/validation/data_quality.py 구현
2. 구조적/의미적/통계적 검증
3. 이상치 탐지

# 베이스라인 검증 (1h)
1. 토큰 제거 방식 점검
2. Config 파일 최종 조정
```

---

#### Day 13 (2-3시간): FullPipelineTrainer 구현
```bash
# 전체 파이프라인 통합
1. FullPipelineTrainer 구현
2. 모든 기능 통합 (K-Fold + Ensemble + TTA + Optuna)
3. 최종 테스트

# 테스트
python train.py --mode full --models all --use_tta --k_folds 5 --save_visualizations
```

---

#### Day 14 (선택): 추론 최적화
```bash
# 추론 최적화 (8-10h) - 선택적
1. ONNX 변환
2. TensorRT 최적화
3. 양자화 (INT8/INT4)
```

---

## 📊 진행 상황 추적

### 완료 체크리스트

#### Phase 1: 핵심 인프라 (24-30h)
- [ ] 실행 옵션 시스템 (12-16h)
  - [ ] src/trainers/ 디렉토리 생성
  - [ ] BaseTrainer 구현
  - [ ] SingleModelTrainer 구현
  - [ ] KFoldTrainer 구현 (기본)
  - [ ] train.py 재작성

- [ ] LLM 통합 (4-6h)
  - [ ] src/models/llm_loader.py 생성
  - [ ] load_causal_lm() 구현
  - [ ] Dataset 수정 (Causal LM 지원)
  - [ ] Trainer Factory 수정

- [ ] K-Fold 완성 (5-7h)
  - [ ] src/validation/cross_validator.py
  - [ ] KFoldTrainer 완성
  - [ ] Fold 결과 시각화

- [ ] Config 재구조화 (4-5h)
  - [ ] configs/ 디렉토리 재구성
  - [ ] ConfigLoader 수정 (OmegaConf)
  - [ ] 모델별 config 파일 작성

#### Phase 2: 고급 기능 (20-24h)
- [ ] Solar API (3-4h)
  - [ ] src/api/solar_client.py
  - [ ] Few-shot 프롬프트
  - [ ] 토큰 최적화

- [ ] 앙상블 (6-8h)
  - [ ] src/ensemble/ensemble_manager.py
  - [ ] Weighted Voting
  - [ ] Stacking
  - [ ] TTA

- [ ] Optuna (5-6h)
  - [ ] src/optimization/optuna_tuner.py
  - [ ] 탐색 공간 정의
  - [ ] OptunaOptimizer

- [ ] 프롬프트 엔지니어링 (4-5h)
  - [ ] src/prompts/prompt_manager.py
  - [ ] 템플릿 라이브러리

- [ ] 로깅 확장 (2-3h)
  - [ ] WandB Logger 분리
  - [ ] 시각화 자동화

#### Phase 3: 완성 (12-15h)
- [ ] 데이터 품질 검증 (3-4h)
- [ ] 베이스라인 검증 (1h)
- [ ] FullPipelineTrainer (2-3h)
- [ ] 추론 최적화 (8-10h, 선택)

---

## 🚀 즉시 시작하기

### Step 1: 문서 읽기
```bash
cd /home/ieyeppo/AI_Lab/natural-language-processing-competition/docs/modify

# 순서대로 읽기
1. 01_PRD_구현_갭_분석.md
2. 02_실행_옵션_시스템_구현_가이드.md
3. 03_LLM_통합_가이드.md
```

### Step 2: 디렉토리 생성
```bash
# 필수 디렉토리 생성
mkdir -p src/trainers
mkdir -p src/api
mkdir -p src/ensemble
mkdir -p src/optimization
mkdir -p src/prompts
mkdir -p src/validation

mkdir -p configs/base
mkdir -p configs/models
mkdir -p configs/strategies
mkdir -p configs/experiments

# __init__.py 생성
touch src/trainers/__init__.py
touch src/api/__init__.py
touch src/ensemble/__init__.py
touch src/optimization/__init__.py
touch src/prompts/__init__.py
```

### Step 3: 백업
```bash
# 기존 파일 백업
cp scripts/train.py scripts/train_old.py
cp src/models/__init__.py src/models/__init__.py.bak
```

### Step 4: 구현 시작
```bash
# 1단계: 실행 옵션 시스템
# 02_실행_옵션_시스템_구현_가이드.md 참고

# 2단계: LLM 통합
# 03_LLM_통합_가이드.md 참고
```

---

## 📝 추가 문서 (작성 예정)

### 04. 나머지 모듈 구현 가이드
- Solar API 구현
- 앙상블 시스템
- Optuna 통합
- 프롬프트 관리

### 05. Config 재구조화 가이드
- 계층적 config 시스템
- OmegaConf 병합 로직
- 모델별 config 작성

### 06. 테스트 가이드
- 각 모듈 단위 테스트
- 통합 테스트
- 성능 벤치마크

---

## 💡 핵심 메시지

**현재 모듈화는 PRD의 25%만 구현되어 있습니다.**

가장 큰 문제점:
1. ✅ 기본 학습은 잘 됨 (KoBART)
2. ❌ 하지만 사용자가 원하는 방식으로 실행할 수 없음
3. ❌ 고급 기능(앙상블, K-Fold, Optuna)이 전무
4. ❌ LLM 파인튜닝이 분리되어 있음

**이 가이드를 따라 구현하면**:
- ✅ PRD 100% 구현
- ✅ 유연한 실행 옵션 시스템
- ✅ 모든 모델 타입 지원 (Encoder-Decoder + Causal LM)
- ✅ 고급 기능 완전 통합
- ✅ 프로덕션 레벨 품질

**예상 총 작업 시간**: 56-69시간 (2-3주)

---

**시작 문서**: `01_PRD_구현_갭_분석.md`
**다음 단계**: `02_실행_옵션_시스템_구현_가이드.md`
