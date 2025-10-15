# KoBART 대화 요약 모듈화 프레임워크

## 개요

이 프레임워크는 KoBART 대화 요약 프로젝트의 학습/추론 파이프라인을 모듈화하여 실험 관리를 용이하게 합니다.

### 주요 특징

- **통합 설정 관리**: 모든 실험 설정을 `config/experiments.yaml`에서 관리
- **재사용 가능한 모듈**: `core/`와 `utils/` 디렉토리의 모듈화된 코드
- **자동 가중치 샘플링**: 설정에 따라 WeightedRandomSampler 자동 적용
- **간단한 CLI**: 실험 이름과 체크포인트만 지정하면 실행 가능
- **기존 scripts 활용**: `scripts/` 디렉토리의 검증된 유틸리티 재사용

## 디렉토리 구조

```
code/
├── config/
│   └── experiments.yaml          # 모든 실험 설정 통합
├── core/
│   ├── __init__.py
│   ├── data.py                   # 데이터 로딩 및 가중치 샘플링
│   ├── model.py                  # 모델 로드/저장
│   ├── trainer.py                # 학습 로직
│   └── inference.py              # 추론 로직
├── utils/
│   ├── __init__.py
│   ├── config.py                 # Config 파싱
│   ├── logger.py                 # 로깅
│   └── metrics.py                # ROUGE 계산
├── scripts/                      # 기존 유틸리티 (수정 없음)
│   ├── utils.py
│   ├── data_loader.py
│   ├── tokenizer_utils.py
│   ├── model_utils.py
│   ├── dataset.py
│   ├── trainer_utils.py
│   ├── inference_utils.py
│   └── wandb_utils.py
├── train.py                      # 학습 CLI 진입점
└── inference.py                  # 추론 CLI 진입점
```

## 사용법

### 1. 학습

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# Exp #7-A: 가중치 없음 (자연 분포)
python train.py --experiment exp7a

# Exp #7-F: 가중치 샘플링 적용
python train.py --experiment exp7f
```

**학습 과정**:
1. `config/experiments.yaml`에서 실험 설정 자동 로드
2. defaults + 실험별 설정 자동 병합
3. 가중치 샘플링 자동 적용 (use_weights=true인 경우)
4. Wandb 자동 초기화 및 로깅
5. 학습 완료 후 best checkpoint 자동 저장

### 2. 추론

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# Exp #7-A 추론 (기본 출력 경로)
python inference.py --experiment exp7a --checkpoint checkpoint-2068

# Exp #7-F 추론 (사용자 지정 출력 경로)
python inference.py --experiment exp7f --checkpoint checkpoint-1880 --output ./results/submission_7f.csv
```

**추론 과정**:
1. 실험 설정 자동 로드
2. 체크포인트 경로 자동 생성 (`output_dir/checkpoint_name`)
3. Test 데이터 로딩 및 전처리
4. 요약 생성 (beam search + length penalty)
5. 특수 토큰 제거 (`<s>`, `</s>`, `<usr>`, `<pad>`)
6. Competition 형식으로 저장 (index 포함)

## 실험 설정 (config/experiments.yaml)

### 구조

```yaml
defaults:
  # 공통 설정 (모든 실험에 적용)
  general: { ... }
  tokenizer: { ... }
  training: { ... }
  inference: { ... }
  wandb: { ... }

experiments:
  exp7a:
    # exp7a 전용 설정 (defaults를 오버라이드)
    description: "증강 데이터 학습 (가중치 없음)"
    general:
      output_dir: .../submission_exp7a
    data:
      use_weights: false
    wandb:
      name: exp7a-augmented-baseline

  exp7f:
    description: "증강 데이터 학습 (최종 가중치 전략)"
    general:
      output_dir: .../submission_exp7f
    data:
      use_weights: true
      weight_config:
        domain_threshold: 500
        subcluster_threshold: 300
        domain_weights: { ... }
        subcluster_weights: { ... }
    wandb:
      name: exp7f-augmented-final-weights
```

### 새 실험 추가 방법

1. `config/experiments.yaml`에 실험 추가:

```yaml
experiments:
  exp8:
    description: "새로운 실험"
    general:
      output_dir: /Competition/NLP/.../submission_exp8
    training:
      learning_rate: 2.0e-05  # defaults를 오버라이드
      num_train_epochs: 30
    data:
      use_weights: false
    wandb:
      name: exp8-custom-experiment
```

2. 학습 실행:

```bash
python train.py --experiment exp8
```

**끝!** 별도의 `train_exp8.py` 파일이나 `config_exp8.yaml` 불필요.

## 가중치 샘플링

### 자동 적용 조건

`config/experiments.yaml`에서 `data.use_weights: true`로 설정 시 자동 적용:

```yaml
experiments:
  exp7f:
    data:
      use_weights: true
      weight_config:
        domain_threshold: 500        # 500개 이상: 1.0x 유지
        subcluster_threshold: 300    # 300개 이상: 1.0x 유지
        domain_weights:
          '노동/고용': 3.70           # 135개 → 3.70배
          '환경': 2.50                # 200개 → 2.50배
          ...
        subcluster_weights:
          '감정 지원': 2.31           # 130개 → 2.31배
          ...
```

### 가중치 계산 로직

`core/data.py`의 `DataManager._calculate_weights()`:

```python
# 인간관계/일상 도메인: 서브클러스터 가중치 사용
if domain == '인간관계/일상':
    weight = subcluster_weights.get(subcluster, 1.0)
# 기타 도메인: 도메인 가중치 사용
else:
    weight = domain_weights.get(domain, 1.0)
```

이 로직은 `train_exp7f.py`의 검증된 가중치 계산 방식을 그대로 이식한 것입니다.

## 출력 파일

### 학습 출력

```
{output_dir}/
├── checkpoint-xxx/           # 각 에포크별 체크포인트
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
├── runs/                     # TensorBoard 로그
└── trainer_state.json        # Trainer 상태
```

### 추론 출력

```csv
,fname,summary
0,test_0,요약문...
1,test_1,요약문...
...
498,test_498,요약문...
```

- **중요**: 첫 번째 컬럼은 index (`,fname,summary` 헤더)
- Competition 제출 형식에 맞춰 자동 생성

## 기존 코드와의 차이점

### Before (기존 방식)

```bash
# 매 실험마다 파일 복제 필요
cp train_exp7a.py train_exp8.py
cp config_exp7a.yaml config_exp8.yaml

# 파일 내 하드코딩된 경로 수정
vim train_exp8.py  # output_dir, wandb name 등 수정
vim config_exp8.yaml  # 설정 수정

# 실행
python train_exp8.py
```

**문제점**:
- 코드 중복 (20+ 파일)
- 유지보수 어려움 (버그 수정 시 모든 파일 수정)
- 실험 추적 어려움

### After (프레임워크 방식)

```bash
# config/experiments.yaml에 실험 추가만
vim config/experiments.yaml  # exp8 섹션 추가

# 실행
python train.py --experiment exp8
```

**장점**:
- 코드 재사용 (단일 train.py)
- 유지보수 용이 (core/ 모듈만 수정)
- 실험 추적 용이 (experiments.yaml에서 한눈에 확인)

## 검증 방법

### 기존 실험 재현

```bash
# Exp #7-A 재현
python train.py --experiment exp7a
python inference.py --experiment exp7a --checkpoint checkpoint-XXXX

# 기존 결과와 비교
diff submission_exp7a.csv submission_exp7a/checkpoint-2068/output.csv
```

**예상 결과**: 동일한 ROUGE 점수 (차이 < 0.1%)

### 단위 테스트 (선택)

```bash
# Config 로딩 테스트
python -c "from utils.config import load_experiment_config; print(load_experiment_config('./config/experiments.yaml', 'exp7a'))"

# 가중치 계산 테스트
python -c "from core.data import DataManager; dm = DataManager(config, tokenizer); weights = dm._create_weighted_sampler(data_path)"
```

## 주의사항

### 1. scripts/ 디렉토리 수정 금지

프레임워크는 `scripts/` 디렉토리의 기존 유틸리티를 **그대로 활용**합니다:

- `data_loader.py`
- `tokenizer_utils.py`
- `model_utils.py`
- `dataset.py`
- `trainer_utils.py`
- `inference_utils.py`
- `wandb_utils.py`

이 파일들은 **수정하지 마세요**. 프레임워크가 자동으로 import합니다.

### 2. CSV 형식

Competition 제출 파일은 **index 포함**이 필수:

```python
# ✅ 올바른 방식 (프레임워크 자동 처리)
result_df.to_csv(output_path, index=True)

# ❌ 잘못된 방식
result_df.to_csv(output_path, index=False)
```

### 3. 토큰 정리

추론 시 특수 토큰 자동 제거 (config에서 설정):

```yaml
inference:
  remove_tokens:
    - <usr>
    - <s>
    - </s>
    - <pad>
```

## 문제 해결

### ImportError 발생 시

```bash
# scripts 경로 확인
python -c "import sys; print('/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/scripts' in sys.path)"

# sys.path에 수동 추가 (train.py, inference.py 상단에 이미 포함됨)
```

### Wandb 로그인 필요 시

```bash
# .env 파일 확인
cat /Competition/NLP/.env

# Wandb 환경변수 설정 확인
echo $WANDB_API_KEY
```

### 가중치 샘플링 오류 시

```bash
# CSV에 adjusted_label, subcluster_label 컬럼 확인
python -c "import pandas as pd; df = pd.read_csv('data/train.csv'); print(df.columns)"

# 컬럼이 없으면 augmentation_final.csv 사용
cp data/augmentation_final.csv data/train.csv
```

## 향후 개선 사항 (선택)

1. **실험 로그 저장**: `experiment_log.json`에 모든 실험 기록 자동 저장
2. **자동 하이퍼파라미터 튜닝**: Optuna 통합
3. **멀티 GPU 지원**: DistributedDataParallel 적용
4. **모델 앙상블**: 여러 체크포인트 자동 앙상블
5. **자동 제출**: Competition API 통합

## 참고 자료

- 기존 베이스라인: `/Competition/NLP/docs/baseline_code_summary.md`
- 실험 가이드: `/Competition/NLP/docs/RESTART_GUIDE.md`
- 가중치 샘플링: `train_exp7f.py` (원본 로직)

## 라이선스

이 프레임워크는 KoBART 대화 요약 Competition 프로젝트의 일부입니다.
