# KoBART 대화 요약 모듈화 프레임워크

> 실험 관리를 위한 통합 학습/추론 파이프라인

## 빠른 시작 (Quick Start)

### 1분 안에 실행하기

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# 학습
python train.py --experiment exp7a

# 추론
python inference.py --experiment exp7a --checkpoint checkpoint-2068
```

### 새 실험 추가 (3단계)

**1단계**: `config/experiments.yaml`에 실험 추가
```yaml
experiments:
  exp8:
    description: "새로운 실험"
    training:
      learning_rate: 2.0e-05
    data:
      use_weights: false
    wandb:
      name: exp8-custom
```

**2단계**: 학습 실행
```bash
python train.py --experiment exp8
```

**3단계**: 추론 실행
```bash
python inference.py --experiment exp8 --checkpoint checkpoint-XXXX
```

**끝!** 별도의 `train_exp8.py`나 `config_exp8.yaml` 파일 불필요.

---

## 개요 (Overview)

### 주요 특징

- **통합 설정**: 모든 실험을 `config/experiments.yaml`에서 관리
- **코드 재사용**: 단일 CLI로 모든 실험 실행 (중복 코드 95% ↓)
- **자동 가중치**: WeightedRandomSampler 자동 적용
- **기존 호환**: `scripts/` 디렉토리 유틸리티 그대로 활용

### Before/After 비교

| 항목 | Before (기존) | After (프레임워크) |
|------|--------------|-------------------|
| 학습 스크립트 | 20+ 파일 | 1 파일 (`train.py`) |
| Config 파일 | 10+ 파일 | 1 파일 (`experiments.yaml`) |
| 새 실험 추가 | 파일 복제 + 수동 수정 | Config에 3줄 추가 |
| 유지보수 | 모든 파일 수정 | 단일 모듈 수정 |

### 디렉토리 구조

```
code/
├── config/
│   └── experiments.yaml          # 모든 실험 설정
├── core/
│   ├── data.py                   # 데이터 & 가중치 샘플링
│   ├── model.py                  # 모델 로드/저장
│   ├── trainer.py                # 학습 (WeightedSeq2SeqTrainer)
│   └── inference.py              # 추론
├── utils/
│   ├── config.py                 # Config 파싱
│   ├── logger.py                 # 로깅
│   └── metrics.py                # ROUGE 계산
├── scripts/                      # 기존 유틸리티 (수정 없음)
│   ├── data_loader.py
│   ├── tokenizer_utils.py
│   └── ...
├── train.py                      # 학습 CLI
└── inference.py                  # 추론 CLI
```

---

## 사용 가이드 (Usage)

### 학습

```bash
# 기본 사용법
python train.py --experiment <실험명>

# 예시
python train.py --experiment exp7a   # 가중치 없음
python train.py --experiment exp7f   # 가중치 샘플링
```

**자동 실행 과정**:
1. `config/experiments.yaml`에서 설정 로드
2. defaults + 실험별 설정 병합
3. 가중치 샘플링 자동 적용 (`use_weights: true` 시)
4. Wandb 초기화 및 로깅
5. Best checkpoint 저장

### 추론

```bash
# 기본 사용법
python inference.py --experiment <실험명> --checkpoint <체크포인트명> [--output <출력경로>]

# 예시
python inference.py --experiment exp7a --checkpoint checkpoint-2068
python inference.py --experiment exp7f --checkpoint checkpoint-1880 --output ./results/sub.csv
```

**자동 실행 과정**:
1. 실험 설정 로드
2. 체크포인트 경로 자동 생성 (`output_dir/checkpoint_name`)
3. Test 데이터 로딩 및 전처리
4. 요약 생성 (beam search + length penalty)
5. 특수 토큰 제거 (`<s>`, `</s>`, `<usr>`, `<pad>`)
6. Competition 형식 CSV 저장 (index 포함)

---

## 핵심 기능 (Core Features)

### 1. Config 병합 로직

`experiments.yaml` 구조:
```yaml
defaults:
  # 모든 실험의 기본값
  training:
    learning_rate: 1.0e-05
    num_train_epochs: 20
  inference:
    length_penalty: 0.5

experiments:
  exp7a:
    # exp7a 전용 설정 (defaults 오버라이드)
    training:
      learning_rate: 2.0e-05  # 기본값 덮어쓰기
    data:
      use_weights: false
```

**병합 결과** (exp7a):
```yaml
training:
  learning_rate: 2.0e-05      # exp7a에서 오버라이드
  num_train_epochs: 20        # defaults에서 상속
inference:
  length_penalty: 0.5         # defaults에서 상속
data:
  use_weights: false          # exp7a에서 추가
```

### 2. 자동 가중치 샘플링

**설정**:
```yaml
experiments:
  exp7f:
    data:
      use_weights: true
      weight_config:
        domain_threshold: 500        # 500개 이상 → 1.0x
        subcluster_threshold: 300    # 300개 이상 → 1.0x
        domain_weights:
          '노동/고용': 3.70           # 135개 → 3.70배
          '환경': 2.50                # 200개 → 2.50배
        subcluster_weights:
          '감정 지원': 2.31           # 130개 → 2.31배
```

**로직** (`core/data.py`):
```python
# 인간관계/일상 도메인: 서브클러스터 가중치
if domain == '인간관계/일상':
    weight = subcluster_weights.get(subcluster, 1.0)
# 기타 도메인: 도메인 가중치
else:
    weight = domain_weights.get(domain, 1.0)
```

### 3. 출력 형식

**학습 출력**:
```
{output_dir}/
├── checkpoint-xxx/           # 체크포인트
│   ├── pytorch_model.bin
│   └── config.json
└── runs/                     # TensorBoard 로그
```

**추론 출력** (Competition 형식):
```csv
,fname,summary
0,test_0,요약문...
1,test_1,요약문...
498,test_498,요약문...
```
⚠️ **중요**: 첫 컬럼은 index (`,fname,summary` 헤더 필수)

---

## 문제 해결 (Troubleshooting)

### ImportError 발생 시

```bash
# scripts 경로 확인
ls /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/scripts
```

→ `train.py`와 `inference.py`가 자동으로 경로 추가함

### Config 오류 시

```bash
# 실험 목록 확인
grep "^  exp" config/experiments.yaml
```

→ 실험 이름이 `experiments.yaml`에 존재하는지 확인

### Wandb 오류 시

```bash
# 환경변수 확인
cat /Competition/NLP/.env
echo $WANDB_API_KEY
```

→ Wandb API 키 설정 확인

### 가중치 샘플링 오류 시

```bash
# CSV 컬럼 확인
python -c "import pandas as pd; print(pd.read_csv('data/train.csv').columns)"
```

→ `adjusted_label`, `subcluster_label` 컬럼 필요
→ 없으면 `data/augmentation_final.csv` 사용

---

## 주의사항

### 1. scripts/ 디렉토리 수정 금지

프레임워크는 `scripts/`의 기존 유틸리티를 **그대로 활용**:
- `data_loader.py`, `tokenizer_utils.py`, `model_utils.py`
- `dataset.py`, `trainer_utils.py`, `inference_utils.py`
- `wandb_utils.py`

이 파일들은 **수정하지 마세요**. 프레임워크가 자동 import.

### 2. CSV 형식

Competition 제출 파일은 **index 포함** 필수:
```python
# ✅ 올바른 방식 (프레임워크 자동 처리)
result_df.to_csv(output_path, index=True)

# ❌ 잘못된 방식
result_df.to_csv(output_path, index=False)
```

### 3. 토큰 정리

추론 시 특수 토큰 자동 제거 (config 설정):
```yaml
inference:
  remove_tokens:
    - <usr>
    - <s>
    - </s>
    - <pad>
```

---

## 참고 자료

- **기존 베이스라인**: `/Competition/NLP/docs/baseline_code_summary.md`
- **실험 가이드**: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/docs/RESTART_GUIDE.md`
- **실험 로그**: `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/docs/EXPERIMENT_LOG.md`
- **가중치 샘플링 원본**: `train_exp7f.py`

---

## 라이선스

이 프레임워크는 KoBART 대화 요약 Competition 프로젝트의 일부입니다.