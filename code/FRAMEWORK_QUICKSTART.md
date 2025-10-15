# 프레임워크 빠른 시작 가이드

## 1분 안에 실행하기

### 학습

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# Exp #7-A 학습 (가중치 없음)
python train.py --experiment exp7a

# Exp #7-F 학습 (가중치 샘플링)
python train.py --experiment exp7f
```

### 추론

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# Exp #7-A 추론
python inference.py --experiment exp7a --checkpoint checkpoint-2068

# Exp #7-F 추론
python inference.py --experiment exp7f --checkpoint checkpoint-1880
```

## 새 실험 추가하기 (3단계)

### 1단계: config 추가

`config/experiments.yaml`에 새 실험 추가:

```yaml
experiments:
  exp8:
    description: "새로운 실험"
    general:
      output_dir: /Competition/NLP/.../submission_exp8
    training:
      learning_rate: 2.0e-05
    data:
      use_weights: false
    wandb:
      name: exp8-custom
```

### 2단계: 학습 실행

```bash
python train.py --experiment exp8
```

### 3단계: 추론 실행

```bash
python inference.py --experiment exp8 --checkpoint checkpoint-XXXX
```

**끝!** 별도의 `train_exp8.py` 파일이나 `config_exp8.yaml` 불필요.

## 주요 파일

| 파일 | 설명 |
|------|------|
| `config/experiments.yaml` | 모든 실험 설정 |
| `train.py` | 학습 CLI |
| `inference.py` | 추론 CLI |
| `core/trainer.py` | 학습 로직 |
| `core/inference.py` | 추론 로직 |
| `core/data.py` | 가중치 샘플링 |
| `README_framework.md` | 상세 문서 |

## 기존 방식 vs 프레임워크

### Before (기존)

```bash
# 매 실험마다
cp train_exp7a.py train_exp8.py
cp config_exp7a.yaml config_exp8.yaml
vim train_exp8.py  # 수동 수정
vim config_exp8.yaml  # 수동 수정
python train_exp8.py
```

### After (프레임워크)

```bash
# config만 추가
vim config/experiments.yaml  # exp8 추가
python train.py --experiment exp8
```

## 문제 해결

### ImportError

```bash
# scripts 경로 확인
ls /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/scripts
```

→ train.py와 inference.py가 자동으로 경로 추가

### Config 오류

```bash
# 실험 목록 확인
grep "^  exp" config/experiments.yaml
```

→ 실험 이름이 experiments.yaml에 있는지 확인

### Wandb 오류

```bash
# 환경변수 확인
cat /Competition/NLP/.env
```

→ WANDB_API_KEY 설정 확인

## 더 자세한 내용

전체 문서는 `README_framework.md`를 참고하세요:

- 디렉토리 구조
- 가중치 샘플링 로직
- Config 병합 규칙
- 출력 파일 형식
- 검증 방법
