# 프로젝트 아카이브 가이드

**프로젝트**: Dialogue Summarization 경진대회
**작성일**: 2025-10-15
**목적**: 대회 종료 후 프로젝트 아카이브 및 용량 관리 가이드

---

## 목차

1. [개요](#개요)
2. [현재 디스크 사용량](#현재-디스크-사용량)
3. [Checkpoint 분석](#checkpoint-분석)
4. [정리 권장사항](#정리-권장사항)
5. [실행 스크립트](#실행-스크립트)
6. [Git 아카이브](#git-아카이브)
7. [재현 가이드](#재현-가이드)

---

## 개요

대회 종료 후 프로젝트를 효율적으로 아카이브하고 디스크 공간을 관리하기 위한 가이드입니다.

### 목표

1. 최고 성능 checkpoint를 보존합니다 (**47.41점**, **checkpoint-2068**)
2. 불필요한 checkpoint를 정리합니다 (실패 실험, 중간 checkpoint)
3. 디스크 용량을 절감합니다 (21GB → 8.4GB, **12.6GB 절감**)
4. 재현 가능성을 유지합니다 (문서 + 최고 성능 모델)

---

## 현재 디스크 사용량

### 전체 사용량 요약

| 항목 | 사용량 | 한도 대비 | 상태 |
|------|--------|-----------|------|
| **루트 전체** | 279GB | 17% (1.8TB 중) | 안전 |
| **프로젝트 전체** | 21GB | 7.5% | 여유 있음 |
| **150GB 한도 대비** | 21GB | 14% | 충분한 여유 |

결론: 현재 디스크 사용량은 매우 안정적이며, 150GB 한도의 14%만 사용 중입니다.

### 프로젝트 디렉토리 구성

```
/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/ (21GB)
├── submission_exp7a/        7.0GB  (최고 성능 실험)
│   ├── checkpoint-1504/     1.4GB
│   ├── checkpoint-1692/     1.4GB
│   ├── checkpoint-1880/     1.4GB
│   ├── checkpoint-2068/     1.4GB  [최고 성능: 47.41점]
│   └── checkpoint-2256/     1.4GB
├── submission_exp7f/        7.0GB  (실패 실험)
│   ├── checkpoint-1128/     1.4GB
│   ├── checkpoint-1316/     1.4GB
│   ├── checkpoint-1504/     1.4GB
│   ├── checkpoint-1692/     1.4GB
│   └── checkpoint-1880/     1.4GB
├── submission/              7.0GB  (Baseline 실험)
│   ├── checkpoint-1996/     1.4GB
│   ├── checkpoint-2245/     1.4GB
│   ├── checkpoint-2494/     1.4GB
│   ├── checkpoint-2743/     1.4GB
│   └── checkpoint-2992/     1.4GB
├── code/                    6.3MB
│   └── wandb/               6.3MB
├── docs/                    1.2MB
└── data/                    297KB
```

---

## Checkpoint 분석

### Checkpoint 파일 구조

**각 checkpoint 크기**: 1.4GB

**파일 구성**:
```
checkpoint-XXXX/
├── model.safetensors       473MB  (모델 가중치)
├── optimizer.pt            946MB  (옵티마이저 상태)
├── scheduler.pt            1.5KB
├── scaler.pt               1.4KB
├── training_args.bin       5.9KB
├── config.json             수 KB
└── generation_config.json  수 KB
```

### 최고 성능 Checkpoint 식별

#### 보존 필수: `submission_exp7a/checkpoint-2068/`

**성능**:
- **Test Score**: **47.41점** (최고 2위)
- **Loss Gap**: +0.50 (안정적 학습)
- **Dev ROUGE-1**: 36.18%
- **Best Epoch**: 11/20

**학습 설정**:
```yaml
data:
  train_samples: 13,465 (원본 12,457 + 증강 1,009)
  weighted_sampler: false

training:
  learning_rate: 1e-5
  batch_size: 24
  gradient_accumulation_steps: 3
  epochs: 20
  early_stopping_patience: 3

generation:
  length_penalty: 0.5
  num_beams: 4
  no_repeat_ngram_size: 2
```

**보존 이유**:
1. 최고 성능 (**47.41점**)
2. Loss Gap 양수 (+0.50, 건강한 일반화)
3. 증강 데이터 효과 검증됨
4. 향후 실험의 기준점

**재현 방법**:
```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code
python inference.py --experiment exp7a --checkpoint checkpoint-2068
```

---

#### 삭제 대상 Checkpoint

**1. `submission_exp7f/` 전체 (7.0GB)**

**이유**:
- Test Score: 46.62점 (실패)
- WeightedRandomSampler로 인한 과적합
- Loss Gap +0.47 (exp7a보다 낮음)
- 재현 가치 없음

**2. `submission_exp7a/` 일부 (5.6GB)**

**보존**: **checkpoint-2068/** (최고 성능)
**삭제**: 나머지 4개 checkpoint
- `checkpoint-1504` (1.4GB)
- `checkpoint-1692` (1.4GB)
- `checkpoint-1880` (1.4GB)
- `checkpoint-2256` (1.4GB)

**이유**: Best Epoch 11 (**checkpoint-2068**)만 필요

**3. `submission/` 전체 (7.0GB) - 선택적**

**이유**:
- Baseline 실험용
- 재현은 코드로 가능
- 역사적 가치만 있음

**주의**: Baseline 재현이 필요하면 보존

---

### Wandb 로그 분석

| Run ID | 크기 | 날짜 | 실험명 추정 |
|--------|------|------|------------|
| `run-20251014_225447-7ocejui1` | 944KB | 2025-10-14 | Exp #7 초기 |
| `run-20251014_234846-enivk05t` | 1.8MB | 2025-10-14 | Exp #7-A 또는 #7-C |
| `run-20251015_011706-9oaf8vko` | 1.3MB | 2025-10-15 | Exp #7-C/F |
| `run-20251015_014000-wxum8khm` | 592KB | 2025-10-15 | Exp #7-F |
| `run-20251015_015350-qak765vu` | 1.9MB | 2025-10-15 | Exp #7-F 최종 |
| **총합** | **6.3MB** | - | 5개 run |

평가: Wandb 로그는 무시할 수 있을 정도로 작음 (6.3MB). 삭제 불필요.

---

## 정리 권장사항

### 시나리오 A: 보수적 정리 (권장)

**삭제**:
- `submission_exp7f/` 전체 (7.0GB)
- `submission_exp7a/` 일부 checkpoint (5.6GB)

**절감**: **12.6GB**
**남는 용량**: **8.4GB**
**보존**: 최고 성능 checkpoint + baseline

**장점**:
- Baseline 재현 가능
- 최고 성능 checkpoint 보존
- 충분한 디스크 공간 확보

**명령어**:
```bash
# 실패 실험 삭제
rm -rf /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7f/

# exp7a 일부 checkpoint 삭제
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7a/
rm -rf checkpoint-1504 checkpoint-1692 checkpoint-1880 checkpoint-2256
```

---

### 시나리오 B: 공격적 정리

**삭제**:
- 시나리오 A +
- `submission/` 전체 (7.0GB)

**절감**: **19.6GB**
**남는 용량**: **1.4GB** (최고 성능 checkpoint만)

**장점**:
- 최대 디스크 공간 확보
- 최고 성능 checkpoint만 보존

**단점**:
- Baseline 재현 불가 (코드로는 가능)

**명령어**:
```bash
# 시나리오 A 명령어 + 아래 추가
rm -rf /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission/
```

---

### 시나리오 C: 최소 정리

**삭제**:
- `submission_exp7f/` 전체만 (7.0GB)

**절감**: **7.0GB**
**남는 용량**: **14GB**

**장점**:
- 최소한의 변경
- 모든 성공 실험 보존

**명령어**:
```bash
rm -rf /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7f/
```

---

### 정리 시나리오 비교표

| 시나리오 | 삭제 대상 | 절감 | 남는 크기 | 보존 | 권장도 |
|---------|----------|------|-----------|------|--------|
| **A (보수적)** | exp7f + exp7a 일부 | **12.6GB** | **8.4GB** | Best + Baseline | 권장 |
| **B (공격적)** | exp7f + exp7a 일부 + baseline | **19.6GB** | **1.4GB** | Best만 | 차선 |
| **C (최소)** | exp7f만 | **7.0GB** | **14GB** | All success | 최소 |

---

## 실행 스크립트

### 정리 스크립트 (시나리오 A)

```bash
#!/bin/bash
# 대회 종료 정리 스크립트 (시나리오 A: 보수적 정리)
# 경로: /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/cleanup_archive.sh

set -e  # 에러 시 중단

echo "=========================================="
echo "   대회 종료 정리 (시나리오 A: 보수적)"
echo "=========================================="
echo ""

# 현재 디스크 사용량 확인
echo "[1/6] 현재 디스크 사용량 확인..."
BEFORE_SIZE=$(du -sh /Competition/NLP/naturallanguageprocessingcompetition-nlp-1 2>/dev/null | cut -f1)
echo "정리 전 크기: ${BEFORE_SIZE}"
echo ""

# 백업 확인
echo "[2/6] 최고 성능 checkpoint 존재 확인..."
BEST_CHECKPOINT="/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7a/checkpoint-2068"
if [ -d "${BEST_CHECKPOINT}" ]; then
    echo "[확인] checkpoint-2068 존재함 (보존)"
else
    echo "[오류] checkpoint-2068 없음! 정리 중단"
    exit 1
fi
echo ""

# 실패 실험 삭제
echo "[3/6] 실패 실험 삭제 (submission_exp7f/)..."
EXP7F_DIR="/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7f"
if [ -d "${EXP7F_DIR}" ]; then
    rm -rf "${EXP7F_DIR}"
    echo "[완료] submission_exp7f/ 삭제 완료 (7.0GB 절감)"
else
    echo "[경고] submission_exp7f/ 이미 없음"
fi
echo ""

# exp7a 불필요한 checkpoint 삭제
echo "[4/6] exp7a 일부 checkpoint 삭제..."
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7a/

CHECKPOINTS_TO_DELETE=("checkpoint-1504" "checkpoint-1692" "checkpoint-1880" "checkpoint-2256")
for CKPT in "${CHECKPOINTS_TO_DELETE[@]}"; do
    if [ -d "${CKPT}" ]; then
        rm -rf "${CKPT}"
        echo "  [완료] ${CKPT} 삭제"
    else
        echo "  [경고] ${CKPT} 이미 없음"
    fi
done
echo "[완료] exp7a 4개 checkpoint 삭제 완료 (5.6GB 절감)"
echo ""

# 최종 확인
echo "[5/6] 정리 후 디스크 사용량..."
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1
AFTER_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "정리 후 크기: ${AFTER_SIZE}"
echo ""

# 보존된 checkpoint 확인
echo "[6/6] 보존된 checkpoint 확인..."
if [ -d "submission_exp7a/checkpoint-2068" ]; then
    echo "[확인] 최고 성능 checkpoint 보존: submission_exp7a/checkpoint-2068/"
fi
if [ -d "submission" ]; then
    echo "[확인] Baseline checkpoint 보존: submission/"
fi
echo ""

echo "=========================================="
echo "   정리 완료"
echo "=========================================="
echo "정리 전 크기: ${BEFORE_SIZE}"
echo "정리 후 크기: ${AFTER_SIZE}"
echo "절감량: 약 12.6GB"
echo ""
echo "보존된 파일:"
echo "  - submission_exp7a/checkpoint-2068/  (최고 성능: 47.41점)"
echo "  - submission/                         (Baseline)"
echo "  - code/, docs/, data/                 (코드 및 문서)"
echo "=========================================="
```

**실행 방법**:
```bash
# 스크립트 생성
cat > /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/cleanup_archive.sh << 'EOF'
[위 스크립트 내용]
EOF

# 실행 권한 부여
chmod +x /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/cleanup_archive.sh

# 실행
/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/cleanup_archive.sh
```

---

### 수동 정리 명령어

**시나리오 A (보수적 정리)**:
```bash
# 실패 실험 삭제
rm -rf /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7f/

# exp7a 일부 checkpoint 삭제
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7a/
rm -rf checkpoint-1504 checkpoint-1692 checkpoint-1880 checkpoint-2256

# 확인
du -sh /Competition/NLP/naturallanguageprocessingcompetition-nlp-1
```

**시나리오 B (공격적 정리)**:
```bash
# 시나리오 A + Baseline 삭제
rm -rf /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission/
```

**시나리오 C (최소 정리)**:
```bash
# 실패 실험만 삭제
rm -rf /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7f/
```

---

## Git 아카이브

### .gitignore 업데이트

```gitignore
# Checkpoints (최고 성능 checkpoint만 보존)
submission_exp7f/
submission/
submission_exp7a/checkpoint-*/
!submission_exp7a/checkpoint-2068/

# Wandb logs
code/wandb/

# Python cache
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Data files
data/*.csv
!data/README.md

# Submission outputs
code/prediction/*.csv

# Temporary files
*.log
*.tmp
.DS_Store
```

### Git LFS 설정 (선택적)

**대용량 파일 관리**:
```bash
# Git LFS 초기화
git lfs install

# 최고 성능 checkpoint 추적
git lfs track "submission_exp7a/checkpoint-2068/**"

# .gitattributes 업데이트
git add .gitattributes
```

---

### 최종 커밋

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1

# 정리 전 현재 상태 커밋
git add .
git commit -m "$(cat <<'EOF'
대회 종료 전 최종 스냅샷

현재 상태:
- 최고 점수: 47.47점 (Phase 1: LP=0.5)
- 총 제출: 12/12 사용 완료
- 실험 문서화 완료

다음 단계: 정리 작업 (시나리오 A)

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# 정리 작업 수행 (cleanup_archive.sh)

# 정리 후 최종 커밋
git add .
git commit -m "$(cat <<'EOF'
대회 종료 후 아카이브 정리 (시나리오 A)

정리 내용:
- submission_exp7f/ 전체 삭제 (7.0GB 절감)
- submission_exp7a/ 일부 checkpoint 삭제 (5.6GB 절감)
- 총 절감량: 12.6GB (21GB → 8.4GB)

보존:
- submission_exp7a/checkpoint-2068/ (최고 성능: 47.41점)
- submission/ (Baseline)
- 모든 코드 및 문서

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# GitHub 푸시
git push origin main
```

---

## 재현 가이드

### 최고 성능 모델 재현

#### 1. 환경 설정

```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code

# 가상환경 생성 (선택적)
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

#### 2. 데이터 준비

```bash
# 원본 데이터 확인
ls -lh /Competition/NLP/data/
# train.csv (12,457개)
# dev.csv (499개)
# test.csv (499개)

# 증강 데이터 (Exp #7-A용)
# augmentation_final.csv (1,009개)
```

#### 3. 모델 추론

**최고 성능 모델 (47.41점)**:
```bash
python inference.py \
    --experiment exp7a \
    --checkpoint checkpoint-2068 \
    --output submission_exp7a.csv
```

**설정 파일**:
```bash
# config/experiments.yaml
experiments:
  exp7a:
    general:
      output_dir: /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission_exp7a

    tokenizer:
      encoder_max_len: 512
      decoder_max_len: 100

    inference:
      length_penalty: 0.5
      num_beams: 4
      no_repeat_ngram_size: 2
      max_length: 100
      early_stopping: true

    wandb:
      name: kobart-exp7a-augmented
      tags:
        - exp7a
        - augmented
        - no-weights
      group: phase2-experiments
```

#### 4. 결과 확인

```bash
# 출력 파일 확인
head submission_exp7a.csv

# 예상 출력:
# ,fname,summary
# 0,test_0,요약문...
# 1,test_1,요약문...
```

---

### Baseline 재현 (47.12점)

**Option 1: 기존 checkpoint 사용** (시나리오 A/C 적용 시)
```bash
python inference.py \
    --config config.yaml \
    --checkpoint submission/checkpoint-XXXX \
    --output submission_baseline.csv
```

**Option 2: 처음부터 학습** (시나리오 B 적용 시)
```bash
# 1. 학습 (20분 소요)
python train.py --config config.yaml

# 2. 추론
python inference.py \
    --config config.yaml \
    --checkpoint checkpoint-XXXX \
    --output submission_baseline.csv
```

---

### 재학습 (Exp #7-A 재현)

```bash
# 1. 증강 데이터 준비
# augmentation_final.csv 필요

# 2. 학습 (3시간 소요)
python train.py --experiment exp7a

# 3. 추론 (Best Epoch checkpoint 사용)
python inference.py \
    --experiment exp7a \
    --checkpoint checkpoint-2068 \
    --output submission_exp7a.csv

# 4. 예상 결과
# Loss Gap: +0.50
# Dev ROUGE-1: ~36.18%
# Test Score: ~47.41점
```

---

## 다음 단계 체크리스트

### 정리 전

- [ ] 현재 디스크 사용량 확인 (`du -sh`)
- [ ] **checkpoint-2068** 백업 확인
- [ ] `EXPERIMENT_LOG.md` 최신 상태 확인
- [ ] Git commit (현재 상태)

### 정리 실행

- [ ] 시나리오 선택 (A: 보수적 / B: 공격적 / C: 최소)
- [ ] `cleanup_archive.sh` 스크립트 실행
- [ ] 정리 후 디스크 사용량 재확인
- [ ] 최고 성능 checkpoint 존재 확인

### 문서 업데이트

- [완료] `COMPETITION_FINAL_REPORT.md` 최신화
- [완료] `LESSONS_LEARNED.md` 작성
- [완료] `ARCHIVE.md` 작성
- [ ] `README.md` 최종 성과 기록
- [ ] `.gitignore` 업데이트

### Git 커밋

- [ ] 정리 전 현재 상태 커밋
- [ ] 정리 후 최종 커밋
- [ ] GitHub 푸시

### 최종 아카이브

- [ ] **checkpoint-2068** 외부 백업 (선택적)
- [ ] 실험 로그 PDF 변환 (선택적)
- [ ] 프로젝트 회고 작성 (선택적)

---

## 관련 문서

- [COMPETITION_FINAL_REPORT.md](COMPETITION_FINAL_REPORT.md) - 대회 최종 결과
- [LESSONS_LEARNED.md](LESSONS_LEARNED.md) - 실험 교훈 및 Best Practices
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) - 전체 실험 상세 기록
- [NEXT_STEPS.md](NEXT_STEPS.md) - 향후 개선 방향
- [RESTART_GUIDE.md](RESTART_GUIDE.md) - 재시작 가이드
- [code/README.md](../code/README.md) - 프레임워크 사용법

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-10-15
**작성자**: AI Assistant (Claude Code)
**상태**: 최종본

**권장 조치**: 시나리오 A (보수적 정리) 실행 → **12.6GB 절감**, 최고 성능 + Baseline 보존
