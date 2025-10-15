# 일상 대화 요약 모델 성능 개선

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HS6nBbT4)

NIKLuge 2024 일상 대화 요약 대회 참가를 위해 구축한 KoBART 기반 파이프라인입니다. 포트폴리오 용도로 재구성하여 실험 근거, 교훈, 재현 절차를 명확히 남겼습니다. 프로젝트에서 관리하는 실험 로그와 문서를 참고해 최신 상태를 유지합니다.

## 프로젝트 하이라이트
- **최종 점수 47.47 / Baseline 47.12**: Phase 1 파이프라인(LP=0.5)이 달성했습니다.
- **Baseline 대비 +0.35p 개선**: 공식 Baseline(47.12) → Length Penalty 조정(47.47)으로 12초 만에 향상을 확인했습니다.
- **재현 가능한 워크플로우**: 모듈화 코드, CLI 스크립트, 실험 로그를 문서화하여 누구나 동일한 결과를 낼 수 있도록 구성했습니다.
- **핵심 교훈 확립**: Loss Gap 분석, WeightedRandomSampler 함정, Dev/Test 불일치 현상을 문서화했습니다.
- **대회 현황**: 2025-10-15 종료. 총 12회 제출 완료 (12/12 사용).

## 문서 허브
| 문서 | 설명 |
| --- | --- |
| [`docs/COMPETITION_FINAL_REPORT.md`](docs/COMPETITION_FINAL_REPORT.md) | 대회 개요, 최종 결과, 실험 요약 |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | Baseline ~ Exp #7까지 상세 실험 기록 |
| [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) | 실험 교훈, Best Practices, 재사용 가능한 체크리스트 |
| [`docs/ARCHIVE.md`](docs/ARCHIVE.md) | 프로젝트 아카이브, 디스크 관리, 재현 가이드 |
| [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) | 향후 개선 방향, 미시도 아이디어 |
| [`docs/RESTART_GUIDE.md`](docs/RESTART_GUIDE.md) | 재시작 전략, Phase별 체크리스트 |

## 빠른 시작
1. 저장소 루트로 이동해 환경을 구성합니다.
   ```bash
   cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code
   pip install -r requirements.txt
   ```
2. Baseline 실행으로 재현 가능성을 확인합니다.
   ```bash
   jupyter notebook baseline.ipynb
   # 모든 셀 실행 → prediction/output.csv 생성
   ```
3. 최고 성능 모델(LP=0.5)을 재현합니다.
   ```bash
   # config.yaml에서 length_penalty를 0.5로 수정 후
   python inference.py --checkpoint checkpoint-XXXX
   ```
4. CLI 기반 학습/추론으로 실험을 진행합니다.
   ```bash
   python train.py --experiment exp7a
   python inference.py --experiment exp7a --checkpoint checkpoint-2068
   ```

## 재현 가능한 워크플로우
1. **Baseline 재현**
   ```bash
   cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code
   jupyter notebook baseline.ipynb
   # 기대 결과: 47.12점
   ```
2. **최고 성능 모델 (47.47점)**
   ```bash
   # config.yaml 수정: length_penalty=0.5
   python inference.py --checkpoint checkpoint-XXXX
   # 소요 시간: 12초
   ```
3. **데이터 증강 실험 (47.41점)**
   ```bash
   # 증강 데이터 생성 (1,009개)
   python -m src.data.augmentation --output augmentation_final.csv

   # 학습 (가중치 샘플링 미사용)
   python train.py --experiment exp7a

   # 추론
   python inference.py --experiment exp7a --checkpoint checkpoint-2068
   ```
4. **실험 결과 문서화**: 실험 결과는 `wandb/` 및 `logs/`에 저장되고, 요약은 `docs/EXPERIMENT_LOG.md`에 반영합니다.

## 프로젝트 구조
```
naturallanguageprocessingcompetition-nlp-1/
├── code/
│   ├── baseline.ipynb              # 공식 Baseline (47.12점)
│   ├── baseline_modular.ipynb      # 모듈화 버전
│   ├── config.yaml                 # Baseline 하이퍼파라미터
│   ├── config/experiments.yaml     # 실험별 설정
│   ├── core/                       # 프레임워크 모듈
│   ├── utils/                      # 유틸리티 함수
│   ├── train.py                    # CLI 학습 스크립트
│   └── inference.py                # CLI 추론 스크립트
├── docs/                           # 포트폴리오 문서
├── data/
│   ├── train.csv                   # 학습 데이터 (12,457개)
│   ├── dev.csv                     # 검증 데이터 (499개)
│   └── test.csv                    # 테스트 데이터 (499개)
├── submission_exp7a/
│   └── checkpoint-2068/            # 최고 성능 checkpoint (1.4GB)
└── submission/                     # Baseline checkpoint (7.0GB)
```

## 실험 결과

### 완료된 실험 요약

| 실험명 | 설명 | Test 점수 | 변화 | 날짜 | 상태 |
|--------|------|-----------|------|------|------|
| Baseline | 공식 코드 재현 | 47.12 | - | 10/12 | 성공 |
| **Phase 1: LP=0.5** | **Length Penalty 최적화** | **47.47** | **+0.35** | 10/13 | 최고 |
| Phase 1: LP=0.3 | LP 추가 실험 | 47.15 | +0.03 | 10/13 | 성공 |
| Phase 1: LP=0.7 | LP 추가 실험 | 47.22 | +0.10 | 10/13 | 성공 |
| Exp #5 | no_repeat_ngram=3 | 47.03 | -0.44 | 10/14 | 실패 |
| Exp #6 | Learning Rate 3e-5 | - | - | - | 보류 |
| Exp #7-A | 증강 데이터 (가중치 없음) | 47.41 | -0.06 | 10/15 | 안정 |
| Exp #7-C | 가중치 max=5.0 | - | - | 10/15 | 스킵 |
| Exp #7-F | 가중치 조정 (최종) | 46.62 | -0.79 | 10/15 | 실패 |

**제출 통계**:
- 총 제출: 12/12 (100%)
- 점수 범위: 46.62~47.47 (0.85점 차이)
- 평균 개선: +0.35점 (Baseline 대비)

상세 기록: [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)

### 성공 사례

**Phase 1: Length Penalty 조정 (47.47점)**
- `length_penalty` 1.0 → 0.5로 변경했습니다.
- 학습 없이 추론만 재실행했습니다 (12초 소요).
- Baseline 대비 +0.35점 개선을 확인했습니다.

**Exp #7-A: 데이터 증강 (47.41점)**
- 증강 샘플 1,009개를 추가했습니다 (+8.1%).
- WeightedRandomSampler를 사용하지 않았습니다.
- Loss Gap +0.50으로 건강한 일반화를 달성했습니다.

### 실패 사례

**Exp #5: N-gram 패널티 증가 (47.03점, -0.44)**
- `no_repeat_ngram_size`를 2 → 3으로 증가시켰습니다.
- 과도한 제약으로 반복 억제가 오히려 성능을 저하시켰습니다.

**Exp #7-F: 가중치 샘플링 (46.62점, -0.79)**
- WeightedRandomSampler를 사용했습니다 (최대 가중치: 3.70배).
- 소수 카테고리(135개)가 epoch당 500회 반복되어 암기 현상이 발생했습니다.
- Dev ROUGE는 36.43%로 높았으나 Test Score는 46.62로 낮았습니다.
- Train 분포와 Test 분포 불일치가 원인이었습니다.

## 주요 발견

### 1. Loss Gap 분석

**정의**: `Loss Gap = Train Loss - Eval Loss`

**해석**:
- 양수 (+0.15 이상): 정상 학습, 제출 가능
- 음수: 과적합, 제출 금지

**실험 증거**:
- Exp #7-A: Loss Gap +0.50 → Test 47.41 (성공)
- Exp #7-F: Loss Gap +0.47 → Test 46.62 (분포 왜곡으로 실패)

### 2. 가중치 샘플링의 함정

**문제 패턴**:
```
소수 카테고리 (135개) × 3.70배 가중치
→ epoch당 500회 반복
→ 모델 암기 현상
→ Test 실패
```

**교훈**: 증강 없이 가중치를 사용하면 안 됩니다.

### 3. Dev vs Test 불일치

| 실험 | Dev ROUGE-1 | Test Score | 상관관계 |
|------|-------------|------------|----------|
| Exp #7-A | 36.18% | 47.41 | Dev 낮음, Test 높음 |
| Exp #7-F | 36.43% | 46.62 | Dev 높음, Test 낮음 |

**교훈**: Dev는 학습 분포 영향을 받으므로 Test 제출 결과만 신뢰해야 합니다.

### 4. 단순함의 효과

| 접근법 | 복잡도 | 소요시간 | 점수 변화 | 효율성 |
|--------|--------|----------|-----------|--------|
| LP=0.5 | 낮음 | 12초 | +0.35 | 높음 |
| 데이터 증강 | 높음 | 3시간 | -0.06 | 낮음 |
| 가중치 조정 | 중간 | 3시간 | -0.79 | 역효과 |

**교훈**: 간단한 변경이 복잡한 방법보다 효과적입니다.

## 기술 스택
- **언어**: Python 3.10
- **프레임워크**: PyTorch 2.5.1, Transformers 4.46.3
- **모델**: KoBART (digit82/kobart-summarization)
- **평가**: ROUGE (rouge 1.0.1)
- **실험 추적**: Weights & Biases (선택)
- **버전 관리**: Git, GitHub

## 스토리지 & 운영 정책
- **용량 관리**: 작업 전 `du -sh /Competition/NLP`로 디스크 사용량(150GB 한도)을 확인합니다.
- **산출물 관리**: 최고 성능 checkpoint만 보존하고(`submission_exp7a/checkpoint-2068/`), 나머지는 정리했습니다 (21GB → 8.5GB, 12.5GB 절감).
- **외부 백업**: 최종 결과 파일은 GitHub에 커밋하고, 대용량 checkpoint는 별도 문서에 경로를 명시했습니다.
- **실험 추적**: 핵심 실험만 `docs/EXPERIMENT_LOG.md`에 남기고, Wandb 로그는 최근 5개 run만 보존했습니다.

## 개인 회고
- Length Penalty 조정만으로 +0.35점 개선이 가능했던 점이 가장 인상적이었습니다. 복잡한 기법보다 기본 파라미터 튜닝의 중요성을 깨달았습니다.
- WeightedRandomSampler 실험(Exp #7-F)에서 Dev ROUGE는 높지만 Test Score가 낮은 현상을 겪으며, Dev 점수에 의존하지 말고 Loss Gap과 Test 제출 결과를 신뢰해야 함을 배웠습니다.
- 데이터 증강은 성능 향상보다 학습 안정성(Loss Gap +0.50) 확보에 기여했고, 향후 균등 증강 전략을 시도할 여지를 남겼습니다.
- 제출 횟수 제한(12회/일)으로 신중한 실험 전략이 필요했고, Loss Gap 비교를 통해 Exp #7-C를 사전에 스킵하여 제출 횟수를 절약할 수 있었습니다.
- 팀 협업 측면: 단독 프로젝트였으나, 체계적인 문서화와 모듈화 코드 구성으로 향후 협업 가능성을 열어두었습니다.
- 딥러닝 실험 범위: KoBART 외에 더 큰 모델(mBART, KoT5)을 실험하려 했으나, 시간과 GPU 메모리 제약으로 시도하지 못했습니다.
- 후처리 검증 한계: Special token 추가나 후처리 개선 아이디어가 있었지만, 제출 횟수 소진으로 충분히 검증하지 못했습니다.

## 라이선스
MIT License

---

**대회**: NIKLuge 2024 - 일상 대화 요약
**프로젝트 기간**: 2025-10-12 ~ 2025-10-15 (4일)
**최고 점수**: 47.47 (Phase 1: LP=0.5)
**상태**: 대회 종료, 문서화 완료
**마지막 업데이트**: 2025-10-15