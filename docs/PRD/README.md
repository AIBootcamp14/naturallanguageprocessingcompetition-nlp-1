# 📚 PRD (Product Requirements Document)

## 🎯 목적
이 문서들은 **Dialogue Summarization | 일상 대화 요약** NLP 경진대회 프로젝트의 체계적인 관리와 성공적인 수행을 위해 작성되었습니다.

## 📁 문서 구조

### [01. 프로젝트 개요](./01_프로젝트_개요.md)
- 대회 정보 및 목표
- 평가 기준 (ROUGE Score)
- 데이터셋 구성
- 대회 규칙 및 제한사항

### [02. 프로젝트 구조](./02_프로젝트_구조.md)
- 시스템 아키텍처 다이어그램
- 디렉토리 구조 설계
- 파일 명명 규칙
- 모듈화 전략

### [03. 브랜치 전략](./03_브랜치_전략.md)
- Git 브랜치 구조
- 브랜치별 작업 계획
- 커밋 컨벤션
- 머지 규칙

### [04. 성능 개선 전략](./04_성능_개선_전략.md)
- LLM 파인튜닝 중심 전략
- Solar API 토큰 최적화
- 교차 검증 시스템
- 예상 성능: 68-80점

### [05. 실험 추적 관리](./05_실험_추적_관리.md)
- 실험 명명 규칙
- 실험 템플릿
- 성능 추적 테이블
- 실험 관리 도구 (WandB, MLflow)

### [06. 기술 요구사항](./06_기술_요구사항.md)
- 개발 환경 설정
- 필요 라이브러리
- 시스템 요구사항
- 코드 스타일 가이드

### [07. 리스크 관리](./07_리스크_관리.md)
- 주요 리스크 식별
- 대응 방안
- 비상 대응 계획
- 체크리스트

### [08. LLM 파인튜닝 전략](./08_LLM_파인튜닝_전략.md) 🆕
- 디코더 전용 LLM 접근법
- LoRA/QLoRA 효율적 파인튜닝
- Instruction Tuning
- 예상 성능 및 리소스 사용량

### [09. Solar API 최적화](./09_Solar_API_최적화.md) 🆕
- CSV 데이터 전처리로 토큰 절약
- 배치 처리 및 캐싱 전략
- 70-75% 토큰 사용량 감소
- 하이브리드 접근법

### [10. 교차 검증 시스템](./10_교차_검증_시스템.md) 🆕
- 듀얼 생성 시스템 (모델 + API)
- 품질 평가 메트릭
- 최적 요약 선택 알고리즘
- A/B 테스팅 프레임워크

### [11. 로깅 및 모니터링 시스템](./11_로깅_및_모니터링_시스템.md) 🆕
- 통합 로거 시스템
- GPU 최적화 및 자동 배치
- 7가지 학습 시각화
- WandB 통합 추적

### [12. 다중 모델 앙상블 전략](./12_다중_모델_앙상블_전략.md) 🆕
- NLP 최적 모델 선택 (SOLAR, Polyglot-Ko)
- 5-모델 앙상블 시스템
- Text Test Augmentation (TTA)
- 스태킹/블렌딩 전략

### [13. Optuna 하이퍼파라미터 최적화](./13_Optuna_하이퍼파라미터_최적화.md) 🆕
- 베이지안 최적화 전략
- NLP 특화 파라미터 공간
- 단계적 최적화
- 병렬 실행 가이드

### [14. 실행 옵션 시스템](./14_실행_옵션_시스템.md) 🆕
- 통합 train.py 스크립트
- 5가지 실행 모드 (single/kfold/multi/optuna/full)
- 유연한 명령행 인터페이스
- 실험 관리 자동화

### [15. 프롬프트 엔지니어링 전략](./15_프롬프트_엔지니어링_전략.md) 🔥
- Few-shot/Zero-shot 템플릿 라이브러리
- Chain-of-Thought (CoT) 프롬프팅
- 대화 길이별 동적 프롬프트
- 프롬프트 A/B 테스팅 프레임워크

### [16. 데이터 품질 검증 시스템](./16_데이터_품질_검증_시스템.md) 🔥
- 라벨 일관성 검증 메트릭
- 대화-요약 정보 일치도 체크
- 이상치 탐지 알고리즘
- 데이터 증강 품질 검증

### [17. 추론 최적화 전략](./17_추론_최적화_전략.md) 🔥
- ONNX 변환 파이프라인
- TensorRT 가속화 설정
- INT8/INT4 양자화 전략
- 배치 추론 최적화

## 🚀 Quick Start

### 1. 환경 설정
```bash
# Python 가상환경 생성
pyenv virtualenv 3.11.9 nlp_py3_11_9
pyenv activate nlp_py3_11_9

# 필요 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# 데이터 다운로드 (이미 완료됨)
# data/raw/ 폴더에 train.csv, dev.csv, test.csv 확인
```

### 3. GPU 체크 및 자동 설정
```bash
# GPU 호환성 체크
python src/utils/gpu_optimization/team_gpu_check.py

# 최적 배치 크기 자동 탐색
python train.py --mode single --auto_batch --gpu_check
```

### 4. 빠른 실험 실행
```bash
# 단일 모델 테스트
python train.py --mode single --models solar-10.7b --epochs 1 --debug

# K-Fold 검증
python train.py --mode kfold --models polyglot-ko-12.8b --k_folds 5

# 다중 모델 앙상블
python train.py --mode multi_model --models solar-10.7b polyglot-ko-12.8b kullm-v2

# Optuna 최적화
python train.py --mode optuna --optuna_trials 50

# 풀 파이프라인 (최종)
python train.py --mode full --models all --use_tta --save_visualizations
```

## 📅 프로젝트 일정

| 주차 | 기간 | 주요 작업 | 목표 |
|-----|------|-----------|------|
| Week 1 | 09/26-10/02 | 프로젝트 셋업, EDA, 전처리 | 베이스라인 구축 |
| Week 2 | 10/03-10/09 | 모델 실험, 하이퍼파라미터 튜닝 | 성능 개선 |
| Week 3 | 10/10-10/15 | 앙상블, 최종 최적화 | 최종 제출 |

## 🎯 목표 성능 (업데이트 v3.0)

| 단계 | 예상 점수 | 전략 |
|------|-----------|------|
| 베이스라인 | 47.12 | KoBART 기본 |
| 1차: 단일 모델 | 70-73 | **SOLAR-10.7B** (최고 성능 모델) |
| 2차: LLM + LoRA | 68-71 | **Polyglot-Ko + LoRA** (효율성) |
| 3차: K-Fold | 72-75 | **5-Fold 교차 검증** |
| 4차: 다중 모델 | 75-80 | **5-모델 앙상블** |
| 5차: +TTA | 77-82 | **Text Augmentation** 추가 |
| 6차: +Optuna | 80-85 | **하이퍼파라미터 최적화** |
| 최종 목표 | **85+** | **풀 파이프라인** (모든 기법 통합) |

## 👥 팀 역할 분담

### 브랜치별 담당자 (권장)
- **feature/eda**: 데이터 분석 담당
- **feature/custom-dataset**: 데이터 엔지니어
- **feature/modularization**: 소프트웨어 엔지니어
- **feature/training**: ML 엔지니어
- **feature/inference**: 최적화 담당
- **feature/full-pipeline**: 프로젝트 매니저

## 📋 체크리스트

### 시작 전
- [x] PRD 문서 작성
- [x] 프로젝트 구조 설계
- [x] 베이스라인 분석
- [ ] 팀 역할 분담
- [ ] 개발 환경 통일

### 진행 중
- [ ] 일일 스탠드업 미팅
- [ ] 실험 결과 공유
- [ ] 코드 리뷰
- [ ] 문서 업데이트

### 마무리
- [ ] 최종 모델 선정
- [ ] 코드 정리 및 문서화
- [ ] 제출물 검증
- [ ] 사후 분석

## 💡 팁

### 🏆 성능 개선 우선순위 (v3.0)
1. **모델 선택**: SOLAR-10.7B > Polyglot-Ko-12.8B > KULLM-v2
2. **실행 모드 진화**: single → kfold → multi_model → optuna → full
3. **TTA 전략**: paraphrase(필수) + reorder + synonym
4. **앙상블 방법**: weighted_avg > stacking > majority_vote
5. **Optuna 단계**: 모델선택 → 학습률 → 생성파라미터 → 미세조정

### 🚀 추천 실행 순서
```bash
# 1. GPU 체크 및 환경 확인
python src/utils/gpu_optimization/team_gpu_check.py

# 2. 빠른 모델 테스트 (어떤 모델이 좋은지)
python train.py --mode single --models all --epochs 1 --debug

# 3. 최고 모델로 Optuna 최적화
python train.py --mode optuna --models solar-10.7b --optuna_trials 100

# 4. 최적 파라미터로 K-Fold 검증
python train.py --mode kfold --k_folds 5 --config best_config.yaml

# 5. 상위 3개 모델 앙상블
python train.py --mode multi_model --models top3 --ensemble_strategy stacking

# 6. 최종 제출 (모든 기법 통합)
python train.py --mode full --use_tta --optuna_trials 50 --save_visualizations
```

### 주의사항
- ⚠️ **DialogSum 데이터셋 사용 금지**
- ⚠️ **일일 제출 횟수 12회 제한**
- ⚠️ **무료 API만 사용 (Solar API 제외)**

---

*Last Updated: 2025.10.10*
*Version: 3.0*