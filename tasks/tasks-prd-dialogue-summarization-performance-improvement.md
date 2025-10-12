# Task List: 일상 대화 요약 모델 성능 개선

**기반 PRD**: `prd-dialogue-summarization-performance-improvement.md`
**작성일**: 2025-10-12
**목표**: Baseline 47점 → 50점 이상 달성

---

## 현재 코드베이스 상태

### 기존 구조
```
/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/
├── code/
│   ├── baseline.ipynb          ✅ 존재
│   ├── solar_api.ipynb         ✅ 존재
│   ├── config.yaml             ✅ 존재
│   └── requirements.txt        ✅ 존재
├── data/                       ✅ 존재
├── .gitignore                  ✅ 생성됨
└── README.md                   ✅ 존재 (템플릿)
```

### 생성 필요
- `scripts/` 디렉토리 및 자동화 스크립트
- `docs/experiment_logs.md`
- W&B 통합 코드
- 전처리/증강 스크립트

---

## Relevant Files

### 새로 생성할 파일
- `naturallanguageprocessingcompetition-nlp-1/scripts/train.py` - 학습 자동화 스크립트 (W&B 통합)
- `naturallanguageprocessingcompetition-nlp-1/scripts/inference.py` - 추론 자동화 스크립트
- `naturallanguageprocessingcompetition-nlp-1/scripts/preprocess.py` - 데이터 전처리 (노이즈 처리)
- `naturallanguageprocessingcompetition-nlp-1/scripts/augment.py` - LLM 기반 데이터 증강
- `naturallanguageprocessingcompetition-nlp-1/scripts/utils.py` - 공통 유틸리티 함수
- `naturallanguageprocessingcompetition-nlp-1/scripts/config_manager.py` - Config 관리 유틸리티
- `docs/experiment_logs.md` - 실험 기록 문서

### 수정할 파일
- `naturallanguageprocessingcompetition-nlp-1/code/config.yaml` - 하이퍼파라미터 조정 시 수정
- `naturallanguageprocessingcompetition-nlp-1/README.md` - 프로젝트 문서 업데이트
- `naturallanguageprocessingcompetition-nlp-1/.gitignore` - 필요 시 추가 패턴

### 참조할 파일
- `naturallanguageprocessingcompetition-nlp-1/code/baseline.ipynb` - 베이스라인 구현 참조
- `naturallanguageprocessingcompetition-nlp-1/code/config.yaml` - 기본 설정 참조
- `docs/RESTART_GUIDE.md` - 전략 및 체크리스트 참조

### Notes
- W&B API 키는 `.env` 파일에 저장 (Git 제외)
- 모든 스크립트는 `/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/scripts/` 내에 위치
- 실험 로그는 `/Competition/NLP/docs/experiment_logs.md`에 자동 추가
- 체크포인트/예측 결과는 `.gitignore`로 제외됨

---

## Tasks

- [ ] 1.0 Phase 1: Baseline 재현 및 실험 인프라 구축
- [ ] 2.0 Phase 2: 단계별 성능 개선 (한 번에 하나씩)
- [ ] 3.0 Phase 3: 데이터 증강 파이프라인 구축
- [ ] 4.0 Phase 4: W&B 통합 및 실험 추적 시스템
- [ ] 5.0 Phase 5: 자동화 파이프라인 및 검증 시스템
- [ ] 6.0 Phase 6: CausalLM 재검증 (조건부)
- [ ] 7.0 Phase 7: Git/문서 자동화 및 최종 정리
- [ ] 8.0 Phase 8: 최종 제출 및 코드 정리

---

**다음 단계**: "Go"를 입력하시면 각 태스크의 상세 서브태스크를 생성합니다.