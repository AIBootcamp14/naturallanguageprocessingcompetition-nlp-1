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

### Phase 1: Baseline 재현 및 실험 인프라 구축

- [ ] 1.0 Phase 1: Baseline 재현 및 실험 인프라 구축
  - [ ] 1.1 환경 검증 및 사전 준비
    - [ ] 디스크 용량 확인 (`du -sh / 2>/dev/null`, 150GB 미만 확인)
    - [ ] GPU 상태 확인 (`nvidia-smi`, RTX 3090 24GB 확인)
    - [ ] 데이터 파일 존재 확인 (train.csv, dev.csv, test.csv, sample_submission.csv)
    - [ ] baseline.ipynb 파일 존재 확인
  - [ ] 1.2 Baseline 실행 (수정 금지!)
    - [ ] `code/baseline.ipynb` 열기
    - [ ] 모든 셀 순차 실행 (어떤 코드도 수정하지 말 것!)
    - [ ] 학습 완료 대기 (~20분)
    - [ ] `prediction/output.csv` 생성 확인
  - [ ] 1.3 CSV 검증
    - [ ] `head -5 prediction/output.csv`로 포맷 확인 (`,fname,summary`)
    - [ ] `wc -l prediction/output.csv`로 샘플 수 확인 (500 = header + 499)
    - [ ] Index 컬럼 존재 확인
    - [ ] 5개 샘플 수동 확인
  - [ ] 1.4 Test 제출 및 Baseline 검증
    - [ ] 대회 플랫폼에 `prediction/output.csv` 제출
    - [ ] 점수 확인: 46-48점 (공식: 47.1244) 달성 여부
    - [ ] 실패 시 중단하고 원인 분석
  - [ ] 1.5 실험 로그 시스템 구축
    - [ ] `/Competition/NLP/docs/experiment_logs.md` 생성
    - [ ] 실험 #0 (Baseline) 기록 작성
    - [ ] 템플릿 작성: 날짜, 변경사항, 설정, 결과, 판단, 다음 단계
  - [ ] 1.6 Git 커밋 (Baseline 재현 완료)
    - [ ] `experiment_logs.md` Git에 추가
    - [ ] 커밋 메시지: "Experiment #0: Baseline reproduction - XX.XX points"
    - [ ] GitHub 푸시

---

### Phase 2: 단계별 성능 개선 (한 번에 하나씩!)

- [ ] 2.0 Phase 2: 단계별 성능 개선 (한 번에 하나씩)
  - [ ] 2.1 실험 #1: Learning Rate 튜닝
    - [ ] `code/config.yaml` 백업
    - [ ] `learning_rate: 1e-5 → 5e-5` 변경
    - [ ] 학습 실행 (baseline.ipynb 재실행 또는 스크립트)
    - [ ] Dev ROUGE 점수 기록
    - [ ] Test 제출 및 점수 비교 (Baseline 47 → ?)
    - [ ] 판단: +1점 이상이면 유지, 아니면 롤백
    - [ ] `experiment_logs.md` 업데이트 (실험 #1)
    - [ ] Git 커밋
  - [ ] 2.2 실험 #2: 데이터 노이즈 처리
    - [ ] `scripts/preprocess.py` 생성
    - [ ] `\n` → `\\n` 정규화 구현
    - [ ] `<br>` 태그 처리 구현
    - [ ] 특수문자 정리 로직 추가
    - [ ] 전처리된 데이터로 학습
    - [ ] Dev 평가 → 유망 시 Test 제출
    - [ ] `experiment_logs.md` 업데이트 (실험 #2)
    - [ ] Git 커밋
  - [ ] 2.3 실험 #3: 특수 토큰 최적화
    - [ ] 현재 special_tokens 검토 (config.yaml)
    - [ ] PII 토큰 추가/수정 필요 여부 판단
    - [ ] Tokenizer 재설정 (필요 시)
    - [ ] 학습 및 평가
    - [ ] Dev 평가 → 유망 시 Test 제출
    - [ ] `experiment_logs.md` 업데이트 (실험 #3)
    - [ ] Git 커밋
  - [ ] 2.4 실험 #4: Generation 파라미터 튜닝
    - [ ] `num_beams: 4 → 5` 실험
    - [ ] `num_beams: 4 → 6` 실험 (5 실패 시)
    - [ ] `no_repeat_ngram_size: 2 → 3` 실험
    - [ ] `generate_max_length` 조정 실험
    - [ ] 최적 조합 선정
    - [ ] Dev 평가 → 유망 시 Test 제출
    - [ ] `experiment_logs.md` 업데이트 (실험 #4)
    - [ ] Git 커밋
  - [ ] 2.5 실험 #5: 후처리 로직 구현
    - [ ] 생성 토큰 정리 로직 작성 (`<s>`, `</s>`, `<usr>`, `<pad>`)
    - [ ] 문장 정규화 (공백, 구두점 정리)
    - [ ] 후처리 적용 후 추론
    - [ ] Dev 평가 → 유망 시 Test 제출
    - [ ] `experiment_logs.md` 업데이트 (실험 #5)
    - [ ] Git 커밋
  - [ ] 2.6 Phase 2 중간 점검
    - [ ] 현재까지 Best score 기록
    - [ ] Dev/Test 격차 분석
    - [ ] 다음 개선 방향 결정

---

### Phase 3: 데이터 증강 파이프라인 구축

- [ ] 3.0 Phase 3: 데이터 증강 파이프라인 구축
  - [ ] 3.1 전처리 효과 검증
    - [ ] Phase 2의 Best 개선사항들 조합
    - [ ] 조합된 설정으로 학습
    - [ ] Dev/Test 격차 계산
    - [ ] 격차 5점 이내인지 확인
    - [ ] 격차 10점 이상 시 과적합 경고
  - [ ] 3.2 증강 스크립트 준비
    - [ ] `scripts/augment.py` 생성
    - [ ] LLM 모델 선택 (Solar Mini 또는 Llama-3.2-Korean-3B)
    - [ ] 증강 프롬프트 작성 (번역투 스타일 유지)
    - [ ] 규정 준수 확인 (DialogSum 미사용)
  - [ ] 3.3 소량 테스트 (1,000개)
    - [ ] Train 데이터 중 1,000개 샘플링
    - [ ] LLM으로 증강 (2배: 2,000개 생성)
    - [ ] 증강 품질 수동 검증 (10개 샘플)
    - [ ] 증강 데이터로 학습
    - [ ] Dev 평가로 효과 검증
  - [ ] 3.4 전체 증강 (조건부)
    - [ ] 소량 테스트 성공 시 진행
    - [ ] 전체 Train 데이터 증강 (12,457 → 25,000~37,000)
    - [ ] GPU 병렬 처리로 시간 단축
    - [ ] 디스크 용량 확인 (150GB 미만 유지)
  - [ ] 3.5 증강 데이터 학습
    - [ ] 증강 데이터 포함 학습
    - [ ] Dev 평가
    - [ ] Test 제출 및 점수 확인
    - [ ] `experiment_logs.md` 업데이트 (실험 #6: 데이터 증강)
    - [ ] Git 커밋

---

### Phase 4: W&B 통합 및 실험 추적 시스템

- [ ] 4.0 Phase 4: W&B 통합 및 실험 추적 시스템
  - [ ] 4.1 W&B 환경 설정
    - [ ] W&B 계정 생성/로그인
    - [ ] API 키 `.env`에 저장
    - [ ] 프로젝트 생성: `dialogue-summarization`
    - [ ] Entity 설정 (팀/개인)
  - [ ] 4.2 기본 메트릭 추적 구현
    - [ ] Train/Val Loss 로깅
    - [ ] ROUGE-1, ROUGE-2, ROUGE-L (F1/Precision/Recall) 로깅
    - [ ] Learning Rate (epoch별) 로깅
    - [ ] Gradient Norm 로깅
    - [ ] Run naming 규칙: `{date}_{experiment_name}_{lr}`
  - [ ] 4.3 상세 메트릭 추적 구현
    - [ ] 샘플 예측 결과 Table 추가 (각 epoch 5개)
    - [ ] 하이퍼파라미터 비교 Dashboard 구성
    - [ ] Confusion Matrix (길이별 성능) 추가
  - [ ] 4.4 체크포인트 관리 연동
    - [ ] Best model 자동 저장 (ROUGE 기준)
    - [ ] W&B Artifacts로 모델 업로드
    - [ ] save_total_limit=5 유지
  - [ ] 4.5 W&B 통합 테스트
    - [ ] 이전 실험 재실행 (W&B 연동)
    - [ ] Dashboard 확인
    - [ ] 메트릭 정상 로깅 여부 확인

---

### Phase 5: 자동화 파이프라인 및 검증 시스템

- [ ] 5.0 Phase 5: 자동화 파이프라인 및 검증 시스템
  - [ ] 5.1 학습 자동화 스크립트 작성
    - [ ] `scripts/train.py` 생성
    - [ ] Config YAML 로드 기능
    - [ ] 데이터 전처리 통합
    - [ ] 모델 학습 (W&B 추적 포함)
    - [ ] Best model 저장
    - [ ] 결과 로깅 (console + file)
    - [ ] 실행 예시: `python scripts/train.py --config code/config.yaml`
  - [ ] 5.2 추론 자동화 스크립트 작성
    - [ ] `scripts/inference.py` 생성
    - [ ] Best model 자동 로드
    - [ ] Test 데이터 추론
    - [ ] CSV 생성 (`,fname,summary` 형식)
    - [ ] 자동 검증 (499 samples, index 포함)
    - [ ] 실행 예시: `python scripts/inference.py --model checkpoints/best`
  - [ ] 5.3 검증 스크립트 작성
    - [ ] `scripts/utils.py` 내 validate_csv 함수
    - [ ] CSV 포맷 검증
    - [ ] 샘플 수 확인
    - [ ] Index 컬럼 확인
    - [ ] 특수 토큰 제거 확인
    - [ ] 5개 샘플 출력
  - [ ] 5.4 Config 관리 유틸리티
    - [ ] `scripts/config_manager.py` 생성
    - [ ] Config 로드/저장 함수
    - [ ] 파라미터 변경 헬퍼 함수
    - [ ] Config 비교 함수
  - [ ] 5.5 파이프라인 통합 테스트
    - [ ] train.py 실행 테스트
    - [ ] inference.py 실행 테스트
    - [ ] 전체 파이프라인 (학습 → 추론 → 검증) 테스트
    - [ ] Git 커밋 (자동화 스크립트)

---

### Phase 6: CausalLM 재검증 (조건부: KoBART 50점 달성 후)

- [ ] 6.0 Phase 6: CausalLM 재검증 (조건부)
  - [ ] 6.1 조건 확인
    - [ ] KoBART Best score 50점 이상 확인
    - [ ] 대회 마감일 7일 이상 남음 확인
    - [ ] 진행 여부 최종 결정
  - [ ] 6.2 검증된 전처리 적용
    - [ ] Phase 2-3의 전처리를 Korean_DCS_2024에 적용
    - [ ] 노이즈 처리 로직 이식
    - [ ] 특수 토큰 최적화 적용
  - [ ] 6.3 하이퍼파라미터 신중한 튜닝
    - [ ] Learning rate: 2e-5 기준으로 1e-5, 3e-5 테스트
    - [ ] Epoch: 5 → 3, 7 실험
    - [ ] Batch size/Gradient accumulation 조정
    - [ ] Dev 평가로 유망 조합 선정
  - [ ] 6.4 CausalLM 학습 및 평가
    - [ ] 최적 하이퍼파라미터로 학습
    - [ ] Dev ROUGE 점수 기록
    - [ ] Dev/Test 격차 분석
  - [ ] 6.5 A/B 테스트
    - [ ] KoBART Best model 점수
    - [ ] CausalLM Best model 점수
    - [ ] Dev 격차 비교
    - [ ] 유망 시 CausalLM Test 제출
    - [ ] 최종 모델 선택 (KoBART vs CausalLM)

---

### Phase 7: Git/문서 자동화 및 최종 정리

- [ ] 7.0 Phase 7: Git/문서 자동화 및 최종 정리
  - [ ] 7.1 Git 자동 커밋 스크립트 (선택적)
    - [ ] `scripts/auto_commit.py` 생성
    - [ ] 실험 완료 시 자동 커밋 기능
    - [ ] 커밋 메시지 자동 생성: `Experiment #{N}: {description} - {score}`
    - [ ] 수동 확인 프롬프트 추가
  - [ ] 7.2 실험 로그 자동 업데이트
    - [ ] `experiment_logs.md` 자동 추가 기능
    - [ ] 템플릿 자동 채우기
    - [ ] 결과 자동 기록
  - [ ] 7.3 README 업데이트
    - [ ] Best score 및 성능 개선 기록
    - [ ] 실행 방법 업데이트
    - [ ] 재현 스크립트 경로 추가
    - [ ] 성능 개선 그래프 추가 (선택적)
  - [ ] 7.4 최종 문서 정리
    - [ ] 모든 실험 로그 검토 및 정리
    - [ ] RESTART_GUIDE.md 업데이트 (성공 사례 추가)
    - [ ] 트러블슈팅 가이드 작성

---

### Phase 8: 최종 제출 및 코드 정리

- [ ] 8.0 Phase 8: 최종 제출 및 코드 정리
  - [ ] 8.1 최종 모델 선택
    - [ ] Dev/Test 격차 5점 이내 모델 확인
    - [ ] Test 50점 이상 모델 확인
    - [ ] 재현성 테스트 (3회 학습, 표준편차 ±0.5 이내)
    - [ ] 최종 모델 선정 및 기록
  - [ ] 8.2 최종 제출 파일 생성
    - [ ] 최종 모델로 추론
    - [ ] CSV 생성
    - [ ] 다중 검증 (포맷, 샘플 수, Index, 토큰)
    - [ ] sample_submission.csv와 완전 일치 확인
    - [ ] 5개 샘플 수동 최종 검토
  - [ ] 8.3 제출
    - [ ] 대회 플랫폼에 최종 CSV 제출
    - [ ] 점수 확인 및 기록
    - [ ] 최종 제출 2개 선택 (대회 규정)
  - [ ] 8.4 재현 스크립트 작성
    - [ ] `reproduce.sh` 또는 `reproduce.py` 생성
    - [ ] 환경 설정부터 최종 제출까지 전체 과정
    - [ ] 의존성 freeze: `pip freeze > requirements_frozen.txt`
    - [ ] 실행 방법 문서화
  - [ ] 8.5 코드 정리 및 문서화
    - [ ] 불필요한 파일 삭제
    - [ ] 주석 보완 (모든 Class/def)
    - [ ] 코드 포맷팅 (Black/autopep8)
    - [ ] 트러블슈팅 가이드 완성
    - [ ] 베스트 프랙티스 문서 작성
  - [ ] 8.6 최종 Git 커밋 및 푸시
    - [ ] 모든 변경사항 커밋
    - [ ] 태그 생성: `git tag v1.0-final`
    - [ ] GitHub 푸시
    - [ ] Release 노트 작성 (GitHub)

---

**상태**: ✅ 전체 Task List 생성 완료 (48개 서브태스크)

**다음 단계**: `@process-task-list.md` 사용하여 Task 1.1부터 단계별 구현 시작