# PRD: 일상 대화 요약 모델 성능 개선

## 1. Introduction/Overview

본 프로젝트는 **일상 대화 한국어 요약 경진대회**에서 현재 Baseline 성능(47점 ROUGE-F1)을 **50점 이상으로 향상**시켜 **1등을 달성**하는 것을 목표로 합니다.

### 핵심 문제
- 이전 세션에서 Dev set 과적합 문제 발생 (Dev 94점 → Test 20점)
- 복잡한 개선 시도로 인한 디버깅 불가능 상태
- CausalLM 방식 실패 (Dev 76.30 → Test 27.21)

### 해결 방안
**"한 번에 하나씩, Test로 검증"** 원칙을 바탕으로 안전하고 체계적인 성능 개선 파이프라인을 구축하여, 재현 가능하고 검증된 개선을 통해 목표 달성

---

## 2. Goals

### 주요 목표
1. **Baseline 47점 → 50점 이상 달성** (ROUGE-F1 기준)
2. **Dev/Test 격차 5점 이내 유지** (과적합 방지)
3. **재현 가능한 파이프라인 구축** (모든 단계 문서화)
4. **제출 횟수 12회 내 효율적 활용** (Daily 제한)

### 부가 목표
5. W&B 기반 실험 추적 시스템 구축
6. KoBART Baseline 안정화 후 CausalLM 방식 재검증
7. Git/문서 자동 최신화 파이프라인 구축
8. 대회 1등 달성

---

## 3. User Stories

### Story 1: 연구자/개발자로서의 실험 관리
```
As a: 대회 참가자 (연구자/개발자)
I want to: 각 개선사항의 효과를 개별적으로 측정하고 추적
So that: 무엇이 성능 향상에 기여했는지 명확히 파악하고 재현할 수 있다
```

### Story 2: 안정적인 성능 개선
```
As a: 대회 참가자
I want to: Dev set 점수를 참고만 하고 Test set으로 최종 검증
So that: 이전처럼 Dev 과적합 문제를 피하고 실전 성능을 정확히 예측할 수 있다
```

### Story 3: 효율적인 제출 관리
```
As a: 대회 참가자
I want to: Dev set으로 빠른 실험 후 3-4개 개선사항마다 Test 제출
So that: 제출 횟수 12회 제한 내에서 최대한 많은 검증을 수행할 수 있다
```

### Story 4: 데이터 기반 의사결정
```
As a: 연구자
I want to: W&B 대시보드로 Loss, ROUGE, 하이퍼파라미터, 샘플 예측 결과를 시각화
So that: 데이터 기반으로 다음 개선 방향을 결정할 수 있다
```

### Story 5: 자동화된 워크플로우
```
As a: 개발자
I want to: 학습/추론/CSV 생성을 자동화하되 제출 전 수동 검증
So that: 시간을 절약하면서도 에러를 최소화할 수 있다
```

---

## 4. Functional Requirements

### Phase 1: Baseline 재현 및 검증 (필수)

#### FR1.1: Baseline 실행 환경 구축
- `baseline.ipynb`를 **수정 없이 그대로** 실행
- 예상 출력: `prediction/output.csv` (499 samples)
- 예상 성능: 46-48점 (ROUGE-F1)

#### FR1.2: Baseline 검증
- Test set 제출
- 47±1점 달성 확인
- 실패 시 중단 및 원인 분석

#### FR1.3: 실험 기록 시스템 구축
- `/Competition/NLP/docs/experiment_logs.md` 생성
- 템플릿: 변경사항, 설정, 결과, 판단, 다음 단계
- 모든 실험을 순차적으로 기록

---

### Phase 2: 단계별 개선 (한 번에 하나씩!)

#### FR2.1: 하이퍼파라미터 튜닝 (1순위)
- Learning rate만 변경: `1e-5 → 5e-5`
- 학습 → Dev 평가 → Test 제출
- 예상 효과: +0.5~1.5점
- 판단 기준:
  - +1점 이상: ✅ 유지
  - +0.5~1점: ✅ 유지 가능
  - 0~0.5점: ⚠️ 재검토
  - 음수: ❌ 롤백

#### FR2.2: 데이터 노이즈 처리 (2순위)
- `\n` → `\\n` 정규화
- `<br>` 태그 처리
- 특수문자 정리
- Dev 평가 → 유망 시 Test 제출

#### FR2.3: 특수 토큰 최적화 (3순위)
- PII 마스킹 토큰 검토 (`#Person1#`, `#PhoneNumber#` 등)
- Tokenizer special_tokens 추가/수정
- Dev 평가 → 유망 시 Test 제출

#### FR2.4: Generation 파라미터 튜닝 (4순위)
- `num_beams`: 4 → 5, 6 실험
- `no_repeat_ngram_size`: 2 → 3 실험
- `max_length` 조정
- Dev 평가 → 유망 시 Test 제출

#### FR2.5: 후처리 로직 구현 (5순위)
- 생성 토큰 정리 (`<s>`, `</s>`, `<usr>`, `<pad>` 제거)
- 문장 정규화
- Dev 평가 → 유망 시 Test 제출

---

### Phase 3: 데이터 증강 (단계적)

#### FR3.1: 전처리 효과 검증 (먼저)
- Phase 2의 개선사항들을 조합
- Dev/Test 격차 분석
- 격차 5점 이내 확인

#### FR3.2: LLM 기반 증강 (검증 후)
- 번역투 스타일 유지하며 데이터 증강
- Solar Mini 또는 Llama-3.2-Korean-3B 활용
- 규정 준수 (DialogSum 미사용)
- 소량 테스트 (1,000개) → 효과 검증 → 전체 증강
- 2-3배 증강 (12,457 → 25,000~37,000)
- Dev 평가 → 유망 시 Test 제출

---

### Phase 4: W&B 추적 시스템

#### FR4.1: W&B 기본 설정
- 프로젝트: `dialogue-summarization`
- Entity: 팀/개인 계정
- Run naming: `{date}_{experiment_name}_{lr}`

#### FR4.2: 추적 메트릭
**기본 메트릭**:
- Train/Val Loss
- ROUGE-1, ROUGE-2, ROUGE-L (F1/Precision/Recall)
- Learning Rate (epoch별)
- Gradient Norm

**상세 메트릭**:
- 샘플 예측 결과 (Table 형식, 각 epoch마다 5개)
- 하이퍼파라미터 비교 대시보드
- Confusion Matrix (길이별 성능)

#### FR4.3: 체크포인트 관리
- Best model (ROUGE 기준)
- Last checkpoint
- Epoch별 체크포인트 (save_total_limit=5)

---

### Phase 5: 자동화 파이프라인

#### FR5.1: 학습 자동화 스크립트
```python
# scripts/train.py
- Config 로드
- 데이터 전처리
- 모델 학습 (W&B 추적)
- Best model 저장
- 결과 로깅
```

#### FR5.2: 추론 자동화 스크립트
```python
# scripts/inference.py
- Best model 로드
- Test 데이터 추론
- CSV 생성 (,fname,summary 형식)
- 검증 (499 samples, index 포함)
```

#### FR5.3: 수동 검증 체크리스트
- [ ] CSV 포맷 확인 (`head -5 prediction/output.csv`)
- [ ] 샘플 수 확인 (`wc -l prediction/output.csv` = 500)
- [ ] Index 컬럼 존재 확인
- [ ] 특수 토큰 제거 확인
- [ ] 수동으로 5개 샘플 확인

---

### Phase 6: CausalLM 재검증 (Baseline 50점 달성 후)

#### FR6.1: 검증된 전처리 적용
- Phase 2-3에서 검증된 전처리를 Korean_DCS_2024에 적용
- 노이즈 처리, 특수 토큰 최적화

#### FR6.2: 하이퍼파라미터 신중한 튜닝
- Learning rate: 2e-5 기준으로 작은 범위 탐색
- Epoch: 5 → 3, 7 실험
- Batch size/Gradient accumulation 조정

#### FR6.3: A/B 테스트
- KoBART best model vs CausalLM best model
- Dev 평가 → 격차 분석 → 유망 시 Test 제출

---

### Phase 7: Git/문서 자동 최신화

#### FR7.1: Git 자동 커밋 (선택적)
- 실험 완료 시 자동 커밋 옵션
- 커밋 메시지: `Experiment #{N}: {description} - {score}`
- 수동 확인 후 Push

#### FR7.2: 실험 로그 자동 업데이트
- `experiment_logs.md`에 자동 추가
- 형식: 실험 번호, 변경사항, 설정, 결과, 다음 단계

#### FR7.3: README 업데이트
- Best score 기록
- 성능 개선 그래프 (선택적)

---

### Phase 8: 최종 제출

#### FR8.1: 최종 모델 선택
- Dev/Test 격차 5점 이내
- Test 50점 이상
- 재현 가능성 확인

#### FR8.2: 제출 파일 생성
- 최종 CSV 생성 및 다중 검증
- sample_submission.csv와 포맷 일치 확인

#### FR8.3: 코드 정리
- 재현 스크립트 작성
- 요구사항 문서 업데이트
- 트러블슈팅 가이드 작성

---

## 5. Non-Goals (Out of Scope)

### 명시적으로 제외되는 항목

❌ **Phase 1 완료 전 복잡한 개선 시도**
- Baseline 47점 달성 전 LLM, 앙상블, 새 모델 시도 금지

❌ **한 번에 여러 개선사항 적용**
- 반드시 하나씩 테스트 (디버깅 가능성 유지)

❌ **Dev set 점수만 보고 만족**
- Test 검증 없이 진행 금지

❌ **DialogSum 데이터셋 사용**
- 직접/간접 사용 모두 금지 (대회 규정)

❌ **유료 API 사용**
- Solar 제외, 다른 유료 API 금지

❌ **평가 데이터를 학습에 활용**
- 분석만 가능, Label 생성 학습 금지

❌ **완전 자동화 제출**
- 제출 전 반드시 수동 검증

---

## 6. Design Considerations

### 6.1 디렉토리 구조
```
/Competition/NLP/
├── naturallanguageprocessingcompetition-nlp-1/  # Git 저장소
│   ├── code/
│   │   ├── baseline.ipynb
│   │   ├── config.yaml
│   │   └── requirements.txt
│   ├── scripts/                                  # 새로 추가
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── preprocess.py
│   │   └── augment.py
│   ├── checkpoints/
│   ├── prediction/
│   └── logs/
├── docs/
│   ├── experiment_logs.md                        # 새로 추가
│   └── ...
└── tasks/
    ├── prd-dialogue-summarization-performance-improvement.md  # 본 문서
    └── tasks-prd-dialogue-summarization.md                    # 생성 예정
```

### 6.2 실험 로그 템플릿
```markdown
## 실험 #N: [실험명]

**날짜**: 2025-10-__
**베이스**: Baseline / 실험 #(N-1)

### 변경사항
- [한 가지만 명시]

### 설정
```yaml
[변경된 파라미터만]
```

### 결과
- Baseline/이전: XX.XX
- 현재 Dev: XX.XX
- 현재 Test: XX.XX (제출한 경우)
- **Dev/Test 격차**: XX.XX
- **변화**: +X.XX ✅/❌

### 판단
- [유지/롤백/재시도]
- [이유]

### 다음 단계
- [다음에 시도할 것]
```

### 6.3 W&B 대시보드 구성
- **Overview**: 전체 실험 비교 (ROUGE 점수)
- **Training**: Loss, Learning rate, Gradient norm
- **Evaluation**: ROUGE-1/2/L, Dev/Test 격차
- **Samples**: 예측 결과 샘플 (5개)
- **Hyperparameters**: 파라미터 비교 테이블

---

## 7. Technical Considerations

### 7.1 환경
- GPU: RTX 3090 24GB
- Python: 3.10
- PyTorch: 2.5.1
- Transformers: 4.46.3 (Baseline)

### 7.2 주요 의존성
```
transformers==4.46.3
rouge==1.0.1
wandb==0.16.1
pandas==2.1.4
torch==2.5.1
tqdm==4.66.1
```

### 7.3 디스크 용량 관리
- ⚠️ **150GB 제한 절대 준수**
- 모든 run 전 `du -sh / 2>/dev/null` 확인
- 체크포인트: save_total_limit=5
- 예측 결과: 이전 버전 삭제

### 7.4 제출 횟수 관리
- Daily 12회 제한
- 전략: Dev로 3-4개 실험 → Test 1회
- 추적: 스프레드시트/문서로 제출 이력 관리

### 7.5 Dev/Test 격차 모니터링
- 목표: 격차 5점 이내
- 경고: 격차 10점 이상 시 과적합 의심
- 조치: 정규화, Dropout, Early stopping 강화

### 7.6 재현성 보장
- Random seed: 42 (고정)
- 모든 설정 YAML/config 파일로 관리
- Git commit hash 기록
- 환경 정보 저장 (`pip freeze > requirements_frozen.txt`)

---

## 8. Success Metrics

### 8.1 주요 성공 지표

**Metric 1: Test ROUGE-F1 점수**
- 목표: **50점 이상**
- 측정: 대회 플랫폼 제출
- 빈도: 3-4 실험마다 1회

**Metric 2: Dev/Test 격차**
- 목표: **5점 이내**
- 측정: Dev 점수 - Test 점수 (절대값)
- 빈도: 매 Test 제출 시

**Metric 3: 제출 효율성**
- 목표: **12회 제출 내 50점 달성**
- 측정: 제출 횟수 추적
- 평가: 제출당 평균 개선폭 (+0.5점 이상)

### 8.2 부가 성공 지표

**Metric 4: 재현성**
- 목표: 동일 설정으로 ±0.5점 이내 재현
- 측정: 3회 재학습 후 표준편차
- 빈도: 최종 모델 선정 시

**Metric 5: 실험 속도**
- 목표: 1회 실험(학습+평가) 30분 이내
- 측정: W&B run time
- 개선: 불필요한 로깅 제거, 배치 최적화

**Metric 6: 문서화 완성도**
- 목표: 모든 실험 로그 기록
- 측정: `experiment_logs.md` 항목 수
- 평가: 누락 없음

### 8.3 최종 목표
- **대회 순위**: 1등 (또는 상위 3위)
- **학습 성과**: 재현 가능한 성능 개선 파이프라인 구축
- **지식 축적**: 트러블슈팅 가이드 및 베스트 프랙티스 문서화

---

## 9. Open Questions

### Q1: 하이퍼파라미터 탐색 범위
- Learning rate: 5e-5 외에 1e-4, 2e-5도 시도?
- Epoch: 30, 40까지 늘려볼지?
- → 답변: Phase 2.1 결과 보고 결정

### Q2: 데이터 증강 규모
- 2배 vs 3배 증강 중 선택?
- 증강 품질 검증 방법?
- → 답변: 소량 테스트(1,000개) 후 결정

### Q3: CausalLM 재시도 조건
- KoBART 50점 달성 시 무조건 시도?
- 시간 부족 시 스킵?
- → 답변: 대회 마감 7일 전까지 50점 달성 시 시도

### Q4: 앙상블 전략
- KoBART + CausalLM 앙상블 고려?
- 규칙 기반 후처리 + 모델 조합?
- → 답변: 50점 달성 후 검토

### Q5: W&B 공개 설정
- Public 프로젝트로 설정? (다른 참가자 볼 수 있음)
- Private 유지?
- → 답변: Private 유지 (대회 종료 후 공개 고려)

### Q6: Git 브랜치 전략
- main: 안정 버전
- experiment: 실험용 브랜치 분리?
- → 답변: 단순하게 main만 사용 (각 실험은 commit으로 관리)

### Q7: 비상 시나리오
- 50점 달성 실패 시 대안?
  - A: 47점 안정화 + 코드 품질 향상
  - B: 앙상블로 48-49점 목표
  - C: 완전히 새로운 접근 (mBART 등)
- → 답변: A 우선 (재현 가능성 최우선)

---

## 10. Implementation Timeline (예상)

### Week 1: Foundation
- Day 1-2: Baseline 재현 + 실험 시스템 구축
- Day 3-4: Phase 2.1-2.2 (하이퍼파라미터, 노이즈 처리)
- Day 5-7: Phase 2.3-2.5 (특수 토큰, Generation, 후처리)

### Week 2: Optimization
- Day 8-10: Phase 3 (데이터 증강)
- Day 11-12: Phase 4 (W&B 최적화)
- Day 13-14: 중간 점검 및 재조정

### Week 3: Advanced & Finalization
- Day 15-17: Phase 6 (CausalLM 재검증, 조건부)
- Day 18-19: 최종 모델 선정 및 검증
- Day 20-21: 코드 정리, 문서 완성, 제출

---

## Appendix: 체크리스트

### 실험 전 체크리스트
- [ ] 디스크 용량 150GB 미만 확인
- [ ] Git 최신 상태 확인
- [ ] 이전 실험 로그 작성 완료
- [ ] Config 파일 백업
- [ ] 제출 횟수 확인 (12회 이내)

### 실험 중 체크리스트
- [ ] W&B run 시작 확인
- [ ] GPU 사용률 모니터링
- [ ] Loss 수렴 확인
- [ ] 샘플 예측 결과 확인

### 실험 후 체크리스트
- [ ] Dev ROUGE 점수 기록
- [ ] Dev/Test 격차 분석 (Test 제출 시)
- [ ] 실험 로그 작성
- [ ] Best model 체크포인트 확인
- [ ] Git 커밋 (변경사항 설명)
- [ ] 다음 실험 계획 수립

### Test 제출 전 체크리스트
- [ ] CSV 포맷 검증 (,fname,summary)
- [ ] 샘플 수 확인 (500 lines = header + 499)
- [ ] Index 컬럼 존재
- [ ] 특수 토큰 제거 확인
- [ ] 5개 샘플 수동 검토
- [ ] Dev 점수와 비교 분석
- [ ] 제출 횟수 체크

---

**작성일**: 2025-10-12
**버전**: 1.0
**작성자**: Claude Code + User
**다음 단계**: `generate-tasks.md`로 상세 Task List 생성