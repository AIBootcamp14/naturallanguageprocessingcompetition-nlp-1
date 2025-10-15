# 다음 단계 가이드

## 현재 상황

- **최고 점수**: 47.47점 (Phase 1, LP=0.5)
- **목표**: 50점 돌파
- **제출 횟수**: **12/12 사용 완료** ⚠️
- **다음 제출**: 일일 제한 리셋 후 신중하게 선택 필요

## 제안 실험 (우선순위순)

### 1. Learning Rate 튜닝 (우선순위: ⭐⭐⭐)

**실험 설계**:
```bash
cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code
python train.py --experiment exp8_lr3e5
```

**config/experiments.yaml 추가**:
```yaml
exp8_lr3e5:
  description: "증강 데이터 + LR 3e-5"
  general:
    output_dir: /Competition/NLP/.../submission_exp8_lr3e5
  training:
    learning_rate: 3.0e-05  # 3배 증가
    num_train_epochs: 20
    early_stopping_patience: 3
  data:
    use_weights: false
```

**근거**:
- 증강 데이터로 안정성 확보 (13,465개)
- 더 빠른 수렴 가능
- Exp #6에서 보류했던 실험

**예상 효과**: +0.5~1.5점
**리스크**: Low (LR 증가는 일반적인 튜닝)

---

### 2. 더 긴 학습 (우선순위: ⭐⭐)

**변경사항**:
```yaml
training:
  num_train_epochs: 30  # 20 → 30
  early_stopping_patience: 5  # 3 → 5
```

**근거**:
- 현재 Epoch 12에 조기 종료
- Loss가 아직 수렴하지 않음
- 더 학습할 여지 있음

**예상 효과**: +0.3~0.8점
**리스크**: Medium (과적합 가능성)

---

### 3. 균등 증강 (우선순위: ⭐⭐⭐)

**작업 단계**:

1. **추가 증강 데이터 생성**:
   - 노동/고용: 365개 추가 (135 → 500)
   - 감정 지원: 170개 추가 (130 → 300)
   - 친구 상담: 156개 추가 (144 → 300)
   - 일상 대화: 92개 추가 (208 → 300)
   - 기타 소규모 카테고리들

2. **균등 분포로 학습** (가중치 없음):
   ```python
   # WeightedRandomSampler 사용 안 함
   # 자연스러운 분포로 학습
   ```

**근거**:
- WeightedSampler 실패 원인 해결
- 모든 카테고리가 충분한 샘플 수
- 자연스러운 분포로 일반화 향상

**예상 효과**: +0.8~2.0점
**리스크**: Low (균등 분포는 안전)

---

### 4. 모델 크기 증가 (우선순위: ⭐)

**후보 모델**:
- `gogamza/kobart-base-v2` (더 큰 KoBART)
- `KETI-AIR/ke-t5-base` (T5 기반)
- `psyche/KoT5-summarization` (T5 요약 특화)

**근거**:
- 더 큰 모델 = 더 높은 표현력
- 13,465개 데이터면 충분히 학습 가능

**예상 효과**: +1.0~3.0점
**리스크**: High (학습 시간 3배, 메모리 부족 가능)

---

## 추천 진행 순서

### Week 1 (현재)
- [x] 문서화 완료
- [x] GitHub 백업
- [ ] 증강 데이터 추가 생성 (균등 증강)

### Week 2
1. **Exp #8**: 균등 증강 + LR 3e-5
2. 결과 분석 및 제출 (신중하게)
3. Loss Gap 양수 확인

### Week 3
1. 최적 설정 확정
2. 최종 모델 학습
3. 앙상블 고려 (여러 체크포인트)

---

## 제출 전 체크리스트

실험 완료 후 제출하기 전에 **반드시** 확인:

- [ ] **Loss Gap이 양수**인가? (+0.15 이상 권장)
- [ ] Train Loss와 Eval Loss 추이를 분석했는가?
- [ ] Dev ROUGE보다 **Loss Gap**을 우선 확인했는가?
- [ ] 최소 2개 이상의 체크포인트를 비교했는가?
- [ ] 이전 실험과 설정 차이를 명확히 이해했는가?
- [ ] 실패 시 롤백 계획이 있는가?
- [ ] 제출 횟수 제한을 고려했는가? (12/day)

---

## 피해야 할 함정

### 1. Dev ROUGE 맹신 ❌
```
Dev ROUGE 36.43% → Test 46.62 (실패)
Dev ROUGE 36.18% → Test 47.41 (성공)
```
**교훈**: Loss Gap이 더 신뢰성 높음!

### 2. 제출 횟수 낭비 ❌
- 일일 12회 제한
- 신중하게 사용 (Loss Gap 확인 후)
- 실험 실패 시 제출하지 말 것

### 3. WeightedRandomSampler 과용 ❌
```
증강 없는 카테고리 (135개) × 3.70배 가중치
= 같은 샘플 500회 반복
= 암기 발생
= Test 실패
```
**교훈**: 증강 없으면 가중치 주지 말 것!

### 4. 하이퍼파라미터 동시 변경 ❌
```
잘못된 예: LR + Epochs + Batch Size 동시 변경
→ 무엇이 효과적인지 알 수 없음

올바른 예: LR만 변경 → 결과 확인 → 다음 파라미터
```

---

## 실험 템플릿

새로운 실험을 시작할 때 사용:

### Exp #X: [실험명]

**날짜**: YYYY-MM-DD
**변경사항**: [무엇을 바꿨는지]
**근거**: [왜 바꿨는지]

**하이퍼파라미터**:
```yaml
training:
  learning_rate: X.Xe-XX
  num_train_epochs: XX
  # ...
```

**예상 효과**: +X.X~X.X점
**리스크**: Low/Medium/High

**학습 결과**:
- Loss Gap: +X.XX
- Dev ROUGE-1: XX.XX%
- Best checkpoint: checkpoint-XXXX

**Test 결과**: XX.XX점

**인사이트**: [배운 점]

---

## 참고 링크

- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) - 전체 실험 이력
- [프레임워크 가이드](../code/README.md) - 프레임워크 사용법
- [Competition_Overview](Competition_Overview/) - 대회 규칙

---

## 긴급 연락

문제 발생 시:
1. GitHub Issues 생성
2. EXPERIMENT_LOG.md에 기록
3. 롤백 후 재시도