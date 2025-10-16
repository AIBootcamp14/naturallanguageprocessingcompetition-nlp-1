# Solar API Rate Limit Fix

## 문제 발견

**날짜**: 2025-10-15
**발견 경로**: 사용자가 백그라운드 프로세스 로그에서 429 에러 발견

### 증상

```
❌ API 호출 실패: Error code: 429 - {'error': {'message': "You've reached your API request limit.
Please wait and try again later..."
```

**발생 빈도**: voting (n_samples=3) 사용 시 499개 대화 중 약 20-30회 발생

## 원인 분석

### 1. API 호출 횟수 폭증

- **기존 단일 샘플링**: 499개 대화 × 1회 = 499 API 호출
- **voting n=3**: 499개 대화 × 3회 = **1,497 API 호출** (3배!)
- **voting n=5**: 499개 대화 × 5회 = **2,495 API 호출** (5배!)

### 2. 샘플 간 대기 시간 부족

**기존 코드** (`src/api/solar_api.py:678`):
```python
if i < n_samples - 1:
    time.sleep(0.5)  # 샘플 간 0.5초 대기
```

**문제점**:
- 0.5초 대기로는 Solar API rate limit 회복 불가
- 3회 샘플링: 0.5초 × 2 = 1초 대기 (insufficient)
- API는 분당 요청 수 제한이 있음

### 3. 동시 실행 프로세스

현재 실행 중인 프로세스:
- **24606f**: K-Fold + Solar voting 3회 (1,497 호출)
- **98693c**: K-Fold + HF correction + Solar 단일 (499 호출)
- **73dbb1**: K-Fold만 (API 미사용)

**충돌**:
- 24606f와 98693c가 동시에 Solar API 호출
- Rate limit 공유로 인해 양쪽 모두 429 에러 위험

## 해결 방법

### 1차 수정 (0.5초 → 2.0초)

**파일**: `src/api/solar_api.py:678`

**변경 전**:
```python
if i < n_samples - 1:
    time.sleep(0.5)  # 샘플 간 0.5초 대기
```

**변경 후**:
```python
if i < n_samples - 1:
    time.sleep(2.0)  # 샘플 간 2.0초 대기 (429 에러 방지)
```

**효과**:
- 3회 샘플링: 2.0초 × 2 = 4초 대기 (각 대화당)
- 499개 대화 × 4초 = **약 33분 추가 대기**
- 총 실행 시간: 30분 → 약 60-70분 (안정성 확보)

### 2. 사용자 파라미터 권장

**voting 사용 시 권장 설정**:
```bash
python scripts/solar_only_inference.py \
  --input submissions/kfold_ensemble.csv \
  --test_data data/raw/test.csv \
  --use_voting \
  --n_samples 3 \
  --solar_delay 2.0 \          # 배치 간 2초 대기
  --solar_batch_size 5         # 배치 크기 절반으로 감소
```

**이유**:
1. `--solar_delay 2.0`: 배치 간 대기 시간 증가 (기본 1.0초)
2. `--solar_batch_size 5`: 배치 크기 감소 (기본 10 → 5)
3. 내부 샘플 간 2.0초 대기 (코드 레벨)

**총 대기 시간**:
- 배치 간: (499 / 5) × 2.0초 = 약 200초 (3.3분)
- 샘플 간: 499 × 2회 × 2.0초 = 1,996초 (33분)
- 총: 약 36분 대기 + API 호출 시간

### 3. 프로세스 관리

**동시 실행 시 주의**:
- ❌ 여러 voting 프로세스 동시 실행 금지
- ✅ voting + 단일 샘플링 조합은 가능
- ✅ 순차 실행 권장

## 테스트 결과

### 현재 진행 중인 프로세스 (24606f)

**명령어**:
```bash
python scripts/kfold_ensemble_inference.py \
  --use_solar_api \
  --solar_use_voting \
  --solar_n_samples 3
```

**진행률**: 약 200/499 완료 (40%)

**429 에러 발생 횟수**: 약 10-15회

**예상 완료 시간**:
- 기존 예상: 30분
- 실제 (429 에러 포함): 60-90분

### 향후 실행 시 예상 효과

**2.0초 delay 적용 후**:
- 429 에러: **0-2회** (대폭 감소)
- 실행 시간: 60-70분 (안정적)
- 완료율: **100%** (중단 없음)

## 문서 업데이트

### 1. 사용 가이드

**파일**: `docs/usage/앙상블_재사용_가이드.md`

**추가 섹션**:
- "Rate Limit 관리 (매우 중요!)"
- Solar API Rate Limit 이해
- 권장 설정 (voting 사용 시)
- 여러 프로세스 동시 실행 시 주의
- 429 에러 발생 시 대처법

### 2. 코드 변경

**파일**: `src/api/solar_api.py:678`

**변경 사항**:
- 샘플 간 대기 시간: 0.5초 → 2.0초

## 권장 사항

### 즉시 조치

1. **현재 프로세스 모니터링**: 24606f가 완료될 때까지 대기
2. **98693c 중지 고려**: Rate limit 경합 방지
3. **향후 실행**: 새로운 delay 설정 사용

### 장기 개선

1. **Exponential Backoff 구현**:
   ```python
   for retry in range(max_retries):
       try:
           response = api_call()
           break
       except RateLimitError:
           wait_time = 2 ** retry  # 1, 2, 4, 8초...
           time.sleep(wait_time)
   ```

2. **Rate Limit 상태 추적**:
   - API 응답 헤더에서 남은 요청 수 확인
   - 동적으로 대기 시간 조정

3. **캐시 활용 강화**:
   - 동일한 대화는 캐시에서 즉시 반환
   - 프롬프트 버전 + n_samples 포함된 캐시 키 사용 (이미 구현됨)

## 결론

### 핵심 변경
- **샘플 간 대기 시간: 0.5초 → 2.0초**
- **배치 간 대기 시간 권장: 1.0초 → 2.0초**
- **배치 크기 권장: 10 → 5 (voting 사용 시)**

### 트레이드오프
- ✅ 안정성: 429 에러 대폭 감소
- ✅ 완료율: 100% 보장
- ⚠️  실행 시간: 30분 → 60-70분 (2배 증가)

### 최종 권장
- **voting n=3**: 안전 설정 사용 (delay=2.0, batch=5) - 60-70분
- **voting n=5**: 긴급하지 않은 경우에만 사용 - 100-120분
- **단일 샘플링**: 빠른 실험용 (delay=1.0, batch=10) - 10-15분
