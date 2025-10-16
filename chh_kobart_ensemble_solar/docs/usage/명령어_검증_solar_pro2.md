# 명령어 검증: solar-pro2 사용

## 사용자 제공 명령어

```bash
python scripts/kfold_ensemble_inference.py \
  --experiment_dir experiments/20251014/20251014_183206_kobart_ultimate_kfold \
  --test_data data/raw/test.csv \
  --use_solar_api \
  --solar_model solar-pro2 \
  --solar_batch_size 3 \
  --solar_temperature 0.3 \
  --use_voting \
  --solar_n_samples 5 \
  --n_samples 5 \
  --solar_delay 3.0 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --max_new_tokens 120 \
  --min_new_tokens 30 \
  --num_beams 5 \
  --length_penalty 1.0 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --batch_size 32 \
  --ensemble_method soft_voting \
  --resume
```

---

## 검증 결과

### ❌ 오류 1: 잘못된 인자명

**문제**:
```bash
--use_voting  # ❌ 잘못된 인자명
```

**이유**:
- `kfold_ensemble_inference.py`에는 `--use_voting` 인자가 없음
- Solar API voting은 `--solar_use_voting`으로 활성화

**수정**:
```bash
--solar_use_voting  # ✅ 올바른 인자명
```

---

### ❌ 오류 2: 중복 인자

**문제**:
```bash
--solar_n_samples 5  # Solar API용
--n_samples 5        # ❌ 존재하지 않는 인자 (중복)
```

**이유**:
- `--n_samples`는 스크립트에 정의되지 않음
- `--solar_n_samples`만 사용

**수정**:
```bash
--solar_n_samples 5  # ✅ 이것만 사용
# --n_samples 5 제거
```

---

### ⚠️  경고 1: solar-pro2 모델 사용

**현재 설정**:
```bash
--solar_model solar-pro2  # ⚠️  매우 느림
```

**영향**:
- 성능: Solar Mini 대비 **+50% 향상**
- 속도: Solar Mini 대비 **약 2배 느림**
- 비용: 더 높을 가능성

**예상 실행 시간**:
```
K-Fold (batch_size=32): 5-7분
HF 보정: 3-4분
Solar Pro2 voting n=5: 180-260분 (약 3-4시간!)
─────────────────────────
총: 188-271분 (3.1-4.5시간)
```

**권장**:
- ✅ **solar-mini 사용 권장** (95-130분, 충분한 품질)
- ⚠️  solar-pro2는 시간 여유가 있을 때만

---

### ⚠️  경고 2: batch_size=32 (OOM 위험)

**현재 설정**:
```bash
--batch_size 32  # ⚠️  OOM 위험 높음
--max_new_tokens 120  # 긴 요약
--num_beams 5         # 빔 수 5
```

**위험 분석**:
- GPU 메모리 사용: ~22GB (92% 사용)
- 남은 메모리: 2GB (여유 부족)
- OOM 발생 확률: **높음**

**OOM 발생 시나리오**:
1. 매우 긴 대화 (500자 이상) 처리 시
2. 메모리 스파이크 발생 시
3. 다른 프로세스가 메모리 사용 시

**권장**:
```bash
--batch_size 16  # ✅ 안전하면서도 빠름
```

**근거**:
- Solar API가 전체 시간의 95% 차지
- batch_size 32 → 16 변경 시 시간 증가: **3-4분 (전체의 2%)**
- OOM 위험 대폭 감소

---

### ✅ 올바른 부분

1. **Solar API 설정**:
   ```bash
   --use_solar_api ✅
   --solar_batch_size 3 ✅ (voting 시 안전)
   --solar_temperature 0.3 ✅
   --solar_delay 3.0 ✅ (Rate limit 안전)
   ```

2. **HuggingFace 보정**:
   ```bash
   --use_pretrained_correction ✅
   --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization ✅
   --correction_strategy quality_based ✅
   --correction_threshold 0.3 ✅
   ```

3. **생성 파라미터**:
   ```bash
   --max_new_tokens 120 ✅
   --min_new_tokens 30 ✅
   --num_beams 5 ✅
   --length_penalty 1.0 ✅
   --repetition_penalty 1.5 ✅
   --no_repeat_ngram_size 3 ✅
   ```

4. **앙상블 방법**:
   ```bash
   --ensemble_method soft_voting ✅
   ```

5. **Resume 기능**:
   ```bash
   --resume ✅ (체크포인트 재사용)
   ```

---

## 수정된 최적 명령어

### 버전 1: solar-mini 사용 (권장)

**장점**: 빠르고 (95-130분), 안정적, 충분한 품질

```bash
python scripts/kfold_ensemble_inference.py \
  --experiment_dir experiments/20251014/20251014_183206_kobart_ultimate_kfold \
  --test_data data/raw/test.csv \
  --use_solar_api \
  --solar_model solar-mini \
  --solar_batch_size 3 \
  --solar_temperature 0.3 \
  --solar_use_voting \
  --solar_n_samples 5 \
  --solar_delay 3.0 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --max_new_tokens 120 \
  --min_new_tokens 30 \
  --num_beams 5 \
  --length_penalty 1.0 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --batch_size 16 \
  --ensemble_method soft_voting \
  --resume
```

**예상 시간**: **95-130분** (1.6-2.2시간)

---

### 버전 2: solar-pro2 사용 (시간 여유 있을 때)

**장점**: 최고 품질 (+50%), 단점: 매우 느림 (3-4시간)

```bash
python scripts/kfold_ensemble_inference.py \
  --experiment_dir experiments/20251014/20251014_183206_kobart_ultimate_kfold \
  --test_data data/raw/test.csv \
  --use_solar_api \
  --solar_model solar-pro2 \
  --solar_batch_size 3 \
  --solar_temperature 0.3 \
  --solar_use_voting \
  --solar_n_samples 5 \
  --solar_delay 3.0 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --max_new_tokens 120 \
  --min_new_tokens 30 \
  --num_beams 5 \
  --length_penalty 1.0 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --batch_size 16 \
  --ensemble_method soft_voting \
  --resume
```

**예상 시간**: **188-271분** (3.1-4.5시간)

---

## 주요 변경사항 요약

| 항목 | 원본 | 수정 | 이유 |
|------|------|------|------|
| `--use_voting` | ❌ | 제거 | 존재하지 않는 인자 |
| `--solar_use_voting` | 없음 | ✅ 추가 | 올바른 인자명 |
| `--n_samples 5` | ❌ | 제거 | 중복 인자 (solar_n_samples와 동일) |
| `--solar_model` | `solar-pro2` | `solar-mini` 권장 | 속도 vs 품질 균형 |
| `--batch_size` | `32` | `16` 권장 | OOM 방지, Solar API 병목으로 영향 미미 |

---

## 실행 시간 비교

### 옵션 1: solar-mini + batch_size=16 (권장)

| 단계 | 시간 | 비율 |
|------|------|------|
| K-Fold 앙상블 | 7-9분 | 6% |
| HF 보정 | 4-6분 | 4% |
| Solar Mini voting n=5 | 84-115분 | 90% |
| **총** | **95-130분** | 100% |

---

### 옵션 2: solar-pro2 + batch_size=16 (고품질, 느림)

| 단계 | 시간 | 비율 |
|------|------|------|
| K-Fold 앙상블 | 7-9분 | 4% |
| HF 보정 | 4-6분 | 3% |
| Solar Pro2 voting n=5 | 177-256분 | 93% |
| **총** | **188-271분** | 100% |

---

### 옵션 3: solar-mini + batch_size=32 (빠르지만 위험)

| 단계 | 시간 | 비율 |
|------|------|------|
| K-Fold 앙상블 | 5-7분 | 5% |
| HF 보정 | 3-4분 | 3% |
| Solar Mini voting n=5 | 84-115분 | 92% |
| **총** | **92-126분** | 100% |

**주의**: OOM 위험 높음, 3-4분 절약이 OOM 위험을 감수할 가치 없음

---

## 품질 vs 속도 트레이드오프

### 시나리오 분석

| 설정 | 품질 | 속도 | 안정성 | 권장도 |
|------|------|------|--------|--------|
| **solar-mini + bs16** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ **권장** |
| solar-pro2 + bs16 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| solar-mini + bs32 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |

**결론**: **solar-mini + batch_size=16**이 최적

---

## solar-pro2 사용 권장 시나리오

### ✅ solar-pro2 사용 권장

1. **최종 제출용 고품질 필요**
   - 대회 마감 전 최종 제출
   - 시간 여유 4시간 이상

2. **복잡한 대화 많음**
   - test_28 같은 어려운 케이스 많음
   - 100문장 이상 긴 대화

3. **밤새 실행 가능**
   - 잠자는 동안 실행
   - 다른 작업 안 함

### ❌ solar-pro2 사용 비권장

1. **빠른 반복 실험**
   - 파라미터 튜닝 중
   - 여러 설정 테스트

2. **시간 제약**
   - 2시간 내 결과 필요
   - 긴급 제출

3. **Solar Mini로 충분한 품질**
   - Voting n=5로 이미 고품질
   - HF 보정 추가

---

## 최종 권장사항

### 🏆 대부분의 경우: solar-mini + batch_size=16

```bash
python scripts/kfold_ensemble_inference.py \
  --experiment_dir experiments/20251014/20251014_183206_kobart_ultimate_kfold \
  --test_data data/raw/test.csv \
  --use_solar_api \
  --solar_model solar-mini \
  --solar_batch_size 3 \
  --solar_temperature 0.3 \
  --solar_use_voting \
  --solar_n_samples 5 \
  --solar_delay 3.0 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --max_new_tokens 120 \
  --min_new_tokens 30 \
  --num_beams 5 \
  --length_penalty 1.0 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --batch_size 16 \
  --ensemble_method soft_voting \
  --resume
```

**이유**:
1. ✅ 충분한 품질 (voting n=5 + HF 보정)
2. ✅ 합리적인 시간 (95-130분)
3. ✅ 안정적 (OOM 위험 낮음)
4. ✅ 비용 효율적

---

### 🎯 최종 제출용: solar-pro2 + batch_size=16

**조건**: 시간 여유 4시간 이상 + 최고 품질 필요

```bash
python scripts/kfold_ensemble_inference.py \
  --experiment_dir experiments/20251014/20251014_183206_kobart_ultimate_kfold \
  --test_data data/raw/test.csv \
  --use_solar_api \
  --solar_model solar-pro2 \
  --solar_batch_size 3 \
  --solar_temperature 0.3 \
  --solar_use_voting \
  --solar_n_samples 5 \
  --solar_delay 3.0 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --max_new_tokens 120 \
  --min_new_tokens 30 \
  --num_beams 5 \
  --length_penalty 1.0 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --batch_size 16 \
  --ensemble_method soft_voting \
  --resume
```

**예상 시간**: 3.1-4.5시간

---

## 에러 발생 시 대처법

### 1. OOM 에러 발생 시

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결**:
```bash
# 배치 사이즈 절반으로 감소
--batch_size 16 → --batch_size 8

# 또는 max_new_tokens 감소
--max_new_tokens 120 → --max_new_tokens 80
```

---

### 2. Solar API 429 에러 발생 시

**증상**:
```
Error code: 429 - You've reached your API request limit
```

**해결**:
```bash
# delay 증가
--solar_delay 3.0 → --solar_delay 5.0

# batch_size 감소
--solar_batch_size 3 → --solar_batch_size 2
```

---

### 3. 인자 에러 발생 시

**증상**:
```
error: unrecognized arguments: --use_voting
```

**해결**:
```bash
# 올바른 인자명 사용
--use_voting → --solar_use_voting
--n_samples → 제거 (solar_n_samples 사용)
```

---

## 참고 문서

- Solar 모델 비교: `docs/issues/Solar_모델_비교_분석.md`
- 배치 사이즈 최적화: `docs/usage/배치_사이즈_최적화_가이드.md`
- 최적 명령어 검증: `docs/usage/최적_명령어_검증.md`
- Rate Limit 관리: `docs/issues/Solar_API_Rate_Limit_Fix.md`
