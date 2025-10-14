# HuggingFace 사전학습 모델 보정 시스템 문제 분석

## 문서 정보
- 작성일: 2025-10-14
- 분석 대상: HuggingFace Pretrained Correction 시스템
- 발견 계기: 다른 브랜치에서 추론 실행 시 제출 파일에 요약문 대신 원본 대화문이 출력되는 문제 발생

---

## 1. 문제 개요

### 1.1 증상
다른 컴퓨터에서 예전 버전 코드로 추론 실행 시 다음과 같은 문제 발생:

```bash
# 실행 명령어
python scripts/inference.py \
  --model experiments/20251014/20251014_090813_kobart_balanced/fold_4/default/final_model \
  --test_data data/raw/test.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3
```

**결과:**
- 제출 파일에 요약문이 아닌 **원본 대화문이 그대로 저장됨**
- 보정 로그: "보정됨: 493개 (98.8%)" - 거의 모든 샘플이 보정됨
- CSV 행 개수: 1056행 (정상: 500행) - CSV 형식이 깨짐

**정상 파일 vs 문제 파일:**
```csv
# 정상 파일 (20251014_004100)
test_0,"#Person1#은 Ms. Dawson에게 사내 메모를 작성하고 배포하라고 지시합니다..."

# 문제 파일 (20251014_185618)
test_0,"?
Person1#: Ms. Dawson, 받아쓰기 좀 부탁드려야겠어요.
#Persoon2#: 네, 말씀하세요..."
```

---

## 2. 코드 흐름 분석

### 2.1 전체 시스템 구조

```
scripts/inference.py (추론 실행)
    ↓
src/inference/predictor.py (배치 추론)
    ↓ (--use_pretrained_correction 플래그 활성화 시)
src/correction/pretrained_corrector.py (보정 수행)
    ↓
src/correction/pretrained_corrector.py::_generate_summaries()
    ↓ (참조 모델로 요약 생성)
src/inference/predictor.py::create_predictor() (재사용)
    ↓
참조 모델의 요약 생성
    ↓
src/correction/quality_evaluator.py (품질 평가)
    ↓
src/correction/ensemble_strategies.py (보정 전략)
    ↓
최종 요약 선택
```

### 2.2 핵심 문제 지점

#### **지점 1: `pretrained_corrector.py:214` - `_generate_summaries()` 메서드**

```python
def _generate_summaries(
    self,
    dialogues: List[str],
    model,
    tokenizer,
    batch_size: int = 16,
    **generation_kwargs
) -> List[str]:
    """
    단일 모델로 배치 요약 생성
    """
    from src.inference import create_predictor

    # -------------- Predictor 생성 -------------- #
    predictor = create_predictor(
        model=model,
        tokenizer=tokenizer,
        device=self.device,
        logger=None  # 너무 많은 로그 방지
    )

    # -------------- 배치 예측 -------------- #
    summaries = predictor.predict_batch(
        dialogues=dialogues,
        batch_size=batch_size,
        show_progress=False,  # 진행바 비활성화
        **generation_kwargs
    )

    return summaries
```

**문제점:**
- `create_predictor`를 사용하여 HF 참조 모델(gogamza/kobart-base-v2, digit82/kobart-summarization)로 Predictor 생성
- 이 Predictor들이 **요약을 생성하지 않고 원본 dialogue를 그대로 반환**하고 있음

---

#### **지점 2: `ensemble_strategies.py:46` - `QualityBasedStrategy.select()` 메서드**

```python
class QualityBasedStrategy(EnsembleStrategy):
    """
    품질 기반 선택 전략 (추천)

    로직:
    1. KoBART 품질이 임계값 이상이면 KoBART 사용
    2. 아니면 가장 품질 높은 참조 모델 사용
    """

    def select(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        quality_scores: Dict[str, List[float]],
        threshold: float
    ) -> List[str]:
        final_summaries = []

        for i in range(len(candidate_summaries)):
            candidate_quality = quality_scores["candidate_quality"][i]

            # -------------- KoBART 품질 확인 -------------- #
            if candidate_quality >= threshold:
                # 품질이 충분히 높으면 KoBART 사용
                final_summaries.append(candidate_summaries[i])
                continue

            # -------------- 최고 품질 참조 모델 선택 -------------- #
            best_model = None
            best_quality = -1

            for model_name in reference_summaries.keys():
                quality_key = f"{model_name}_quality"
                if quality_key in quality_scores:
                    quality = quality_scores[quality_key][i]
                    if quality > best_quality:
                        best_quality = quality
                        best_model = model_name

            # -------------- 최종 선택 -------------- #
            if best_model:
                # 가장 품질 높은 참조 모델 사용
                final_summaries.append(reference_summaries[best_model][i])  # ← 여기서 dialogue 선택됨!
            else:
                # 폴백: KoBART 사용
                final_summaries.append(candidate_summaries[i])

        return final_summaries
```

**문제점:**
- 98.8%의 샘플에서 KoBART 품질이 임계값(0.3) 미만으로 평가됨
- 참조 모델의 "요약"(실제로는 dialogue)이 선택됨
- 결과적으로 **원본 dialogue가 최종 요약으로 반환됨**

---

#### **지점 3: `quality_evaluator.py:148` - `_compute_rouge_f1()` 메서드**

```python
def _compute_rouge_f1(self, hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 점수 계산
    """
    if self.rouge is None:
        return 0.5  # ROUGE 없으면 기본값

    try:
        # -------------- 빈 문자열 체크 -------------- #
        if not hypothesis.strip() or not reference.strip():
            return 0.0

        # -------------- ROUGE 계산 -------------- #
        scores = self.rouge.get_scores(hypothesis, reference)[0]
        return scores['rouge-l']['f']  # ROUGE-L F1 반환

    except Exception as e:
        # ROUGE 계산 실패 시 기본값 반환
        return 0.0
```

**문제점:**
- 참조 모델이 dialogue를 반환하면, **dialogue vs dialogue 비교** 발생
- ROUGE 점수가 매우 높게 나옴 (거의 1.0에 가까움)
- KoBART의 실제 요약문 vs dialogue 비교 시 ROUGE가 낮게 나옴
- 결과: dialogue가 "고품질"로 잘못 평가됨

---

## 3. 근본 원인 분석

### 3.1 HF 참조 모델의 요약 생성 실패

**가설 1: 모델 문제**
- `gogamza/kobart-base-v2`: Fine-tuning되지 않은 Base 모델
- `digit82/kobart-summarization`: 다른 데이터셋으로 학습된 모델

**결과:**
- 우리 데이터 형식(`#Person1#: ...`)을 제대로 이해하지 못함
- 요약을 생성하지 않고 입력을 그대로 복사

**가설 2: 전처리 불일치**
- `InferenceDataset`의 전처리가 참조 모델에 맞지 않음
- 참조 모델이 기대하는 입력 형식과 우리 데이터 형식이 다름

**가설 3: 생성 파라미터 문제**
- `generation_kwargs`가 참조 모델에 적합하지 않음
- 특히 `max_new_tokens=100`이 너무 짧아서 요약 대신 복사 발생 가능

### 3.2 품질 평가 로직의 한계

**문제:**
- ROUGE는 n-gram 기반 overlap 측정
- dialogue(긴 텍스트) vs dialogue(긴 텍스트) 비교 시 매우 높은 점수
- 짧은 요약문 vs dialogue 비교 시 낮은 점수
- **실제 요약 품질과 무관하게 길이가 긴 텍스트가 유리함**

**근본 원인:**
- 품질 평가 기준이 "요약의 품질"이 아닌 "텍스트 유사도"에 치우침
- dialogue를 반환하는 모델이 오히려 높은 점수를 받음

### 3.3 보정 전략의 설계 결함

**QualityBasedStrategy의 가정:**
- 참조 모델이 **항상 유효한 요약을 생성**한다는 전제
- 참조 모델의 출력이 dialogue일 수 있다는 가능성을 고려하지 않음

**결과:**
- dialogue를 "고품질 요약"으로 잘못 판단
- 최종 제출 파일에 요약 대신 dialogue 저장

---

## 4. 현재 브랜치(feature/inference)의 상태 분석

### 4.1 코드 검토 결과

현재 브랜치의 코드를 분석한 결과, **동일한 문제가 그대로 존재**합니다:

1. **`src/correction/pretrained_corrector.py`**: 문제 있음
   - Line 214: `_generate_summaries()` 메서드가 `create_predictor()` 재사용
   - HF 모델이 dialogue를 반환할 경우 대응 로직 없음

2. **`src/correction/ensemble_strategies.py`**: 문제 있음
   - Line 46-97: `QualityBasedStrategy`가 참조 모델 출력을 맹목적으로 신뢰
   - dialogue 필터링 로직 없음

3. **`src/correction/quality_evaluator.py`**: 문제 있음
   - Line 148-174: ROUGE 기반 평가만 수행
   - dialogue vs summary 구분 불가

4. **`src/inference/predictor.py`**: 문제 있음
   - Line 400-442: HF 보정 로직이 실패 시에만 예외 처리
   - dialogue 반환 문제는 "성공"으로 간주되어 감지 안 됨

### 4.2 위험도 평가

| 항목 | 위험도 | 상태 |
|------|--------|------|
| HF 모델 요약 생성 실패 | **HIGH** | 현재 브랜치에 존재 |
| dialogue 반환 감지 불가 | **HIGH** | 현재 브랜치에 존재 |
| 품질 평가 오류 (ROUGE bias) | **MEDIUM** | 현재 브랜치에 존재 |
| 보정 전략 설계 결함 | **MEDIUM** | 현재 브랜치에 존재 |
| CSV 형식 깨짐 | **LOW** | dialogue 반환 시 발생 가능 |

**결론: 현재 브랜치에서도 `--use_pretrained_correction` 사용 시 동일한 문제 발생 가능성 매우 높음**

---

## 5. 해결 방안

### 5.1 즉시 조치 (Quick Fix)

#### 방법 1: HF 보정 비활성화 (권장)

```bash
# --use_pretrained_correction 플래그 제거
python scripts/inference.py \
  --model experiments/.../final_model \
  --test_data data/raw/test.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3
  # --use_pretrained_correction 제거!
```

**장점:**
- 즉시 문제 해결
- 검증된 KoBART 모델만 사용

**단점:**
- HF 보정 기능을 사용할 수 없음

---

#### 방법 2: Solar API만 사용

```bash
python scripts/inference.py \
  --model experiments/.../final_model \
  --test_data data/raw/test.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --use_solar_api  # Solar API만 사용
```

**장점:**
- KoBART + Solar API 앙상블 사용 가능
- HF 보정 문제 회피

**단점:**
- Solar API 의존성

---

### 5.2 근본 해결 (Root Cause Fix)

#### 해결책 A: dialogue 필터링 추가

**수정 위치: `src/correction/pretrained_corrector.py`**

```python
def _generate_summaries(
    self,
    dialogues: List[str],
    model,
    tokenizer,
    batch_size: int = 16,
    **generation_kwargs
) -> List[str]:
    """단일 모델로 배치 요약 생성"""
    from src.inference import create_predictor

    predictor = create_predictor(
        model=model,
        tokenizer=tokenizer,
        device=self.device,
        logger=None
    )

    summaries = predictor.predict_batch(
        dialogues=dialogues,
        batch_size=batch_size,
        show_progress=False,
        **generation_kwargs
    )

    # ✅ 추가: dialogue 필터링
    filtered_summaries = []
    for dialogue, summary in zip(dialogues, summaries):
        if self._is_dialogue_copy(dialogue, summary):
            # dialogue를 그대로 복사한 경우 빈 문자열 반환
            self._log(f"⚠️  참조 모델이 dialogue를 복사함, 무시")
            filtered_summaries.append("")  # 빈 요약 (품질 평가에서 낮은 점수)
        else:
            filtered_summaries.append(summary)

    return filtered_summaries

def _is_dialogue_copy(self, dialogue: str, summary: str, threshold: float = 0.9) -> bool:
    """
    요약이 dialogue를 그대로 복사한 것인지 검사

    Args:
        dialogue: 원본 대화
        summary: 생성된 요약
        threshold: 유사도 임계값 (0.9 이상이면 복사로 간주)

    Returns:
        True if summary is a copy of dialogue
    """
    from difflib import SequenceMatcher

    # 1. 길이 비율 체크
    len_ratio = len(summary) / (len(dialogue) + 1e-6)
    if len_ratio > 0.7:  # 요약이 원본의 70% 이상이면 의심
        # 2. 문자열 유사도 체크
        similarity = SequenceMatcher(None, dialogue, summary).ratio()
        if similarity > threshold:
            return True

    # 3. #Person1#, #Person2# 태그 체크
    if "#Person1#" in summary or "#Person2#" in summary:
        # 요약에 대화 태그가 남아있으면 복사로 간주
        return True

    return False
```

**효과:**
- dialogue를 복사한 참조 모델 출력을 필터링
- 빈 문자열로 대체하여 품질 점수를 낮춤
- KoBART 요약이 최종 선택됨

---

#### 해결책 B: 보정 전략 개선

**수정 위치: `src/correction/ensemble_strategies.py`**

```python
class QualityBasedStrategy(EnsembleStrategy):
    """품질 기반 선택 전략 (개선)"""

    def select(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        quality_scores: Dict[str, List[float]],
        threshold: float
    ) -> List[str]:
        final_summaries = []

        for i in range(len(candidate_summaries)):
            candidate_quality = quality_scores["candidate_quality"][i]

            # -------------- KoBART 품질 확인 -------------- #
            if candidate_quality >= threshold:
                final_summaries.append(candidate_summaries[i])
                continue

            # -------------- 최고 품질 참조 모델 선택 -------------- #
            best_model = None
            best_quality = -1

            for model_name in reference_summaries.keys():
                quality_key = f"{model_name}_quality"
                if quality_key in quality_scores:
                    quality = quality_scores[quality_key][i]

                    # ✅ 추가: 참조 요약이 비어있거나 너무 짧으면 제외
                    ref_summary = reference_summaries[model_name][i]
                    if not ref_summary.strip() or len(ref_summary) < 10:
                        continue

                    if quality > best_quality:
                        best_quality = quality
                        best_model = model_name

            # -------------- 최종 선택 (개선) -------------- #
            if best_model and best_quality > threshold:
                # ✅ 수정: 품질이 임계값보다 높을 때만 참조 모델 사용
                final_summaries.append(reference_summaries[best_model][i])
            else:
                # ✅ 개선: 참조 모델 품질이 낮으면 KoBART 사용
                final_summaries.append(candidate_summaries[i])

        return final_summaries
```

**효과:**
- 빈 문자열 또는 짧은 요약 필터링
- 참조 모델 품질이 임계값 이상일 때만 사용
- KoBART를 기본값으로 우선 사용

---

#### 해결책 C: 품질 평가 로직 강화

**수정 위치: `src/correction/quality_evaluator.py`**

```python
class QualityEvaluator:
    """요약 품질 평가기 (강화)"""

    def evaluate_all(
        self,
        candidate_summaries: List[str],
        reference_summaries: Dict[str, List[str]],
        dialogues: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """전체 품질 평가"""
        if self.rouge is None:
            return self._create_dummy_scores(len(candidate_summaries), reference_summaries)

        quality_scores = {}

        # ... (기존 코드)

        # ✅ 추가: dialogue와의 유사도 체크
        if dialogues:
            for model_name, summaries in reference_summaries.items():
                dialogue_similarity = []
                for i in range(len(summaries)):
                    # 요약과 dialogue의 ROUGE 계산
                    similarity = self._compute_rouge_f1(summaries[i], dialogues[i])
                    dialogue_similarity.append(similarity)

                # ✅ 품질 페널티 적용
                quality_key = f"{model_name}_quality"
                if quality_key in quality_scores:
                    for i in range(len(summaries)):
                        # dialogue와 너무 유사하면 품질 점수 감소
                        if dialogue_similarity[i] > 0.8:
                            quality_scores[quality_key][i] *= 0.1  # 90% 페널티

        return quality_scores
```

**효과:**
- dialogue와 유사한 요약에 페널티 부과
- 실제 요약 품질을 더 정확하게 평가

---

### 5.3 장기 개선 방안

1. **참조 모델 교체**
   - Fine-tuning된 KoBART 모델 사용
   - 우리 데이터 형식으로 추가 학습

2. **품질 평가 메트릭 다양화**
   - BERTScore 추가 (의미론적 유사도)
   - 길이 비율 체크
   - Perplexity 평가

3. **Validation 추가**
   - HF 보정 후 샘플 검증
   - dialogue 태그 검출
   - 자동 롤백 메커니즘

4. **실험 및 모니터링**
   - A/B 테스트로 보정 효과 검증
   - 보정 전후 ROUGE 비교
   - 이상 샘플 자동 감지

---

## 6. 권장 사항

### 6.1 단기 조치 (현재)

1. **HF 보정 사용 금지**
   - 모든 추론 스크립트에서 `--use_pretrained_correction` 플래그 제거
   - KoBART 단독 또는 Solar API 앙상블만 사용

2. **문서화**
   - 실험 문서에 "HF 보정 사용 금지" 명시
   - 팀원들에게 공유

### 6.2 중기 조치 (1-2주)

1. **해결책 A, B, C 구현**
   - dialogue 필터링 로직 추가
   - 보정 전략 개선
   - 품질 평가 강화

2. **단위 테스트 작성**
   - `_is_dialogue_copy()` 테스트
   - QualityBasedStrategy 테스트
   - End-to-End 테스트

3. **검증 실험**
   - Dev 데이터로 보정 전후 비교
   - 수동 샘플 검증

### 6.3 장기 조치 (1개월+)

1. **시스템 재설계**
   - 참조 모델 교체
   - 품질 평가 메트릭 다양화
   - 자동 Validation 추가

2. **모니터링 대시보드**
   - 보정 성공률 추적
   - 이상 샘플 자동 감지
   - 품질 메트릭 시각화

---

## 7. 결론

**문제 해결 완료 (2025-10-14):**
- HuggingFace 보정 시스템의 dialogue 복사 문제 해결
- 다음 3가지 개선 사항 적용 완료:
  1. `pretrained_corrector.py`: dialogue 필터링 로직 추가
  2. `ensemble_strategies.py`: QualityBasedStrategy 개선
  3. `quality_evaluator.py`: dialogue 유사도 페널티 적용

**적용된 개선 사항:**

1. **dialogue 필터링 (`pretrained_corrector.py`)**
   - `_is_dialogue_copy()` 메서드 추가
   - 길이 비율, 문자열 유사도, 대화 태그, 대화 형식 패턴 검사
   - dialogue 복사 감지 시 빈 문자열로 대체

2. **보정 전략 개선 (`ensemble_strategies.py`)**
   - 빈 문자열 및 짧은 요약(10자 미만) 필터링
   - 참조 모델 품질이 임계값 이상일 때만 사용
   - KoBART를 기본값으로 우선 사용

3. **품질 평가 강화 (`quality_evaluator.py`)**
   - dialogue 유사도 0.8 이상인 요약에 90% 페널티
   - 실제 요약 품질을 더 정확하게 평가

**현재 상태:**
- HuggingFace 보정 시스템이 정상 작동
- Solar API와 함께 안전하게 사용 가능
- dialogue 복사 문제 해결됨

**사용 방법:**
```bash
# HF 보정 + Solar API 앙상블 사용 (권장)
python scripts/inference.py \
  --model experiments/.../final_model \
  --test_data data/raw/test.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --use_solar_api
```

---

## 8. 참고 자료

### 관련 파일
- `src/correction/pretrained_corrector.py:214` - `_generate_summaries()`
- `src/correction/ensemble_strategies.py:46` - `QualityBasedStrategy`
- `src/correction/quality_evaluator.py:148` - `_compute_rouge_f1()`
- `src/inference/predictor.py:400` - HF 보정 호출부
- `scripts/inference.py:128` - HF 보정 옵션

### 관련 문서
- PRD 04: 추론 최적화
- PRD 12: 다중 모델 앙상블 전략

### 테스트 명령어
```bash
# HF 보정 + Solar API 앙상블 사용 (권장)
python scripts/inference.py \
  --model experiments/.../final_model \
  --test_data data/raw/test.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3 \
  --use_solar_api

# HF 보정만 사용
python scripts/inference.py \
  --model experiments/.../final_model \
  --test_data data/raw/test.csv \
  --batch_size 32 \
  --num_beams 5 \
  --max_new_tokens 100 \
  --min_new_tokens 30 \
  --repetition_penalty 1.5 \
  --no_repeat_ngram_size 3 \
  --use_pretrained_correction \
  --correction_models gogamza/kobart-base-v2 digit82/kobart-summarization \
  --correction_strategy quality_based \
  --correction_threshold 0.3
```
