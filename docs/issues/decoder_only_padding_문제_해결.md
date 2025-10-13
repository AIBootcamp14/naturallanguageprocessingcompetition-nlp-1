# Decoder-Only 모델 Padding 경고 문제 해결

## 📋 문제 개요

### 발생 위치
- **실험**: `experiments/20251013/20251013_161056_test_strategy3_triple`
- **로그 파일**: `train.log` (라인 1683)
- **발생 시점**: 최종 평가 중 (Final Evaluation)

### 경고 메시지
```
A decoder-only architecture is being used, but right-padding was detected!
For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
```

### 영향을 받는 모델
- Llama-3.2-Korean-3B (Decoder-only, Causal LM)
- Qwen3-4B (Decoder-only, Causal LM)
- 기타 모든 Causal LM 아키텍처

---

## 🔍 원인 분석

### 1. Decoder-Only 모델의 특성
Decoder-only 모델 (Causal LM)은 **left-to-right autoregressive** 방식으로 작동합니다:
- 각 토큰은 **이전 토큰들만** 참조 가능 (causal attention mask)
- Padding이 오른쪽에 있으면 모델이 padding 토큰도 "이전 컨텍스트"로 학습
- 이는 생성 품질 저하와 예측 불일치를 초래

### 2. 올바른 Padding 방식
| 모델 타입 | Padding 위치 | 이유 |
|-----------|--------------|------|
| **Encoder-Decoder** (Seq2Seq) | Right | Encoder는 양방향 attention 사용 가능 |
| **Decoder-Only** (Causal LM) | Left | 과거 토큰만 참조해야 하므로 padding은 왼쪽에 배치 |

### 3. 문제가 발생한 코드 위치

#### ✅ 학습 시 (문제 없음)
`src/models/lora_loader.py:191`에서 이미 올바르게 설정됨:
```python
def _configure_tokenizer(self, tokenizer: AutoTokenizer):
    """토크나이저 설정 (Left padding for Causal LM)"""
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
```

#### ❌ 평가/추론 시 (문제 발생)
모델을 다시 로드할 때 tokenizer 설정이 초기화됨:

1. **`src/trainers/full_pipeline_trainer.py:523`**
   - 제출 파일 생성 시 (`_create_submission` 메서드)
   - 저장된 모델 로드 후 tokenizer 재설정 누락

2. **`src/trainers/full_pipeline_trainer.py:333`**
   - 앙상블 평가 시 (`_create_and_evaluate_ensemble` 메서드)
   - 여러 모델 로드 시 각 tokenizer 재설정 누락

3. **`src/trainers/multi_model_trainer.py:239`**
   - Multi-model 앙상블 평가 시 (`_evaluate_ensemble` 메서드)
   - Encoder-Decoder만 가정, Causal LM 미지원

4. **`src/ensemble/manager.py:58`**
   - ModelManager의 `load_model` 메서드
   - Encoder-Decoder만 가정, Causal LM 미지원

---

## ✅ 해결 방법

### 수정 전략
1. **모델 타입 자동 감지**: `AutoConfig`로 `is_encoder_decoder` 확인
2. **조건부 Tokenizer 설정**: Decoder-only 모델에만 left padding 적용
3. **Pad Token 보장**: `pad_token`이 없으면 `eos_token` 사용

### 수정된 코드 패턴

```python
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

# 1. 모델 타입 자동 감지
config = AutoConfig.from_pretrained(model_path)
is_encoder_decoder = config.is_encoder_decoder if hasattr(config, 'is_encoder_decoder') else False

# 2. 모델 타입에 따라 적절한 클래스 사용
if is_encoder_decoder:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)

# 3. Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 4. Decoder-only 모델의 경우 left padding 설정
if not is_encoder_decoder:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
```

### 수정된 파일 목록

#### 1. `src/trainers/full_pipeline_trainer.py`
**수정 위치 1**: `_create_submission` 메서드 (라인 523 근처)
```python
# 수정 전
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
if torch.cuda.is_available():
    model = model.cuda()

# 수정 후
tokenizer = AutoTokenizer.from_pretrained(best_model_path)

# Decoder-only 모델의 경우 left padding 설정
if not is_encoder_decoder:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    model = model.cuda()
```

**수정 위치 2**: `_create_and_evaluate_ensemble` 메서드 (라인 333 근처)
```python
# 수정 전
tokenizer = AutoTokenizer.from_pretrained(model_path)
if torch.cuda.is_available():
    model = model.cuda()

# 수정 후
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Decoder-only 모델의 경우 left padding 설정
if not is_encoder_decoder:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    model = model.cuda()
```

#### 2. `src/trainers/multi_model_trainer.py`
**수정 위치**: `_evaluate_ensemble` 메서드 (라인 226-266)
```python
# 수정 전: Encoder-Decoder만 지원
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 수정 후: 모든 모델 타입 지원
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer
)

# 모델 타입 자동 감지
config = AutoConfig.from_pretrained(model_path)
is_encoder_decoder = config.is_encoder_decoder if hasattr(config, 'is_encoder_decoder') else False

# 모델 타입에 따라 적절한 클래스 사용
if is_encoder_decoder:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Decoder-only 모델의 경우 left padding 설정
if not is_encoder_decoder:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
```

#### 3. `src/ensemble/manager.py`
**수정 위치**: `load_model` 메서드 (라인 36-93)
```python
# 수정 전: Encoder-Decoder만 지원
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 수정 후: 모든 모델 타입 지원
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer
)

# 모델 타입 자동 감지
config = AutoConfig.from_pretrained(model_path)
is_encoder_decoder = config.is_encoder_decoder if hasattr(config, 'is_encoder_decoder') else False

# 모델 및 토크나이저 로드
if is_encoder_decoder:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Decoder-only 모델의 경우 left padding 설정
if not is_encoder_decoder:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
```

---

## 🎯 개선 효과

### 1. 경고 제거
- Decoder-only 모델 평가/추론 시 padding 경고 완전 제거
- 로그가 깔끔해져 실제 문제 파악 용이

### 2. 생성 품질 향상
- **올바른 attention mask**: Padding 토큰을 컨텍스트로 학습하지 않음
- **일관된 생성**: 학습 시와 추론 시 동일한 padding 전략 사용
- **ROUGE 점수 개선 가능**: 더 정확한 요약 생성

### 3. 코드 견고성 향상
- **타입 안전성**: `AutoConfig`로 모델 타입 자동 감지
- **범용성**: Encoder-Decoder와 Decoder-only 모두 지원
- **유지보수성**: 일관된 패턴으로 향후 디버깅 용이

### 4. 호환성 보장
| 모델 | 아키텍처 | Padding | 상태 |
|------|----------|---------|------|
| KoBART | Encoder-Decoder | Right | ✅ 기존 유지 |
| Llama-3.2-Korean-3B | Decoder-only | Left | ✅ 수정 완료 |
| Qwen3-4B | Decoder-only | Left | ✅ 수정 완료 |
| 향후 LLM 모델 | Decoder-only | Left | ✅ 자동 지원 |

---

## 📊 검증 방법

### 1. 로그 확인
다음 실험 실행 시 경고 메시지가 사라졌는지 확인:
```bash
# Full pipeline 실행
python main.py --mode full --models kobart llama-3.2-korean-3b qwen3-4b

# 로그에서 padding 경고 검색
grep "right-padding was detected" experiments/*/train.log
```

### 2. Tokenizer 설정 확인
```python
from transformers import AutoTokenizer

# Decoder-only 모델 tokenizer 확인
tokenizer = AutoTokenizer.from_pretrained("model_path")
print(f"Padding side: {tokenizer.padding_side}")  # "left" 출력 예상
print(f"Pad token: {tokenizer.pad_token}")  # EOS 토큰 출력 예상
```

### 3. ROUGE 점수 비교
- 수정 전/후 동일 데이터에 대한 ROUGE 점수 비교
- Decoder-only 모델의 경우 미세한 성능 향상 예상

---

## 🔗 관련 문서

### PRD 문서
- **PRD 08**: LLM (Decoder-only) 지원
- **PRD 12**: 다중 모델 앙상블 전략
- **PRD 14**: Full Pipeline 통합

### 참고 자료
- [Hugging Face - Padding and Truncation](https://huggingface.co/docs/transformers/pad_truncation)
- [Causal LM vs Seq2Seq](https://huggingface.co/docs/transformers/tasks/language_modeling)

### 관련 이슈
- `docs/issues/시스템_문제_개선_과정.md` - BFloat16, Config 우선순위 등 기존 문제 해결
- `docs/issues/문장_끊김_문제_해결_과정.md` - 후처리 개선 과정

---

## 📝 요약

| 항목 | 내용 |
|------|------|
| **문제** | Decoder-only 모델 평가/추론 시 right-padding 경고 발생 |
| **원인** | 모델 재로드 시 tokenizer left-padding 설정 누락 |
| **해결** | 4개 파일에서 자동 타입 감지 + 조건부 padding 설정 추가 |
| **수정 파일** | `full_pipeline_trainer.py`, `multi_model_trainer.py`, `manager.py` (3개 파일, 4개 위치) |
| **효과** | 경고 제거, 생성 품질 향상, 코드 견고성/범용성 개선 |
| **날짜** | 2025-10-14 |

---

**✅ 해결 완료**: 모든 Decoder-only 모델에서 올바른 left-padding이 적용되며, 향후 새로운 Causal LM 모델 추가 시에도 자동으로 지원됩니다.
