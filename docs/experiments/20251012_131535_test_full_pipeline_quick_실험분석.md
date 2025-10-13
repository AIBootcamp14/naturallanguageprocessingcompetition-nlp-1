# 실험 분석 보고서: 20251012_131535_test_full_pipeline_quick

## 1. 실험 개요

### 1.1 실험 정보
- **실험 ID**: `20251012_131535_test_full_pipeline_quick`
- **실험 일시**: 2025년 10월 12일 13:15:35
- **실험 모드**: Full Pipeline (학습 + 앙상블 + Solar API + TTA + 추론)
- **실험 목적**: 6개 모델 전체 파이프라인 검증 및 BFloat16 호환성 테스트

### 1.2 모델 구성
| 모델명 | 타입 | 크기 | 상태 |
|--------|------|------|------|
| kobart | Seq2Seq | 123M | ✅ 성공 |
| llama-3.2-korean-3b | Causal LM | 3B | ❌ 실패 |
| qwen3-4b | Causal LM | 4B | ❌ 실패 |
| solar-10.7b | Causal LM | 10.7B | ❌ 실패 |
| polyglot-ko-12.8b | Causal LM | 12.8B | ❌ 실패 |
| kullm-v2 | Causal LM | 12.8B | ❌ 실패 |

### 1.3 실험 타임라인
```mermaid
gantt
    title 실험 진행 타임라인
    dateFormat  HH:mm
    axisFormat %H:%M

    section 모델 학습
    kobart (성공)           :done, kobart, 13:15, 13:19
    llama-3.2-korean-3b (실패) :crit, llama, 13:19, 13:20
    qwen3-4b (실패)         :crit, qwen, 13:20, 13:27
    solar-10.7b (실패)      :crit, solar, 13:27, 13:36
    polyglot-ko-12.8b (실패) :crit, polyglot, 13:36, 13:49
    kullm-v2 (실패)         :crit, kullm, 13:49, 13:50

    section 파이프라인
    Solar API 평가          :done, solar_api, 13:49, 13:50
    최적화 및 추론          :done, inference, 13:50, 13:50
```

## 2. 실험 결과 분석

### 2.1 전체 성공률
- **성공**: 1/6 모델 (16.7%)
- **실패**: 5/6 모델 (83.3%)
- **주요 실패 원인**: BFloat16 AMP 호환성 문제 (80%), Device Map 설정 문제 (20%)

### 2.2 파이프라인 흐름도
```mermaid
flowchart TD
    Start([실험 시작<br/>20251012_131535]) --> ModelLoad[모델 로딩 단계]

    ModelLoad --> KB[kobart<br/>Seq2Seq 123M]
    ModelLoad --> LL[llama-3.2-korean-3b<br/>Causal LM 3B]
    ModelLoad --> QW[qwen3-4b<br/>Causal LM 4B]
    ModelLoad --> SL[solar-10.7b<br/>Causal LM 10.7B]
    ModelLoad --> PG[polyglot-ko-12.8b<br/>Causal LM 12.8B]
    ModelLoad --> KU[kullm-v2<br/>Causal LM 12.8B]

    KB --> KBTrain[학습 진행]
    KBTrain --> KBSuccess[✅ 학습 성공<br/>ROUGE-1: 0.413]

    LL --> LLError[❌ BFloat16 에러<br/>AMP GradScaler 호환 불가]
    QW --> QWError[❌ BFloat16 에러<br/>AMP GradScaler 호환 불가]
    SL --> SLError[❌ BFloat16 에러<br/>AMP GradScaler 호환 불가]
    PG --> PGError[❌ BFloat16 에러<br/>AMP GradScaler 호환 불가]
    KU --> KUError[❌ Device Map 에러<br/>offload_folder 누락]

    KBSuccess --> Ensemble[앙상블<br/>단일 모델로 진행]
    Ensemble --> Solar[Solar API 평가<br/>50 샘플]
    Solar --> TTA[TTA 적용 시도<br/>실패: 단일 모델]
    TTA --> Inference[추론 실행<br/>실패: token_type_ids 에러]

    Inference --> End([실험 종료<br/>부분 성공])

    classDef successNode fill:#a5d6a7,stroke:#1b5e20,color:#000
    classDef errorNode fill:#ffccbc,stroke:#bf360c,color:#000
    classDef warningNode fill:#fff9c4,stroke:#f57f17,color:#000
    classDef processNode fill:#e1f5ff,stroke:#01579b,color:#000

    class KBSuccess,Ensemble,Solar successNode
    class LLError,QWError,SLError,PGError,KUError,TTA,Inference errorNode
    class KB,LL,QW,SL,PG,KU processNode
    class Start,End warningNode
```

## 3. 에러 상세 분석

### 3.1 에러 #1: BFloat16 AMP 호환성 문제 (4개 모델)

#### 영향 받은 모델
- llama-3.2-korean-3b
- qwen3-4b
- solar-10.7b
- polyglot-ko-12.8b

#### 에러 메시지
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

#### 에러 발생 위치
```python
File "torch/amp/grad_scaler.py", line 283, in _unscale_grads_
    torch._amp_foreach_non_finite_check_and_unscale_(...)
```

#### 근본 원인 분석

**1. 문제의 핵심**
- PyTorch의 AMP (Automatic Mixed Precision) GradScaler가 BFloat16 데이터 타입을 지원하지 않음
- `_amp_foreach_non_finite_check_and_unscale_cuda` 커널이 Float16만 지원하고 BFloat16은 미구현 상태

**2. 에러 발생 경로**
```mermaid
flowchart LR
    A[Training Loop] --> B[Gradient Clipping]
    B --> C[accelerator.clip_grad_norm_]
    C --> D[unscale_gradients]
    D --> E[scaler.unscale_]
    E --> F[_unscale_grads_]
    F --> G[_amp_foreach_non_finite_check_and_unscale_cuda]
    G --> H[❌ NotImplementedError<br/>BFloat16 미지원]

    classDef errorNode fill:#ffccbc,stroke:#bf360c,color:#000
    classDef processNode fill:#e1f5ff,stroke:#01579b,color:#000

    class H errorNode
    class A,B,C,D,E,F,G processNode
```

**3. 코드 레벨 원인**

`src/models/lora_loader.py:108-113`:
```python
# dtype 결정 (Llama: bf16, Qwen: fp16)
compute_dtype = torch.bfloat16
if 'qwen' in self.config.model.checkpoint.lower():
    compute_dtype = torch.float16
    self._log("  - Qwen 모델: fp16 사용")
else:
    self._log("  - Llama 모델: bf16 사용")
```

**문제점**:
- Qwen 모델은 조건문으로 Float16으로 설정되지만, **실제로는 여전히 BFloat16으로 실행됨**
- 이유: `llm_loader.py:48`에서 config에 quantization 설정이 없으면 기본값 'float16'을 사용하지만, `lora_loader.py`에서는 하드코딩된 `torch.bfloat16`을 먼저 적용
- Solar, Polyglot 모델도 Qwen이 아니므로 BFloat16으로 설정됨
- Llama 모델도 BFloat16으로 설정되어 동일한 에러 발생

**4. 왜 kobart는 성공했는가?**
- kobart는 Seq2Seq 모델로 `llm_loader.py`나 `lora_loader.py`를 사용하지 않음
- `bart_loader.py`를 통해 로딩되며, QLoRA를 사용하지 않음
- 따라서 BFloat16 관련 설정의 영향을 받지 않음

#### 해결 방법

**방법 1: src/models/lora_loader.py 수정 (권장)**

`src/models/lora_loader.py:108` 라인을 다음과 같이 수정:

```python
# 변경 전
compute_dtype = torch.bfloat16
if 'qwen' in self.config.model.checkpoint.lower():
    compute_dtype = torch.float16
    self._log("  - Qwen 모델: fp16 사용")
else:
    self._log("  - Llama 모델: bf16 사용")

# 변경 후
# PyTorch AMP GradScaler는 BFloat16을 지원하지 않으므로 모든 모델에 Float16 사용
compute_dtype = torch.float16
self._log("  - QLoRA compute dtype: fp16 (AMP 호환)")
```

**방법 2: src/models/llm_loader.py 수정 (대안)**

`src/models/llm_loader.py:46-49` 라인 수정:

```python
# 변경 전
bnb_4bit_compute_dtype=getattr(
    torch,
    config.model.quantization.get('bnb_4bit_compute_dtype', 'float16')
),

# 변경 후 (명시적으로 float16 강제)
bnb_4bit_compute_dtype=torch.float16,  # AMP 호환성을 위해 float16 고정
```

**방법 3: Training 옵션 수정 (임시 방편)**

`configs/base/causal_lm.yaml`에 다음 추가:
```yaml
training:
  bf16: false  # BFloat16 비활성화
  fp16: true   # Float16 사용
```

**권장 사항**: **방법 1**을 권장합니다.
- 가장 직접적이고 근본적인 해결책
- 모든 Causal LM 모델에 일관되게 적용
- AMP GradScaler와의 호환성 보장

### 3.2 에러 #2: Device Map Offload 설정 문제 (1개 모델)

#### 영향 받은 모델
- kullm-v2 (12.8B)

#### 에러 메시지
```
ValueError: The current `device_map` had weights offloaded to the disk.
Please provide an `offload_folder` for them.
Alternatively, make sure you have `safetensors` installed if the model you are using offers the weights in this format.
```

#### 에러 발생 위치
```python
File "transformers/modeling_utils.py", line 5387, in _load_pretrained_model
    raise ValueError(...)
```

#### 근본 원인 분석

**1. 문제의 핵심**
- kullm-v2는 12.8B 파라미터의 대형 모델
- GPU 메모리 부족으로 일부 가중치가 디스크로 오프로드됨
- 디스크 오프로드 시 `offload_folder` 경로가 필요하지만 지정되지 않음

**2. 에러 발생 경로**
```mermaid
flowchart LR
    A[Model Loading] --> B[AutoModelForCausalLM.from_pretrained]
    B --> C[device_map='auto' 적용]
    C --> D[GPU 메모리 체크]
    D --> E{메모리 충분?}
    E -->|Yes| F[GPU에 전체 로딩]
    E -->|No| G[일부 가중치 디스크 오프로드]
    G --> H{offload_folder 설정?}
    H -->|Yes| I[디스크에 임시 저장]
    H -->|No| J[❌ ValueError<br/>offload_folder 누락]

    classDef errorNode fill:#ffccbc,stroke:#bf360c,color:#000
    classDef successNode fill:#a5d6a7,stroke:#1b5e20,color:#000
    classDef processNode fill:#e1f5ff,stroke:#01579b,color:#000

    class J errorNode
    class F,I successNode
    class A,B,C,D,G processNode
```

**3. 코드 레벨 원인**

`src/models/llm_loader.py:58-64`:
```python
model = AutoModelForCausalLM.from_pretrained(
    config.model.checkpoint,
    quantization_config=quantization_config,
    device_map="auto",  # 자동 디바이스 할당
    torch_dtype=torch.bfloat16 if config.training.get('bf16', True) else torch.float16,
    trust_remote_code=True
)
# offload_folder 파라미터 누락!
```

**문제점**:
- `device_map="auto"`는 GPU 메모리가 부족할 때 자동으로 CPU/디스크로 오프로드
- 대형 모델(10B 이상)은 4-bit 양자화를 사용해도 GPU 메모리 부족 가능
- `offload_folder` 미지정 시 디스크 오프로드 불가

#### 해결 방법

**방법 1: offload_folder 추가 (권장)**

`src/models/llm_loader.py:58` 라인 수정:

```python
# 변경 전
model = AutoModelForCausalLM.from_pretrained(
    config.model.checkpoint,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if config.training.get('bf16', True) else torch.float16,
    trust_remote_code=True
)

# 변경 후
from pathlib import Path
offload_dir = Path(config.experiment.get('output_dir', 'outputs')) / 'offload'
offload_dir.mkdir(parents=True, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    config.model.checkpoint,
    quantization_config=quantization_config,
    device_map="auto",
    offload_folder=str(offload_dir),  # 디스크 오프로드 폴더 지정
    torch_dtype=torch.bfloat16 if config.training.get('bf16', True) else torch.float16,
    trust_remote_code=True
)
```

**방법 2: safetensors 설치 확인**

```bash
pip install safetensors
```

일부 모델은 safetensors 형식으로 가중치를 제공하며, 이 경우 더 효율적으로 로딩 가능.

**방법 3: max_memory 설정으로 명시적 제어**

```python
model = AutoModelForCausalLM.from_pretrained(
    config.model.checkpoint,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "20GiB", "cpu": "30GiB"},  # GPU 0: 20GB, CPU: 30GB
    offload_folder=str(offload_dir),
    torch_dtype=torch.float16,
    trust_remote_code=True
)
```

**권장 사항**: **방법 1 + 방법 2**를 조합하여 사용
- safetensors 먼저 설치
- offload_folder를 실험 폴더 내 `offload/` 서브디렉토리로 지정
- 대형 모델의 안정적인 로딩 보장

### 3.3 에러 #3: 추론 단계 에러

#### 에러 메시지
```
The following `model_kwargs` are not used by the model: ['token_type_ids']
```

#### 근본 원인
- 일부 모델(특히 Causal LM)은 `token_type_ids`를 지원하지 않음
- Tokenizer가 자동으로 `token_type_ids`를 생성하지만 모델이 이를 무시
- 경고성 메시지이지만, 추론 실행에는 영향을 미치지 않을 수 있음

#### 해결 방법

`src/inference/predictor.py` 또는 해당 추론 코드에서:

```python
# 변경 전
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
outputs = model.generate(**inputs, ...)

# 변경 후
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# token_type_ids 제거 (Causal LM은 사용하지 않음)
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']
outputs = model.generate(**inputs, ...)
```

## 4. 성공 모델 분석: kobart

### 4.1 학습 성능

| 메트릭 | 값 |
|--------|-----|
| **ROUGE-1** | 0.4134 |
| **ROUGE-2** | 0.2552 |
| **ROUGE-L** | 0.4065 |
| **ROUGE-Sum** | 1.0751 |
| **Loss** | 1.4561 |
| **학습 시간** | ~4분 |
| **평가 속도** | 6.37 samples/sec |

### 4.2 모델 정보
- **체크포인트**: `gogamza/kobart-base-v2`
- **아키텍처**: BART (Seq2Seq)
- **파라미터**: 123M
- **학습 설정**:
  - Epochs: 1
  - Batch Size: 8
  - Learning Rate: 5e-6
  - Gradient Accumulation: 4
  - Warmup Ratio: 0.1

### 4.3 성능 평가

**장점**:
- ✅ 안정적인 학습 완료
- ✅ BFloat16 문제 없음 (Seq2Seq 아키텍처)
- ✅ 양호한 ROUGE 점수 (ROUGE-1: 0.413)
- ✅ 빠른 추론 속도

**한계점**:
- ⚠️ 단일 모델로는 앙상블 효과 없음
- ⚠️ Causal LM 대비 파라미터 수 적음 (123M vs 3B~12.8B)
- ⚠️ 최신 LLM 대비 성능 제한적

## 5. Solar API 평가 결과

### 5.1 Solar API 성능
| 메트릭 | 값 |
|--------|-----|
| **Solar ROUGE-1 F1** | 0.2272 |
| **Solar ROUGE-2 F1** | 0.0765 |
| **Solar ROUGE-L F1** | 0.2177 |
| **평가 샘플 수** | 50 |

### 5.2 kobart vs Solar API 비교

```mermaid
graph LR
    subgraph "ROUGE-1 비교"
    K1[kobart: 0.413]
    S1[Solar API: 0.227]
    end

    subgraph "ROUGE-2 비교"
    K2[kobart: 0.255]
    S2[Solar API: 0.077]
    end

    subgraph "ROUGE-L 비교"
    K3[kobart: 0.407]
    S3[Solar API: 0.218]
    end

    K1 -.->|+81.9%| S1
    K2 -.->|+231.4%| S2
    K3 -.->|+86.9%| S3

    classDef successNode fill:#a5d6a7,stroke:#1b5e20,color:#000
    classDef warningNode fill:#fff9c4,stroke:#f57f17,color:#000

    class K1,K2,K3 successNode
    class S1,S2,S3 warningNode
```

**분석**:
- kobart가 Solar API 대비 **모든 메트릭에서 우수**
- ROUGE-2에서 가장 큰 차이 (231.4% 높음)
- Solar API는 Zero-shot 성능이므로, Fine-tuning된 kobart가 당연히 우수

## 6. TTA 및 최적화 결과

### 6.1 TTA (Test-Time Augmentation)
- **적용 여부**: ❌ 실패
- **이유**: 단일 모델만 성공하여 앙상블 불가
- **설정된 전략**: paraphrase, reorder, synonym, mask (num_aug=2)

### 6.2 최적화
- **최적화 폴더**: `experiments/20251012/20251012_131535_test_full_pipeline_quick/optimized/`
- **상태**: 생성됨 (하지만 추론 실패로 활용되지 않음)

## 7. 실험 학습 사항 및 권장 조치

### 7.1 즉시 적용 필요한 수정 사항

#### 우선순위 1: BFloat16 문제 해결 (Critical)
```python
# src/models/lora_loader.py:108-113
# 변경 전
compute_dtype = torch.bfloat16
if 'qwen' in self.config.model.checkpoint.lower():
    compute_dtype = torch.float16

# 변경 후
compute_dtype = torch.float16  # AMP 호환성을 위해 모든 모델에 Float16 사용
self._log("  - QLoRA compute dtype: fp16 (AMP 호환)")
```

**영향**: 4개 모델 (llama, qwen3, solar, polyglot) 학습 가능

#### 우선순위 2: offload_folder 설정 (High)
```python
# src/models/llm_loader.py:58
from pathlib import Path
offload_dir = Path(config.experiment.get('output_dir', 'outputs')) / 'offload'
offload_dir.mkdir(parents=True, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    ...,
    offload_folder=str(offload_dir),
    ...
)
```

**영향**: kullm-v2 모델 로딩 가능

#### 우선순위 3: token_type_ids 제거 (Medium)
```python
# 추론 코드
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']
```

**영향**: 추론 단계 정상 실행

### 7.2 문서 업데이트 필요 사항

#### PRD 문서
- **PRD 08 (LLM 파인튜닝 전략)**: BFloat16 → Float16 변경 사항 반영
- QLoRA 설정에서 `bnb_4bit_compute_dtype=torch.float16` 명시

#### 모듈화 문서
- **`docs/모듈화/02_핵심_시스템.md`**:
  - LLM Loader 섹션에 AMP 호환성 주의사항 추가
  - offload_folder 설정 가이드 추가

- **`docs/모듈화/05_트러블슈팅_가이드.md`** (존재 시):
  - BFloat16 에러 해결 방법 추가
  - Device Map 에러 해결 방법 추가

### 7.3 재실험 권장 사항

수정 사항 적용 후 다음 실험 권장:

```bash
python scripts/train.py \
  --mode full \
  --models kobart llama-3.2-korean-3b qwen3-4b solar-10.7b polyglot-ko-12.8b kullm-v2 \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 5e-6 \
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --label_smoothing 0.1 \
  --k_folds 2 \
  --fold_seed 42 \
  --ensemble_strategy stacking \
  --num_beams 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50 \
  --repetition_penalty 1.2 \
  --length_penalty 1.0 \
  --no_repeat_ngram_size 3 \
  --experiment_name test_full_pipeline_fixed \
  --seed 42
```

**예상 결과**:
- ✅ 6/6 모델 학습 성공
- ✅ 앙상블 정상 동작
- ✅ TTA 적용 가능
- ✅ 추론 및 제출 파일 생성 성공

## 8. 결론

### 8.1 실험 요약
이번 실험은 **부분적 성공**으로 평가됩니다:
- ✅ kobart 모델 학습 및 평가 성공
- ✅ Solar API 교차 검증 완료
- ❌ 5개 LLM 모델 학습 실패
- ❌ 앙상블 및 TTA 불가
- ❌ 추론 단계 에러

### 8.2 핵심 발견 사항

1. **BFloat16 호환성 문제**: PyTorch AMP GradScaler가 BFloat16을 지원하지 않음을 확인
2. **대형 모델 메모리 관리**: 12.8B 모델은 offload_folder 설정 필수
3. **토크나이저 호환성**: Causal LM은 token_type_ids 불필요

### 8.3 다음 단계
1. ✅ 본 분석 보고서 작성 완료
2. 🔄 `src/models/lora_loader.py` 수정 (BFloat16 → Float16)
3. 🔄 `src/models/llm_loader.py` 수정 (offload_folder 추가)
4. 🔄 추론 코드 수정 (token_type_ids 제거)
5. 🔄 문서 업데이트 (`docs/모듈화/02_핵심_시스템.md`)
6. 🔄 재실험 실행 (`test_full_pipeline_fixed`)


