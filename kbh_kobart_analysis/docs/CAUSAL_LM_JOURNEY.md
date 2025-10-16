# Causal LM 탐색 여정: Llama/Qwen을 활용한 한국어 요약 연구

> **한 문장 요약**: Causal LM을 요약 태스크에 적용하며 템플릿·collator·loss masking 불일치로 수차례 성능 붕괴를 겪었지만, 이를 디버깅하며 **"LLM은 학습 대상이 아니라 도구(Teacher, Rewriter, Reranker)로 쓰는 것이 가장 효율적"**이라는 결론에 도달했다.

## 1. 배경과 동기

처음에는 KoBART/KoT5 같은 Seq2Seq 모델로 안정적인 요약 파이프라인을 운영했지만, 점차 대화형·자유요약 스타일을 구현하려면 Llama/Qwen 계열의 Causal LM 구조가 더 유연하다고 판단했다.

특히 `apply_chat_template()` 같은 시스템·유저·어시스턴트 구조로 **지시문형 데이터(SFT)**를 학습할 수 있다는 점이 매력으로 작용했다.

**기대했던 것:**
- Chat template 기반의 유연한 프롬프팅
- 멀티턴 대화 요약에 특화된 생성 능력
- SFT를 통한 태스크 적응력

## 2. 핵심 도전과제

### 2.1 Chat Template & Format Mismatch

**문제**: 학습 시에는 `apply_chat_template()`로 system/user/assistant 프롬프트를 넣었지만, 추론 때는 raw dialogue만 모델에 넣는 실수를 발견.

**결과**: 모델은 학습 때 본 형식과 완전히 다른 입력을 받았고, **형식 불일치(format mismatch)**로 성능이 붕괴.
- ROUGE 급락
- 요약문에 외국어/깨진 토큰/비논리 문장 등장

### 2.2 Data Collator & Loss Masking

**문제**: 초기에 `DataCollatorForLanguageModeling(mlm=False)`를 그대로 사용.

**이슈**: 프롬프트 영역까지 loss 계산이 들어감 → 모델이 프롬프트를 복사하거나 장황하게 반복.

- `mlm=True`로도 테스트했지만 Causal LM과 부적합(MLM은 BERT류).
- 결국 "왜 loss가 프롬프트까지 계산되는지"를 디버깅하며 **Completion-only Loss**(응답 구간만 학습) 개념을 명확히 이해.

### 2.3 다국어 토큰 혼입 문제

**증상**: 한국어 요약에 영어·일본어·베트남어·중국어 한자·스페인어·아랍어·태국어, 심지어 깨진 유니코드까지 섞임.

**원인**:
1. **모델 특성**: Llama·Qwen은 멀티링궐로 사전학습되어 타언어 토큰의 사전확률이 항상 "켜져" 있음
2. **프롬프트 한계**: "영문/일문 금지" 같은 부정형 지시는 오히려 해당 개념에 주의를 끌어 역트리거가 발생
3. **디코딩 제약 부재**: 토큰 레벨 제약 없이는 보장 불가

**시도한 해결책과 한계**:

| 방법 | 효과 | 한계 |
|------|------|------|
| 강화 프롬프트 ("금지" 명시) | 효과 없음 | 부정형 지시의 역효과 |
| `bad_words_ids` (라틴/가나) | 라틴·가나 차단 성공 | CJK 한자 누락 |
| `bad_words_ids` + CJK URO (4E00–9FFF) | 혼입 95%+ 차단 | 확장 블록(Ext-A/B) 구멍 존재 |

**최종 해결책**:

**A. 화이트리스트 기반 제약 (권장)**
```python
# prefix_allowed_tokens_fn 사용
# 한글(AC00-D7AF) + 숫자 + 기본 구두점만 허용
# 구조적으로 타언어 불가능
```

**B. 정적 로짓 마스크 (LogitsProcessor)**
```python
# 시작 시 disallowed 인덱스에 -inf를 담은 텐서 생성
# 매 스텝 scores += mask만 수행
# 디코드 호출 없이 O(V) 브로드캐스트로 성능 저하 최소
```

**C. 후처리**
- `ftfy`로 유니코드 정규화 (모지바케 교정)
- 정규식으로 잔류 라틴/키릴/태국/아랍 문자 제거
- fastText LID로 ko 확률 임계치 미달 시 재생성

### 2.4 평가 지표 불일치

**문제**: Seq2Seq 계열은 `rouge_sum` 기반으로 체크포인트를 고르는데, Causal LM은 `eval_loss` 기준이라 실제 생성 품질과 괴리.

**결과**: loss는 낮지만 요약 품질이 떨어지는 경우 다수 발생.

**해결**:
- `predict_with_generate=True`
- `metric_for_best_model="rouge_sum"`으로 통일

### 2.5 모델 크기 vs 언어 적합성

**예상**: 파라미터가 클수록 좋을 거라 기대.

**실제**: Llama-3.2-Korean-3B가 Qwen3-4B, Qwen2.5-7B보다 오히려 ROUGE가 높게 나옴.

**교훈**: 원인은 "모델 크기"가 아니라 **언어 적합성·템플릿 일치·학습 설정 정합성**.

### 2.6 리소스 제약

**환경**: RTX 3090 (24GB)

**문제**:
- 8B 모델 LoRA는 자주 OOM
- QLoRA만이 현실적인 옵션 (batch 1 수준에서 GPU Util 87%)

**혼란**: "점유율 높다고 좋은 건가?" → GPU Util이 아닌 **tokens/sec, step time**으로 효율 판단해야 함.

### 2.7 데이터 분포 시프트 (OOD)

**발견**: Train은 요일 1–7이 고르게 섞였지만, Test는 토요일(7)만 존재.

**영향**:
- Test seq의 미등장 비율이 매우 높아 분포가 실제로 달랐음
- CV→LB 일반화 손실이 컸음
- 동일 디코딩 설정을 쓰면 과·과소요약/문체 불일치 발생

### 2.8 환경 불안정성

**증상**:
- `Seq2SeqTrainer` 생성 시 `Accelerate.Accelerator(dispatch_batches)` 인자 충돌
- Broken pipe, 커널 버전 경고(5.4.x), Flash-Attn 빌드 지연
- 토크나이저/모델의 PAD/BOS/EOS 자동 정렬 경고
- 실험이 자주 끊기고 속도가 크게 저하

**원인**: transformers–accelerate 버전 불일치가 직접적인 예외를 만들고, Dataloader 설정/OS 커널/빌드 의존성까지 얽혀 불안정성 증폭.

## 3. 문제 해결 시도와 학습

| 문제 | 시도한 해결책 | 결과 | 배운 점 |
|------|-------------|------|---------|
| **Chat template 불일치** | `apply_chat_template(..., add_generation_prompt=True)` 학습/추론 통일 | 외국어/깨진 토큰 사라짐 | 템플릿 정합성이 필수 |
| **Loss 영역 오적용** | `DataCollatorForCompletionOnlyLM` 사용 (응답만 라벨 유지) | 프롬프트 복사/반복 현상 제거 | Completion-only loss |
| **다국어 혼입** | 화이트리스트(prefix_allowed_tokens_fn) + 로짓 마스크 | 혼입율 95%+ 감소 | 금지보다 허용이 효과적 |
| **평가 기준 불일치** | `predict_with_generate=True`, `metric_for_best_model="rouge_sum"` | loss-품질 괴리 해소 | 생성 기반 지표 필수 |
| **리소스 초과** | 8B→QLoRA, 3B/4B→LoRA | VRAM 안정화, OOM 제거 | 모델별 전략 분리 |
| **OOD 문제** | 토요일 전용 디코딩 프리셋 (길이/penalty/temperature) | 일반화 손실 감소 | 프로파일별 튜닝 유효 |
| **환경 충돌** | transformers-accelerate 버전 정합, Dataloader 최적화 | 학습 안정화 | 환경이 실험의 토대 |
| **효율성 혼동** | GPU Util 대신 tokens/sec / step time 로깅 | 학습 효율 판단 기준 명확화 | 실제 처리량으로 판단 |

## 4. 기술적 깊이

### 4.1 다국어 문제 해결: 디코딩 제약 상세

**화이트리스트 기반 제약 (권장 방법)**

전략: "금지 목록"이 아니라 "허용 토큰만" 열기.

구현: HF `generate()`의 `prefix_allowed_tokens_fn`에 한글(AC00–D7AF)·공백·숫자·기본 문장부호로만 이뤄진 허용 토큰 ID 집합을 반환 → 구조적으로 타언어 불가.

**정적 로짓 마스크 (LogitsProcessor)**

전략: 시작 시 disallowed 인덱스에 -inf를 담은 텐서를 만들고, 매 스텝 `scores += mask`만 수행.

성능: 디코드 호출 없이 O(V) 브로드캐스트 1회로 끝나 성능 저하 최소화.

**bad_words_ids의 한계**

블랙리스트를 계속 쓴다면 CJK Ext-A(3400–4DBF)·Compatibility(F900–FAFF)·Supplement/Ext-B~G(20000–) 등 확장 전반을 포함해야 함. 그래도 토큰 분할 우회 가능성은 남아 **화이트리스트가 더 근본적**.

### 4.2 OOD 대응 전략

**토요일 전용 디코딩 프리셋**
- 길이/penalty/temperature를 토요일 프로파일에 맞춰 조정
- 토픽별 클러스터 기반으로 차등 적용
- 프롬프트 고정으로 일관성 확보

**효과**: 동일 설정으로 모든 요일을 커버하는 것보다 일반화 손실 크게 감소.

### 4.3 학습 안정화

**Gradient Norm 튜닝**

문제: `grad_norm`이 1.0~1.4에 자주 분포 → `max_grad_norm=1.0`이면 상시 클리핑.

해결:
- grad_norm 분포(1.0~1.2, 드물게 1.3)에 맞춰 `max_grad_norm=1.2`로 소폭 상향
- lr↓, warmup↑, grad_acc↑로 grad 분포 스파이크 완화
- 클리핑은 폭주 방지 장치로, 과/소클리핑 모두 피함

**환경 정렬**

조치:
- transformers–accelerate를 호환쌍으로 맞춘 뒤 커널 완전 재시작
- Dataloader에 `num_workers` 축소, `persistent_workers=False`로 안정화
- 의존성 충돌 해소가 실험 속도·재현성의 토대

### 4.4 평가 체계 개선

**복합 지표 도입**:
- **혼입율@N, 한글비율**: 언어 품질 지표로 ROUGE와 함께 추적
- **의미 유사도**: KoBERTScore/KoSimCSE 기반으로 형태론적 잡음에 덜 민감하게 모니터링

**평가 전략**:
- 훈련 중엔 eval loss로 빠르게 모니터링
- 종료 후 소규모 dev에 단 한 번 ROUGE를 돌려 최종 체크포인트 확정 (속도와 품질 타협)

## 5. Seq2Seq vs Causal LM 비교

| 측면 | Seq2Seq (KoBART/KoT5) | Causal LM (Llama/Qwen) |
|------|----------------------|------------------------|
| **요약 적합성** | 높음 (태스크 특화 구조) | 중간 (범용 생성 모델) |
| **언어 제어** | 기본적으로 안정 | 화이트리스트·후처리 필수 |
| **학습 리소스** | Full fine-tune 가능 | QLoRA/LoRA 필수 (8B는 QLoRA만) |
| **템플릿 복잡도** | 단순 (input→output) | Chat template 정합 필수 |
| **Collator** | DataCollatorForSeq2Seq | DataCollatorForCompletionOnlyLM |
| **평가** | ROUGE 기본 지원 | eval_loss 기본, ROUGE 별도 설정 필요 |
| **유연성** | 제한적 | 프롬프팅·멀티턴 대화 가능 |

**결론**: 요약 태스크는 **Seq2Seq(KoBART/KoT5)가 여전히 유리**. Causal LM은 언어 제약 + 템플릿 고정 + 후처리 + Completion-only loss를 모두 동반해야 일관 품질 확보.

## 6. 최종 결론과 하이브리드 전략

### 배운 핵심 교훈

1. **Causal LM은 템플릿-collator-평가 3축 정합이 없으면 절대 성능이 안 나온다.**
2. **Eval_loss만 보고 판단하면 안 된다. 생성기반 지표(ROUGE/BLEU)가 필요하다.**
3. **모델 크기 ≠ 성능. 세팅 정합성과 언어 적합도가 더 중요하다.**
4. **리소스 효율 판단 기준을 GPU Util이 아니라 실제 처리량(tokens/sec)으로 전환해야 한다.**
5. **디코딩은 "금지"보다 "허용"이 세다: 혼입 이슈는 화이트리스트 기반 제약이 구조적으로 해소한다.**
6. **작은 SFT가 큰 차이를 만든다: 한국어 전용 SFT는 디코딩 제약과 결합될 때 ROI가 높다.**
7. **OOD를 무시하지 말 것: test 프로파일에 맞춘 디코딩 프리셋만으로도 일반화 손실을 크게 줄일 수 있다.**
8. **환경은 실험의 토대: 라이브러리 버전 정합–재시작–Dataloader 안정화 없이는 모델 성능 논의가 무의미해진다.**

### 전환한 전략

**학습**: KoBART(Seq2Seq) 중심으로 안정화.

**Causal LM(Llama, Qwen) 활용**: 데이터 증강·후편집·재랭킹 등 **LLM-as-a-Tool**로 활용.

**하이브리드 파이프라인 설계**:
- KoBART 학습 + LLM 재랭킹/후편집
- 입력 증강 (Back-translation/패러프레이즈) LLM 기반
- 불확실 샘플만 LLM inference

**Collator/템플릿 구조 표준화**:
- Encoder류 → `DataCollatorForSeq2Seq`
- Causal류 → `DataCollatorForCompletionOnlyLM`

**로깅 및 실험체계 고도화**: tokens/sec, step time, ROUGE trend, GPU mem tracking 자동화.

## 7. 재사용 가능한 체크리스트

### Causal LM 사용 시 필수 체크

**템플릿 & Collator:**
- [ ] Chat template 정합: 학습/추론 모두 `apply_chat_template(..., add_generation_prompt=True)`
- [ ] Completion-only loss: `DataCollatorForCompletionOnlyLM` 사용
- [ ] 프롬프트 영역 제외한 응답만 라벨 유지 확인

**평가 & 선택:**
- [ ] 생성 기반 평가: `predict_with_generate=True`, `metric_for_best_model="rouge_sum"`
- [ ] eval_loss만으로 체크포인트 선택하지 않기
- [ ] 소규모 dev에서 ROUGE 최종 검증

**다국어 토큰 제어:**
- [ ] 화이트리스트 적용: `prefix_allowed_tokens_fn` (한글+숫자+기본 구두점)
- [ ] 또는 정적 로짓 마스크 (LogitsProcessor)
- [ ] 후처리: `ftfy` 유니코드 정규화, 정규식 필터
- [ ] fastText LID로 ko 확률 임계치 미달 재생성

**학습 안정화:**
- [ ] Gradient norm 분포 분석 후 `max_grad_norm` 조정 (1.2 권장)
- [ ] lr↓, warmup↑, grad_acc↑로 grad 스파이크 완화
- [ ] transformers-accelerate 버전 정합 확인
- [ ] Dataloader: `num_workers` 축소, `persistent_workers=False`

**모델 & 리소스:**
- [ ] 언어 적합성 우선 (크기보다)
- [ ] 8B 모델 → QLoRA, 3B/4B → LoRA
- [ ] 효율은 GPU Util이 아닌 tokens/sec로 판단

**디코딩 설정:**
- [ ] `do_sample=False` (탐욕/빔)
- [ ] `num_beams=4`
- [ ] `no_repeat_ngram_size=3~4`
- [ ] 한국어 화이트리스트 적용

**평가 지표:**
- [ ] ROUGE (표준 지표)
- [ ] 혼입율@N, 한글비율 (언어 품질)
- [ ] KoBERTScore/KoSimCSE (의미 유사도)
- [ ] OOD 프로파일별 성능 추적

### 다음 프로젝트 액션 아이템

- [ ] LogitsProcessor(화이트리스트) + 정규식 후처리 파이프라인 공통 모듈화
- [ ] 한국어 전용 소규모 SFT 데이터셋 정제 → 주기적 증분 학습
- [ ] 토요일 전용 디코딩 프리셋 고정, 토픽별 프리셋 병행
- [ ] 혼입율@N/한글비율/KoBERTScore를 CI 단계에서 자동 리포팅
- [ ] transformers–accelerate 버전 고정 파일과 재현용 스크립트 정리
- [ ] 동일 파이프라인을 Seq2Seq(KoBART/KoT5)에도 적용해 레퍼런스 점수 확보

---

**참고 자료:**
- [Hugging Face `generate()` API 문서](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [TRL `DataCollatorForCompletionOnlyLM` 문서](https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completions-only)
- [Llama 3.x 공식 문서](https://ai.meta.com/llama/)
- [Qwen 공식 문서](https://qwen.readthedocs.io/)
- [PyTorch `clip_grad_norm_` 문서](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)