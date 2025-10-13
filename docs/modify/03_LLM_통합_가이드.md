# ⚠️ 이 문서는 선택적 고급 기능입니다

**상태**: 선택적 (2025-10-11 업데이트)  
**우선순위**: 낮음 (현재 `train_llm.py`로 이미 가능)

현재 `scripts/train_llm.py`가 별도로 존재하여 LLM 파인튜닝이 완전히 가능합니다.
`train.py`와의 통합은 선택적 편의 기능입니다.

**현재 사용 가능**:
- ✅ LLM 파인튜닝 (`scripts/train_llm.py`)
- ✅ QLoRA 지원 (`src/models/lora_loader.py`)
- ✅ Llama, Qwen 모델 지원
- ✅ Chat Template 처리

**선택적 통합** (이 가이드):
- ⚠️ `train.py`에 LLM 로직 통합
- ⚠️ 단일 인터페이스로 모든 모델 학습

현재 시스템으로도 모든 LLM 기능 사용 가능하므로, 이 통합은 필수가 아닙니다.

자세한 내용은 이전 버전의 `03_LLM_통합_가이드.md`를 참고하세요.
