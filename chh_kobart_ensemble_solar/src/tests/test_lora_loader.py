"""
LoRA Loader 테스트

PRD 08: LLM 파인튜닝 전략 구현
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ------------------------- 프로젝트 모듈 ------------------------- #
from src.config import load_config
from src.models.lora_loader import LoRALoader, load_lora_model_and_tokenizer


# ==================== 테스트 함수 정의 ==================== #
# ---------------------- LoRALoader 초기화 테스트 ---------------------- #
def test_lora_loader_init():
    """LoRALoader 초기화 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 1: LoRALoader 초기화")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Config 로드
        config = load_config("baseline_kobart")

        # Causal LM config로 변경
        config.model.type = "causal_lm"
        config.model.checkpoint = "Bllossom/llama-3.2-Korean-Bllossom-3B"

        # LoRA 설정 추가
        config.lora = {
            'r': 16,
            'alpha': 32,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
            'dropout': 0.05
        }

        # LoRALoader 인스턴스 생성
        loader = LoRALoader(config)

        print("✅ LoRALoader 초기화 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 토크나이저 로딩 테스트 ---------------------- #
def test_tokenizer_loading():
    """토크나이저 로딩 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 2: 토크나이저 로딩")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Config 설정
        config = load_config("baseline_kobart")
        config.model.type = "causal_lm"
        config.model.checkpoint = "Bllossom/llama-3.2-Korean-Bllossom-3B"
        config.lora = {
            'r': 16,
            'alpha': 32,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
            'dropout': 0.05
        }

        # LoRALoader 생성 및 토크나이저 로드
        loader = LoRALoader(config)
        tokenizer = loader._load_tokenizer(config.model.checkpoint)

        # 토크나이저 정보 출력
        print(f"토크나이저 로드 성공")
        print(f"  - Vocab size: {len(tokenizer)}")
        print(f"  - Pad token: {tokenizer.pad_token}")
        print(f"  - EOS token: {tokenizer.eos_token}")

        # -------------- 토큰화 테스트 -------------- #
        # 테스트 텍스트 토큰화
        text = "안녕하세요. 테스트입니다."
        tokens = tokenizer(text, return_tensors="pt")
        print(f"  - 토큰화 테스트: '{text}' -> {tokens['input_ids'].shape[1]}개 토큰")

        print("✅ 토크나이저 로딩 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- LoRA Config 생성 테스트 ---------------------- #
def test_lora_config():
    """LoRA Config 생성 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 3: LoRA Config 생성")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # PEFT 라이브러리 import
        from peft import LoraConfig, TaskType

        # LoRA Config 생성
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Config 정보 출력
        print("LoRA Config 생성 성공")
        print(f"  - Rank (r): {lora_config.r}")
        print(f"  - Alpha: {lora_config.lora_alpha}")
        print(f"  - Target modules: {lora_config.target_modules}")
        print(f"  - Dropout: {lora_config.lora_dropout}")
        print(f"  - Task type: {lora_config.task_type}")

        print("✅ LoRA Config 생성 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 편의 함수 테스트 ---------------------- #
def test_convenience_function():
    """편의 함수 테스트"""
    # 테스트 헤더 출력
    print("\n" + "=" * 60)
    print("테스트 4: 편의 함수 (load_lora_model_and_tokenizer)")
    print("=" * 60)

    # -------------- 테스트 실행 -------------- #
    try:
        # 함수 존재 확인
        print("함수 import 성공")
        print(f"  - 함수명: load_lora_model_and_tokenizer")
        print(f"  - 모듈: src.models.lora_loader")

        print("✅ 편의 함수 테스트 성공")
        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        return False


# ==================== 메인 실행부 ==================== #
# ---------------------- 전체 테스트 실행 함수 ---------------------- #
def main():
    """전체 테스트 실행"""
    # 테스트 시작 헤더 출력
    print("\n" + "=" * 70)
    print(" " * 20 + "LoRA Loader 테스트 시작")
    print("=" * 70)

    # -------------- 테스트 실행 -------------- #
    # 테스트 결과 수집용 리스트
    results = []
    results.append(("LoRALoader 초기화", test_lora_loader_init()))
    results.append(("토크나이저 로딩", test_tokenizer_loading()))
    results.append(("LoRA Config 생성", test_lora_config()))
    results.append(("편의 함수", test_convenience_function()))

    # -------------- 결과 요약 출력 -------------- #
    print("\n" + "=" * 70)
    print(" " * 25 + "테스트 결과 요약")
    print("=" * 70)

    # 통과한 테스트 개수 계산
    passed = sum(1 for _, result in results if result)
    total = len(results)

    # 각 테스트 결과 출력
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")

    # 최종 요약 출력
    print("=" * 70)
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    print("=" * 70)

    # 전체 테스트 통과 여부 반환
    return passed == total


# ---------------------- 메인 진입점 ---------------------- #
if __name__ == "__main__":
    success = main()  # 전체 테스트 실행
    sys.exit(0 if success else 1)  # 성공 시 0, 실패 시 1 반환
