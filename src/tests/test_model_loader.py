# ==================== Model Loader 테스트 스크립트 ==================== #
"""
모델 로더 시스템 테스트

테스트 항목:
1. 토크나이저 로딩
2. 특수 토큰 추가
3. 모델 로딩
4. 디바이스 배치
5. 전체 로딩 파이프라인
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 서드파티 라이브러리 ---------------------- #
import torch

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.config import load_config
from src.models import ModelLoader, load_model_and_tokenizer


# ==================== 테스트 함수들 ==================== #
# ---------------------- 토크나이저 로딩 테스트 ---------------------- #
def test_load_tokenizer():
    """토크나이저 로딩 테스트"""
    print("\n" + "="*60)
    print("테스트 1: 토크나이저 로딩")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")                 # 베이스라인 Config

    # 모델 로더 생성
    loader = ModelLoader(config)                            # 로더 초기화

    # 토크나이저 로드
    tokenizer = loader.load_tokenizer()                     # 토크나이저 로드

    # 검증
    assert tokenizer is not None                            # 토크나이저 존재 확인
    assert len(tokenizer) > 0                               # 어휘 크기 확인

    print(f"\n  어휘 크기: {len(tokenizer):,}")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  bos_token: {tokenizer.bos_token}")

    print("\n✅ 토크나이저 로딩 테스트 성공!")


# ---------------------- 특수 토큰 추가 테스트 ---------------------- #
def test_special_tokens():
    """특수 토큰 추가 테스트"""
    print("\n" + "="*60)
    print("테스트 2: 특수 토큰 추가")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")                 # 베이스라인 Config

    # 모델 로더 생성
    loader = ModelLoader(config)                            # 로더 초기화

    # 토크나이저 로드
    tokenizer = loader.load_tokenizer()                     # 토크나이저 로드

    # 특수 토큰 확인
    if hasattr(config.model, 'special_tokens') and config.model.special_tokens:
        special_tokens = list(config.model.special_tokens)  # 특수 토큰 리스트
        print(f"\n  설정된 특수 토큰: {len(special_tokens)}개")

        # 샘플 토큰 출력
        for token in special_tokens[:5]:
            token_id = tokenizer.convert_tokens_to_ids(token)  # 토큰 ID 확인
            print(f"    {token}: {token_id}")

        # 특수 토큰이 어휘에 추가되었는지 확인
        assert all(token in tokenizer.get_vocab() for token in special_tokens)

    print("\n✅ 특수 토큰 추가 테스트 성공!")


# ---------------------- 모델 로딩 테스트 ---------------------- #
def test_load_model():
    """모델 로딩 테스트"""
    print("\n" + "="*60)
    print("테스트 3: 모델 로딩")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")                 # 베이스라인 Config

    # 모델 로더 생성
    loader = ModelLoader(config)                            # 로더 초기화

    # 토크나이저 먼저 로드
    tokenizer = loader.load_tokenizer()                     # 토크나이저 로드

    # 모델 로드
    model = loader.load_model(tokenizer)                    # 모델 로드

    # 검증
    assert model is not None                                # 모델 존재 확인
    assert next(model.parameters()).device.type in ['cuda', 'cpu']  # 디바이스 확인

    print("\n✅ 모델 로딩 테스트 성공!")


# ---------------------- 디바이스 배치 테스트 ---------------------- #
def test_device_placement():
    """디바이스 배치 테스트"""
    print("\n" + "="*60)
    print("테스트 4: 디바이스 배치")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")                 # 베이스라인 Config

    # 모델 로더 생성
    loader = ModelLoader(config)                            # 로더 초기화

    # 디바이스 확인
    print(f"\n  사용 디바이스: {loader.device}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU 개수: {torch.cuda.device_count()}")

    print("\n✅ 디바이스 배치 테스트 성공!")


# ---------------------- 전체 파이프라인 테스트 ---------------------- #
def test_full_pipeline():
    """전체 로딩 파이프라인 테스트"""
    print("\n" + "="*60)
    print("테스트 5: 전체 파이프라인")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")                 # 베이스라인 Config

    # 편의 함수로 한 번에 로드
    model, tokenizer = load_model_and_tokenizer(config)     # 모델 및 토크나이저 로드

    # 검증
    assert model is not None                                # 모델 존재 확인
    assert tokenizer is not None                            # 토크나이저 존재 확인

    # -------------- 간단한 추론 테스트 -------------- #
    print("\n간단한 추론 테스트:")
    test_text = "#Person1#: 안녕하세요\\n#Person2#: 반갑습니다"

    # 토크나이징
    inputs = tokenizer(
        test_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 디바이스로 이동 (token_type_ids 제거)
    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
        if k in ['input_ids', 'attention_mask']             # BART는 이 두 개만 사용
    }

    # 추론 (평가 모드)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=2
        )

    # 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  입력: {test_text[:30]}...")
    print(f"  생성: {generated_text[:50]}...")

    print("\n✅ 전체 파이프라인 테스트 성공!")


# ==================== 메인 실행부 ==================== #
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Model Loader 테스트 시작")
    print("="*60)

    try:
        # 모든 테스트 실행
        test_load_tokenizer()                               # 테스트 1
        test_special_tokens()                               # 테스트 2
        test_load_model()                                   # 테스트 3
        test_device_placement()                             # 테스트 4
        test_full_pipeline()                                # 테스트 5

        # 최종 결과
        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 테스트 실패: {e}")
        print("="*60)
        raise
