# ==================== Config Loader 테스트 스크립트 ==================== #
"""
Config Loader 시스템 테스트

테스트 항목:
1. 기본 설정 로드
2. 모델 타입별 설정 로드
3. 모델별 설정 로드
4. 실험 설정 로드
5. 전체 설정 병합
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.config import ConfigLoader, load_config


# ==================== 테스트 함수들 ==================== #
# ---------------------- 기본 설정 로드 테스트 ---------------------- #
def test_load_base():
    """기본 설정 로드 테스트"""
    print("\n" + "="*60)
    print("테스트 1: 기본 설정 로드")
    print("="*60)

    loader = ConfigLoader()                             # Config 로더 생성
    config = loader.load_base()                         # 기본 설정 로드

    print(f"✅ 실험 이름: {config.experiment.name}")
    print(f"✅ 시드: {config.experiment.seed}")
    print(f"✅ WandB 프로젝트: {config.experiment.wandb_project}")

    assert config.experiment.name == "default"          # 실험 이름 확인
    assert config.experiment.seed == 42                 # 시드 확인

    print("✅ 기본 설정 로드 성공!")


# ---------------------- 모델 타입별 설정 로드 테스트 ---------------------- #
def test_load_model_type():
    """모델 타입별 설정 로드 테스트"""
    print("\n" + "="*60)
    print("테스트 2: 모델 타입별 설정 로드")
    print("="*60)

    loader = ConfigLoader()                             # Config 로더 생성
    config = loader.load_model_type("encoder_decoder")  # Encoder-Decoder 설정 로드

    print(f"✅ 모델 타입: {config.model.type}")
    print(f"✅ 인코더 최대 길이: {config.tokenizer.encoder_max_len}")
    print(f"✅ 디코더 최대 길이: {config.tokenizer.decoder_max_len}")

    assert config.model.type == "encoder_decoder"       # 모델 타입 확인
    assert config.tokenizer.encoder_max_len == 512      # 인코더 최대 길이 확인

    print("✅ 모델 타입별 설정 로드 성공!")


# ---------------------- 모델별 설정 로드 테스트 ---------------------- #
def test_load_model():
    """모델별 설정 로드 테스트"""
    print("\n" + "="*60)
    print("테스트 3: 모델별 설정 로드")
    print("="*60)

    loader = ConfigLoader()                             # Config 로더 생성
    config = loader.load_model("kobart")                # KoBART 설정 로드

    print(f"✅ 모델 이름: {config.model.name}")
    print(f"✅ 체크포인트: {config.model.checkpoint}")
    print(f"✅ 학습률: {config.training.learning_rate}")

    assert config.model.name == "kobart"                # 모델 이름 확인
    assert config.training.learning_rate == 1.0e-05     # 학습률 확인

    print("✅ 모델별 설정 로드 성공!")


# ---------------------- 실험 설정 로드 테스트 ---------------------- #
def test_load_experiment():
    """실험 설정 로드 테스트"""
    print("\n" + "="*60)
    print("테스트 4: 실험 설정 로드")
    print("="*60)

    loader = ConfigLoader()                             # Config 로더 생성
    config = loader.load_experiment("baseline_kobart")  # 실험 설정 로드

    print(f"✅ 실험 이름: {config.experiment.name}")
    print(f"✅ 실험 설명: {config.experiment.description}")
    print(f"✅ 모델 이름: {config.model.name}")

    assert config.experiment.name == "baseline_kobart"  # 실험 이름 확인
    assert config.model.name == "kobart"                # 모델 이름 확인

    print("✅ 실험 설정 로드 성공!")


# ---------------------- 전체 설정 병합 테스트 ---------------------- #
def test_merge_configs():
    """전체 설정 병합 테스트"""
    print("\n" + "="*60)
    print("테스트 5: 전체 설정 병합")
    print("="*60)

    loader = ConfigLoader()                             # Config 로더 생성
    config = loader.merge_configs("baseline_kobart")    # 전체 설정 병합

    # -------------- 실험 정보 확인 -------------- #
    print("\n[실험 정보]")
    print(f"  실험 이름: {config.experiment.name}")
    print(f"  시드: {config.experiment.seed}")
    print(f"  WandB 사용: {config.experiment.use_wandb}")

    # -------------- 모델 정보 확인 -------------- #
    print("\n[모델 정보]")
    print(f"  모델 이름: {config.model.name}")
    print(f"  모델 타입: {config.model.type}")
    print(f"  체크포인트: {config.model.checkpoint}")

    # -------------- 학습 설정 확인 -------------- #
    print("\n[학습 설정]")
    print(f"  에포크 수: {config.training.num_train_epochs}")
    print(f"  학습률: {config.training.learning_rate}")
    print(f"  배치 크기: {config.training.per_device_train_batch_size}")

    # -------------- 추론 설정 확인 -------------- #
    print("\n[추론 설정]")
    print(f"  빔 개수: {config.inference.num_beams}")
    print(f"  반복 방지 n-gram: {config.inference.no_repeat_ngram_size}")

    # -------------- 경로 설정 확인 -------------- #
    print("\n[경로 설정]")
    print(f"  출력 디렉토리: {config.paths.output_dir}")
    print(f"  모델 저장 디렉토리: {config.paths.model_save_dir}")

    # -------------- 검증 -------------- #
    assert config.experiment.name == "baseline_kobart"  # 실험 이름
    assert config.model.name == "kobart"                # 모델 이름
    assert config.model.type == "encoder_decoder"       # 모델 타입
    assert config.training.learning_rate == 1.0e-05     # 학습률
    assert config.training.per_device_train_batch_size == 50  # 배치 크기
    assert config.inference.no_repeat_ngram_size == 2   # 반복 방지 n-gram

    print("\n✅ 전체 설정 병합 성공!")


# ---------------------- 편의 함수 테스트 ---------------------- #
def test_load_config_function():
    """load_config 편의 함수 테스트"""
    print("\n" + "="*60)
    print("테스트 6: load_config 편의 함수")
    print("="*60)

    config = load_config("baseline_kobart")             # 편의 함수로 설정 로드

    print(f"✅ 실험 이름: {config.experiment.name}")
    print(f"✅ 모델 이름: {config.model.name}")

    assert config.experiment.name == "baseline_kobart"  # 실험 이름 확인
    assert config.model.name == "kobart"                # 모델 이름 확인

    print("✅ 편의 함수 테스트 성공!")


# ==================== 메인 실행부 ==================== #
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Config Loader 테스트 시작")
    print("="*60)

    try:
        # 모든 테스트 실행
        test_load_base()                                # 테스트 1
        test_load_model_type()                          # 테스트 2
        test_load_model()                               # 테스트 3
        test_load_experiment()                          # 테스트 4
        test_merge_configs()                            # 테스트 5
        test_load_config_function()                     # 테스트 6

        # 최종 결과
        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 테스트 실패: {e}")
        print("="*60)
        raise
