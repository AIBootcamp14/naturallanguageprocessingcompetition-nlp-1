# ==================== Trainer 테스트 스크립트 ==================== #
"""
Trainer 시스템 테스트

테스트 항목:
1. Trainer 초기화
2. 학습 인자 생성
3. ROUGE 메트릭 계산 함수
4. Trainer 생성
5. 통합 테스트 (초기화만)
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------- 서드파티 라이브러리 ---------------------- #
import torch
import numpy as np

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.config import load_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import ModelTrainer, create_trainer


# ==================== 테스트 함수들 ==================== #
# ---------------------- Trainer 초기화 테스트 ---------------------- #
def test_trainer_initialization():
    """Trainer 초기화 테스트"""
    print("\n" + "="*60)
    print("테스트 1: Trainer 초기화")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")             # 베이스라인 Config
    print(f"  Config 로드 완료")

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(config)  # 모델 로드
    print(f"  모델 로드 완료")

    # 더미 데이터셋 생성
    dummy_dialogues = ["#Person1#: 안녕하세요\\n#Person2#: 반갑습니다"] * 10
    dummy_summaries = ["두 사람이 인사를 나눴다"] * 10

    train_dataset = DialogueSummarizationDataset(      # 학습 데이터셋
        dialogues=dummy_dialogues,
        summaries=dummy_summaries,
        tokenizer=tokenizer,
        encoder_max_len=128,
        decoder_max_len=50,
        preprocess=False
    )
    print(f"  데이터셋 생성 완료: {len(train_dataset)}개 샘플")

    # Trainer 초기화 (WandB 비활성화)
    trainer = ModelTrainer(                             # Trainer 생성
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,                     # 평가용으로 같은 데이터 사용
        use_wandb=False                                 # WandB 비활성화
    )

    # 검증
    assert trainer.model is not None                    # 모델 존재 확인
    assert trainer.tokenizer is not None                # 토크나이저 존재 확인
    assert trainer.train_dataset is not None            # 학습 데이터셋 확인
    assert trainer.rouge_calculator is not None         # ROUGE 계산기 확인

    print("\n✅ Trainer 초기화 테스트 성공!")


# ---------------------- 학습 인자 생성 테스트 ---------------------- #
def test_training_args():
    """학습 인자 생성 테스트"""
    print("\n" + "="*60)
    print("테스트 2: 학습 인자 생성")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")             # 베이스라인 Config

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(config)  # 모델 로드

    # 더미 데이터셋
    dummy_dialogues = ["#Person1#: 안녕"] * 5
    dummy_summaries = ["인사"] * 5

    train_dataset = DialogueSummarizationDataset(
        dialogues=dummy_dialogues,
        summaries=dummy_summaries,
        tokenizer=tokenizer,
        preprocess=False
    )

    # Trainer 초기화
    trainer = ModelTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        use_wandb=False
    )

    # 학습 인자 확인
    args = trainer.training_args                        # 학습 인자
    print(f"\n  출력 디렉토리: {args.output_dir}")
    print(f"  에포크 수: {args.num_train_epochs}")
    print(f"  배치 크기: {args.per_device_train_batch_size}")
    print(f"  학습률: {args.learning_rate}")
    print(f"  평가 전략: {args.eval_strategy}")
    print(f"  Beam 수: {args.generation_num_beams}")

    # 검증
    assert args.num_train_epochs == config.training.epochs  # 에포크 확인
    assert args.per_device_train_batch_size == config.training.batch_size  # 배치 크기 확인
    assert args.predict_with_generate is True           # 생성 모드 확인

    print("\n✅ 학습 인자 생성 테스트 성공!")


# ---------------------- ROUGE 메트릭 계산 함수 테스트 ---------------------- #
def test_compute_metrics():
    """ROUGE 메트릭 계산 함수 테스트"""
    print("\n" + "="*60)
    print("테스트 3: ROUGE 메트릭 계산 함수")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(config)

    # 더미 데이터셋
    train_dataset = DialogueSummarizationDataset(
        dialogues=["안녕"],
        summaries=["인사"],
        tokenizer=tokenizer,
        preprocess=False
    )

    # Trainer 초기화
    trainer = ModelTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        use_wandb=False
    )

    # 더미 예측 데이터 생성
    # 실제로는 모델 출력이지만, 여기서는 토크나이징된 텍스트 사용
    pred_text = "두 사람이 인사를 나눴다"
    label_text = "두 사람이 서로 인사했다"

    # 토크나이징
    pred_ids = tokenizer.encode(pred_text, add_special_tokens=False)
    label_ids = tokenizer.encode(label_text, add_special_tokens=False)

    # 패딩 (최대 길이 30으로)
    max_len = 30
    pred_ids = pred_ids + [tokenizer.pad_token_id] * (max_len - len(pred_ids))
    label_ids = label_ids + [-100] * (max_len - len(label_ids))  # -100은 무시됨

    # numpy 배열로 변환 (배치 형태)
    predictions = np.array([pred_ids])                  # (1, max_len)
    labels = np.array([label_ids])                      # (1, max_len)

    # ROUGE 계산
    eval_preds = (predictions, labels)
    metrics = trainer.compute_metrics(eval_preds)       # 메트릭 계산

    # 결과 출력
    print(f"\n  예측: {pred_text}")
    print(f"  정답: {label_text}")
    print(f"\n  ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"  ROUGE Sum: {metrics['rouge_sum']:.4f}")

    # 검증
    assert 'rouge1' in metrics                          # ROUGE-1 존재 확인
    assert 'rouge_sum' in metrics                       # ROUGE Sum 존재 확인
    assert 0 <= metrics['rouge1'] <= 1                  # 점수 범위 확인

    print("\n✅ ROUGE 메트릭 계산 함수 테스트 성공!")


# ---------------------- HuggingFace Trainer 생성 테스트 ---------------------- #
def test_hf_trainer_creation():
    """HuggingFace Trainer 생성 테스트"""
    print("\n" + "="*60)
    print("테스트 4: HuggingFace Trainer 생성")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(config)

    # 더미 데이터셋
    train_dataset = DialogueSummarizationDataset(
        dialogues=["안녕하세요"] * 5,
        summaries=["인사"] * 5,
        tokenizer=tokenizer,
        preprocess=False
    )

    # Trainer 초기화
    trainer = ModelTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        use_wandb=False
    )

    # HuggingFace Trainer 생성
    hf_trainer = trainer._create_trainer()              # Trainer 생성

    # 검증
    assert hf_trainer is not None                       # Trainer 존재 확인
    assert hf_trainer.model is not None                 # 모델 확인
    assert hf_trainer.train_dataset is not None         # 학습 데이터셋 확인

    print(f"\n  Trainer 타입: {type(hf_trainer).__name__}")
    print(f"  모델: {type(hf_trainer.model).__name__}")
    print(f"  학습 샘플 수: {len(hf_trainer.train_dataset)}")

    print("\n✅ HuggingFace Trainer 생성 테스트 성공!")


# ---------------------- 편의 함수 테스트 ---------------------- #
def test_convenience_function():
    """편의 함수 테스트"""
    print("\n" + "="*60)
    print("테스트 5: 편의 함수")
    print("="*60)

    # Config 로드
    config = load_config("baseline_kobart")

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(config)

    # 더미 데이터셋
    train_dataset = DialogueSummarizationDataset(
        dialogues=["안녕"],
        summaries=["인사"],
        tokenizer=tokenizer,
        preprocess=False
    )

    # 편의 함수로 Trainer 생성
    trainer = create_trainer(                           # 편의 함수 사용
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        use_wandb=False
    )

    # 검증
    assert isinstance(trainer, ModelTrainer)            # 타입 확인

    print(f"\n  Trainer 생성 완료")
    print(f"  타입: {type(trainer).__name__}")

    print("\n✅ 편의 함수 테스트 성공!")


# ==================== 메인 실행부 ==================== #
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Trainer 테스트 시작")
    print("="*60)
    print("\n⚠️ 참고: 실제 학습은 시간이 오래 걸리므로 초기화 및 설정만 테스트합니다.")

    try:
        # 모든 테스트 실행
        test_trainer_initialization()                   # 테스트 1
        test_training_args()                            # 테스트 2
        test_compute_metrics()                          # 테스트 3
        test_hf_trainer_creation()                      # 테스트 4
        test_convenience_function()                     # 테스트 5

        # 최종 결과
        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 테스트 실패: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        raise
