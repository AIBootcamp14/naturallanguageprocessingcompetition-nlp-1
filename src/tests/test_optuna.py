"""
Optuna 최적화 시스템 테스트

PRD 13: Optuna 하이퍼파라미터 최적화 전략 구현
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ------------------------- 서드파티 라이브러리 ------------------------- #
import optuna
from omegaconf import OmegaConf

# ------------------------- 프로젝트 모듈 ------------------------- #
from src.optimization import OptunaOptimizer, create_optuna_optimizer


# ==================== 테스트 함수 정의 ==================== #
# ---------------------- OptunaOptimizer 초기화 테스트 ---------------------- #
def test_optuna_optimizer_init():
    """OptunaOptimizer 초기화 테스트"""
    # 테스트 헤더 출력
    print("\n" + "="*60)
    print("테스트 1: OptunaOptimizer 초기화")
    print("="*60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Mock config 생성
        config = OmegaConf.create({
            'model': {
                'name': 'test_model',
                'hidden_dropout_prob': 0.1,
                'attention_dropout_prob': 0.1
            },
            'training': {
                'learning_rate': 5e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'scheduler_type': 'linear'
            },
            'generation': {
                'temperature': 1.0,
                'top_p': 0.9,
                'num_beams': 4,
                'length_penalty': 1.0
            },
            'lora': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.05
            },
            'logging': {
                'use_wandb': False
            }
        })

        # OptunaOptimizer 인스턴스 생성
        optimizer = OptunaOptimizer(
            config=config,
            train_dataset=None,
            val_dataset=None,
            n_trials=10,
            study_name="test_study"
        )

        # 성공 메시지 출력
        print("✅ OptunaOptimizer 초기화 성공")
        print(f"  - Study 이름: {optimizer.study_name}")
        print(f"  - Trial 횟수: {optimizer.n_trials}")
        print(f"  - 방향: {optimizer.direction}")

        # -------------- 검증 -------------- #
        # 초기화 값 검증
        assert optimizer.study_name == "test_study", "Study 이름이 다름"
        assert optimizer.n_trials == 10, "Trial 횟수가 다름"
        assert optimizer.direction == "maximize", "방향이 다름"

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 탐색 공간 생성 테스트 ---------------------- #
def test_create_search_space():
    """탐색 공간 생성 테스트"""
    # 테스트 헤더 출력
    print("\n" + "="*60)
    print("테스트 2: 탐색 공간 생성")
    print("="*60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Mock config 생성
        config = OmegaConf.create({
            'model': {
                'name': 'test_model',
                'hidden_dropout_prob': 0.1,
                'attention_dropout_prob': 0.1
            },
            'training': {
                'learning_rate': 5e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'scheduler_type': 'linear'
            },
            'generation': {
                'temperature': 1.0,
                'top_p': 0.9,
                'num_beams': 4,
                'length_penalty': 1.0
            },
            'lora': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.05
            }
        })

        # OptunaOptimizer 생성
        optimizer = OptunaOptimizer(
            config=config,
            train_dataset=None,
            val_dataset=None,
            n_trials=1
        )

        # Mock Trial 생성
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        # 탐색 공간 샘플링
        params = optimizer.create_search_space(trial)

        # -------------- 샘플링 결과 출력 -------------- #
        print("샘플링된 파라미터:")
        for key, value in params.items():
            print(f"  - {key}: {value}")

        # -------------- 검증 -------------- #
        # 필수 파라미터 존재 확인
        required_params = [
            'learning_rate', 'batch_size', 'num_epochs',
            'warmup_ratio', 'weight_decay', 'scheduler_type',
            'temperature', 'top_p', 'num_beams', 'length_penalty'
        ]

        for param in required_params:
            assert param in params, f"필수 파라미터 누락: {param}"

        # LoRA 파라미터 검증
        assert 'lora_r' in params, "LoRA r 파라미터 누락"
        assert 'lora_alpha' in params, "LoRA alpha 파라미터 누락"
        assert 'lora_dropout' in params, "LoRA dropout 파라미터 누락"

        # Dropout 파라미터 검증
        assert 'hidden_dropout' in params, "hidden_dropout 파라미터 누락"
        assert 'attention_dropout' in params, "attention_dropout 파라미터 누락"

        print("✅ 탐색 공간 생성 성공")
        print(f"  - 총 파라미터 수: {len(params)}")

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 탐색 공간 범위 테스트 ---------------------- #
def test_search_space_ranges():
    """탐색 공간 범위 테스트"""
    # 테스트 헤더 출력
    print("\n" + "="*60)
    print("테스트 3: 탐색 공간 범위 검증")
    print("="*60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Mock config 생성
        config = OmegaConf.create({
            'model': {
                'name': 'test_model',
                'hidden_dropout_prob': 0.1,
                'attention_dropout_prob': 0.1
            },
            'training': {
                'learning_rate': 5e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'scheduler_type': 'linear'
            },
            'generation': {
                'temperature': 1.0,
                'top_p': 0.9,
                'num_beams': 4,
                'length_penalty': 1.0
            },
            'lora': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.05
            }
        })

        # OptunaOptimizer 생성
        optimizer = OptunaOptimizer(
            config=config,
            train_dataset=None,
            val_dataset=None,
            n_trials=1
        )

        # 여러 번 샘플링하여 범위 확인
        study = optuna.create_study(direction="maximize")

        samples = []
        for _ in range(10):
            trial = study.ask()
            params = optimizer.create_search_space(trial)
            samples.append(params)
            study.tell(trial, 0.5)

        # -------------- 범위 검증 -------------- #
        print("파라미터 범위 확인:")

        # Learning rate: 1e-6 ~ 1e-4
        lr_values = [s['learning_rate'] for s in samples]
        assert all(1e-6 <= lr <= 1e-4 for lr in lr_values), "learning_rate 범위 초과"
        print(f"  ✓ learning_rate: {min(lr_values):.2e} ~ {max(lr_values):.2e}")

        # Batch size: [8, 16, 32, 64]
        bs_values = [s['batch_size'] for s in samples]
        assert all(bs in [8, 16, 32, 64] for bs in bs_values), "batch_size 범위 초과"
        print(f"  ✓ batch_size: {set(bs_values)}")

        # Num epochs: 3 ~ 10
        epoch_values = [s['num_epochs'] for s in samples]
        assert all(3 <= e <= 10 for e in epoch_values), "num_epochs 범위 초과"
        print(f"  ✓ num_epochs: {min(epoch_values)} ~ {max(epoch_values)}")

        # Temperature: 0.1 ~ 1.0
        temp_values = [s['temperature'] for s in samples]
        assert all(0.1 <= t <= 1.0 for t in temp_values), "temperature 범위 초과"
        print(f"  ✓ temperature: {min(temp_values):.2f} ~ {max(temp_values):.2f}")

        # LoRA r: [8, 16, 32, 64]
        lora_r_values = [s['lora_r'] for s in samples]
        assert all(r in [8, 16, 32, 64] for r in lora_r_values), "lora_r 범위 초과"
        print(f"  ✓ lora_r: {set(lora_r_values)}")

        print("✅ 탐색 공간 범위 검증 성공")

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- Sampler 및 Pruner 설정 테스트 ---------------------- #
def test_sampler_and_pruner():
    """Sampler 및 Pruner 설정 테스트"""
    # 테스트 헤더 출력
    print("\n" + "="*60)
    print("테스트 4: Sampler 및 Pruner 설정")
    print("="*60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Optuna 라이브러리 import
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner

        # Sampler 생성
        sampler = TPESampler(seed=42)
        print(f"✓ TPESampler 생성: {type(sampler).__name__}")

        # Pruner 생성
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )
        print(f"✓ MedianPruner 생성: {type(pruner).__name__}")

        # Study 생성
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )

        print(f"✓ Study 생성 완료")
        print(f"  - Sampler: {type(study.sampler).__name__}")
        print(f"  - Pruner: {type(study.pruner).__name__}")

        print("✅ Sampler 및 Pruner 설정 성공")

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- Best params 메서드 테스트 ---------------------- #
def test_best_params_methods():
    """Best params 메서드 테스트"""
    # 테스트 헤더 출력
    print("\n" + "="*60)
    print("테스트 5: Best params 메서드")
    print("="*60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Mock config 생성
        config = OmegaConf.create({
            'model': {'name': 'test'},
            'training': {
                'learning_rate': 5e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'scheduler_type': 'linear'
            },
            'generation': {
                'temperature': 1.0,
                'top_p': 0.9,
                'num_beams': 4,
                'length_penalty': 1.0
            }
        })

        # OptunaOptimizer 생성
        optimizer = OptunaOptimizer(
            config=config,
            train_dataset=None,
            val_dataset=None,
            n_trials=1
        )

        # -------------- optimize() 전 에러 확인 -------------- #
        # optimize() 전에는 에러 발생해야 함
        try:
            optimizer.get_best_params()
            print("❌ optimize() 전에 get_best_params() 호출 가능 (예상: 에러)")
            return False
        except ValueError as e:
            print(f"✓ optimize() 전 에러 발생: {str(e)}")

        # -------------- Mock 결과 설정 -------------- #
        # Mock 결과 설정
        optimizer.best_params = {'learning_rate': 3e-5, 'batch_size': 32}
        optimizer.best_value = 0.45

        # 메서드 테스트
        best_params = optimizer.get_best_params()
        best_value = optimizer.get_best_value()

        print(f"✓ get_best_params(): {best_params}")
        print(f"✓ get_best_value(): {best_value}")

        # -------------- 검증 -------------- #
        assert best_params == {'learning_rate': 3e-5, 'batch_size': 32}, "best_params 불일치"
        assert best_value == 0.45, "best_value 불일치"

        print("✅ Best params 메서드 테스트 성공")

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- 결과 저장 테스트 ---------------------- #
def test_save_results():
    """결과 저장 테스트"""
    # 테스트 헤더 출력
    print("\n" + "="*60)
    print("테스트 6: 결과 저장")
    print("="*60)

    # -------------- 테스트 실행 -------------- #
    try:
        import tempfile
        import shutil
        import json

        # Mock config 생성
        config = OmegaConf.create({
            'model': {'name': 'test'},
            'training': {
                'learning_rate': 5e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'scheduler_type': 'linear'
            },
            'generation': {
                'temperature': 1.0,
                'top_p': 0.9,
                'num_beams': 4,
                'length_penalty': 1.0
            }
        })

        # OptunaOptimizer 생성
        optimizer = OptunaOptimizer(
            config=config,
            train_dataset=None,
            val_dataset=None,
            n_trials=1
        )

        # -------------- Mock Study 생성 -------------- #
        # Mock Study 생성
        study = optuna.create_study(direction="maximize")
        for i in range(3):
            trial = study.ask()
            study.tell(trial, 0.4 + i * 0.05)

        optimizer.study = study
        optimizer.best_params = study.best_params
        optimizer.best_value = study.best_value

        # -------------- 파일 저장 -------------- #
        # 임시 디렉토리에 저장
        temp_dir = tempfile.mkdtemp()

        optimizer.save_results(temp_dir)

        # 파일 존재 확인
        best_params_path = Path(temp_dir) / "best_params.json"
        trials_csv_path = Path(temp_dir) / "all_trials.csv"
        stats_path = Path(temp_dir) / "study_stats.json"

        assert best_params_path.exists(), "best_params.json 없음"
        assert trials_csv_path.exists(), "all_trials.csv 없음"
        assert stats_path.exists(), "study_stats.json 없음"

        # -------------- 파일 내용 확인 -------------- #
        # 내용 확인
        with open(best_params_path, 'r') as f:
            best_data = json.load(f)
            print(f"✓ best_params.json: {best_data['best_value']}")

        with open(stats_path, 'r') as f:
            stats = json.load(f)
            print(f"✓ study_stats.json:")
            print(f"    - 완료: {stats['n_completed']}")
            print(f"    - Best value: {stats['best_value']}")

        # 정리
        shutil.rmtree(temp_dir)

        print("✅ 결과 저장 테스트 성공")

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------- create_optuna_optimizer 편의 함수 테스트 ---------------------- #
def test_create_optuna_optimizer():
    """create_optuna_optimizer 편의 함수 테스트"""
    # 테스트 헤더 출력
    print("\n" + "="*60)
    print("테스트 7: create_optuna_optimizer 편의 함수")
    print("="*60)

    # -------------- 테스트 실행 -------------- #
    try:
        # Mock config 생성
        config = OmegaConf.create({
            'model': {'name': 'test'},
            'training': {
                'learning_rate': 5e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'scheduler_type': 'linear'
            },
            'generation': {
                'temperature': 1.0,
                'top_p': 0.9,
                'num_beams': 4,
                'length_penalty': 1.0
            }
        })

        # 편의 함수로 Optimizer 생성
        optimizer = create_optuna_optimizer(
            config=config,
            train_dataset=None,
            val_dataset=None,
            n_trials=20,
            study_name="test_convenience"
        )

        print("✅ create_optuna_optimizer 성공")
        print(f"  - Trial 횟수: {optimizer.n_trials}")
        print(f"  - Study 이름: {optimizer.study_name}")

        # -------------- 검증 -------------- #
        assert optimizer.n_trials == 20, "Trial 횟수 불일치"
        assert optimizer.study_name == "test_convenience", "Study 이름 불일치"

        return True

    # -------------- 예외 처리 -------------- #
    except Exception as e:
        print(f"❌ 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 메인 실행부 ==================== #
# ---------------------- 전체 테스트 실행 함수 ---------------------- #
def main():
    """전체 테스트 실행"""
    # 테스트 시작 헤더 출력
    print("\n" + "="*70)
    print(" "*22 + "Optuna 최적화 시스템 테스트 시작")
    print("="*70)

    # -------------- 테스트 실행 -------------- #
    # 테스트 결과 수집용 리스트
    results = []
    results.append(("OptunaOptimizer 초기화", test_optuna_optimizer_init()))
    results.append(("탐색 공간 생성", test_create_search_space()))
    results.append(("탐색 공간 범위 검증", test_search_space_ranges()))
    results.append(("Sampler 및 Pruner 설정", test_sampler_and_pruner()))
    results.append(("Best params 메서드", test_best_params_methods()))
    results.append(("결과 저장", test_save_results()))
    results.append(("create_optuna_optimizer 함수", test_create_optuna_optimizer()))

    # -------------- 결과 요약 출력 -------------- #
    print("\n" + "="*70)
    print(" "*25 + "테스트 결과 요약")
    print("="*70)

    # 통과한 테스트 개수 계산
    passed = sum(1 for _, result in results if result)
    total = len(results)

    # 각 테스트 결과 출력
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")

    # 최종 요약 출력
    print("="*70)
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    print("="*70)

    # 주의사항 출력
    print("\n⚠️  참고사항:")
    print("- 실제 optimize() 테스트는 데이터셋과 모델이 필요합니다")
    print("- 이 테스트는 초기화 및 설정 기능만 검증합니다")

    # 전체 테스트 통과 여부 반환
    return passed == total


# ---------------------- 메인 진입점 ---------------------- #
if __name__ == "__main__":
    success = main()  # 전체 테스트 실행
    sys.exit(0 if success else 1)  # 성공 시 0, 실패 시 1 반환
