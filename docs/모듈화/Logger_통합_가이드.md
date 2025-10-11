# Logger 통합 가이드

## 개요

모든 모듈에서 print 대신 `src/logging/logger.py`의 Logger 클래스를 사용해야 합니다.

## Logger 사용법

### 1. Logger 초기화

```python
from src.logging.logger import Logger
from src.utils.core.common import create_log_path

# 로그 파일 경로 생성
log_path = create_log_path("outputs/logs", "train")  # outputs/logs/train_20251011_143000.log

# Logger 초기화
logger = Logger(log_path, print_also=True)

# stdout/stderr 리다이렉션 시작
logger.start_redirect()

# 작업 수행...

# 리다이렉션 종료
logger.stop_redirect()
logger.close()
```

### 2. print 대신 logger.write 사용

```python
# ❌ 기존 방식
print("=" * 60)
print("모델 로딩 시작")
print("=" * 60)

# ✅ 올바른 방식
logger.write("=" * 60)
logger.write("모델 로딩 시작")
logger.write("=" * 60)
```

### 3. 에러 로깅

```python
# 에러 메시지 (빨간색으로 출력)
logger.write("❌ 모델 로드 실패!", print_error=True)
```

## 수정이 필요한 파일들

### src/models/model_loader.py

```python
# 수정 전
print(f"토크나이저 로딩: {checkpoint}")
print(f"  → 특수 토큰 {num_added}개 추가됨")

# 수정 후
if logger:
    logger.write(f"토크나이저 로딩: {checkpoint}")
    logger.write(f"  → 특수 토큰 {num_added}개 추가됨")
```

### src/training/trainer.py

```python
# 수정 전
print("=" * 60)
print("모델 학습 시작")
print("=" * 60)

# 수정 후
if logger:
    logger.write("=" * 60)
    logger.write("모델 학습 시작")
    logger.write("=" * 60)
```

### src/inference/predictor.py

```python
# 수정 전
print("=" * 60)
print("제출 파일 생성 시작")
print("=" * 60)

# 수정 후
if logger:
    logger.write("=" * 60)
    logger.write("제출 파일 생성 시작")
    logger.write("=" * 60)
```

## GPU 유틸리티 통합

### src/utils/gpu_optimization/team_gpu_check.py 활용

```python
from src.utils.gpu_optimization.team_gpu_check import (
    check_gpu_tier,
    get_gpu_info,
    get_optimal_batch_size,
    get_memory_usage
)

# GPU 정보 출력
gpu_info = get_gpu_info()
if logger:
    logger.write(f"GPU 정보: {gpu_info}")

# GPU tier 확인
gpu_tier = check_gpu_tier()
if logger:
    logger.write(f"GPU Tier: {gpu_tier}")

# 최적 배치 크기 추천
optimal_batch_size = get_optimal_batch_size("kobart", gpu_tier)
if logger:
    logger.write(f"추천 배치 크기: {optimal_batch_size}")
```

## 통합 예시

### scripts/train.py에 Logger 통합

```python
from src.logging.logger import Logger
from src.utils.core.common import create_log_path
from src.utils.gpu_optimization.team_gpu_check import get_gpu_info, check_gpu_tier

def main():
    # Logger 초기화
    log_path = create_log_path("outputs/logs", f"train_{args.experiment}")
    logger = Logger(log_path, print_also=True)
    logger.start_redirect()

    try:
        logger.write("=" * 60)
        logger.write(f"학습 시작: {args.experiment}")
        logger.write("=" * 60)

        # GPU 정보 출력
        gpu_info = get_gpu_info()
        logger.write(f"\nGPU 정보:")
        for key, value in gpu_info.items():
            logger.write(f"  {key}: {value}")

        # Config 로드
        logger.write("\n[1/6] Config 로딩...")
        config = load_config(args.experiment)

        # ... 나머지 작업

    finally:
        logger.stop_redirect()
        logger.close()
```

## 체크리스트

- [ ] src/models/model_loader.py - Logger 통합
- [ ] src/training/trainer.py - Logger 통합
- [ ] src/inference/predictor.py - Logger 통합
- [ ] scripts/train.py - Logger 통합
- [ ] scripts/inference.py - Logger 통합
