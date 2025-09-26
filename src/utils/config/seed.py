# 표준 및 외부 라이브러리 임포트
import random, os, numpy as np, torch   # random : 파이썬 난수 발생
                                        # os     : 환경 변수 등 시스템 제어
                                        # numpy  : 수치 연산 난수 제어
                                        # torch  : PyTorch 난수 제어


# ---------------------- 시드 고정 함수 ---------------------- #
def set_seed(seed: int = 42, deterministic: bool = True):  # 시드 설정 함수 정의
    """
    전체 학습 파이프라인에서 재현성을 확보하기 위해
    파이썬, NumPy, PyTorch의 난수를 고정하는 함수
    """

    # Python 내장 random 시드 고정
    random.seed(seed)                             # 파이썬 난수 생성기 시드 설정

    # NumPy 난수 시드 고정
    np.random.seed(seed)                          # NumPy 난수 생성기 시드 설정

    # PyTorch CPU 연산 난수 시드 고정
    torch.manual_seed(seed)                       # PyTorch CPU 난수 생성기 시드 설정

    # PyTorch CUDA 연산(멀티 GPU 포함) 시드 고정
    torch.cuda.manual_seed_all(seed)              # 모든 GPU 디바이스 난수 생성기 시드 설정

    # deterministic 모드 활성화 시
    if deterministic:                             # 결정론적 모드 활성화 조건
        # cudnn의 연산을 결정론적으로 설정 (속도↓, 일관성↑)
        torch.backends.cudnn.deterministic = True  # CUDNN 결정론적 모드 활성화
        # 벤치마크 모드 비활성화 (재현성 확보)
        torch.backends.cudnn.benchmark = False   # CUDNN 벤치마크 모드 비활성화