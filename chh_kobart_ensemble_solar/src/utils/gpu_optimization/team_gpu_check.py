"""
GPU 체크 및 최적화 유틸리티
대화 요약 대회를 위한 GPU 설정 확인
"""

import torch
import subprocess
import os
from typing import Dict, Optional, Tuple


def check_gpu_tier() -> str:
    """
    현재 GPU의 tier를 확인하여 반환

    Returns:
        str: GPU tier (HIGH/MEDIUM/LOW/CPU)
    """
    if not torch.cuda.is_available():
        return "CPU"

    # GPU 메모리 확인 (GB 단위)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0).lower()

    # GPU tier 분류
    if gpu_memory >= 40:  # A100, A6000 등
        return "HIGH"
    elif gpu_memory >= 24:  # RTX 3090, RTX 4090 등
        return "MEDIUM"
    elif gpu_memory >= 10:  # RTX 3060, T4 등
        return "LOW"
    else:
        return "LOW"


def get_gpu_info() -> Dict:
    """
    GPU 상세 정보를 반환

    Returns:
        Dict: GPU 정보 딕셔너리
    """
    info = {}

    if torch.cuda.is_available():
        info['available'] = True
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()

        for i in range(torch.cuda.device_count()):
            device_info = {}
            props = torch.cuda.get_device_properties(i)

            device_info['name'] = props.name
            device_info['total_memory_gb'] = props.total_memory / 1024**3
            device_info['major'] = props.major
            device_info['minor'] = props.minor
            device_info['multi_processor_count'] = props.multi_processor_count

            # 현재 사용중인 메모리
            device_info['allocated_memory_gb'] = torch.cuda.memory_allocated(i) / 1024**3
            device_info['reserved_memory_gb'] = torch.cuda.memory_reserved(i) / 1024**3

            info[f'gpu_{i}'] = device_info
    else:
        info['available'] = False
        info['device_count'] = 0

    return info


def get_optimal_batch_size(model_type: str = "kobart", gpu_tier: Optional[str] = None) -> int:
    """
    모델 타입과 GPU tier에 따른 최적 배치 크기 반환

    Args:
        model_type: 모델 종류 (kobart, solar, polyglot, kullm)
        gpu_tier: GPU tier (None인 경우 자동 감지)

    Returns:
        int: 추천 배치 크기
    """
    if gpu_tier is None:
        gpu_tier = check_gpu_tier()

    # 모델별 GPU tier별 추천 배치 크기
    batch_size_map = {
        "kobart": {
            "HIGH": 32,
            "MEDIUM": 16,
            "LOW": 8,
            "CPU": 2
        },
        "solar": {
            "HIGH": 8,
            "MEDIUM": 4,
            "LOW": 2,
            "CPU": 1
        },
        "polyglot": {
            "HIGH": 8,
            "MEDIUM": 4,
            "LOW": 2,
            "CPU": 1
        },
        "kullm": {
            "HIGH": 8,
            "MEDIUM": 4,
            "LOW": 2,
            "CPU": 1
        }
    }

    model_type = model_type.lower()
    if model_type not in batch_size_map:
        model_type = "kobart"  # 기본값

    return batch_size_map[model_type].get(gpu_tier, 2)


def setup_mixed_precision(gpu_tier: Optional[str] = None) -> Tuple[bool, str]:
    """
    GPU tier에 따른 mixed precision 설정 추천

    Args:
        gpu_tier: GPU tier (None인 경우 자동 감지)

    Returns:
        Tuple[bool, str]: (mixed precision 사용 여부, precision type)
    """
    if gpu_tier is None:
        gpu_tier = check_gpu_tier()

    if gpu_tier == "HIGH":
        return True, "bf16"  # A100 등은 bf16 지원
    elif gpu_tier in ["MEDIUM", "LOW"]:
        return True, "fp16"  # 일반 GPU는 fp16 사용
    else:
        return False, "fp32"  # CPU는 fp32 사용


def clear_gpu_cache():
    """GPU 캐시 클리어"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> Dict:
    """
    현재 GPU 메모리 사용량 반환

    Returns:
        Dict: 메모리 사용량 정보
    """
    if not torch.cuda.is_available():
        return {"available": False}

    memory_info = {
        "available": True,
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
        "free": (torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_reserved()) / 1024**3        # GB
    }

    return memory_info


def check_multi_gpu() -> bool:
    """멀티 GPU 사용 가능 여부 확인"""
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def get_device(gpu_id: int = 0) -> torch.device:
    """
    사용할 device 반환

    Args:
        gpu_id: 사용할 GPU ID

    Returns:
        torch.device: 사용할 device
    """
    if torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu_id}')
        else:
            print(f"Warning: GPU {gpu_id} not available. Using cuda:0")
            return torch.device('cuda:0')
    else:
        return torch.device('cpu')
