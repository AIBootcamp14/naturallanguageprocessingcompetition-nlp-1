#!/usr/bin/env python3
"""
GPU 최적화 자동 배치 크기 조정 유틸리티

GPU 메모리에 따라 최적의 배치 크기를 자동으로 찾아서 설정 파일을 업데이트합니다.
팀 협업 환경에서 다양한 GPU 모델에 대한 최적화를 지원합니다.
"""

import os                                                               # 운영체제 파일 시스템 접근
import sys                                                              # 시스템 관련 기능
import torch                                                            # PyTorch 딥러닝 프레임워크
import argparse                                                         # CLI 인자 파싱
from pathlib import Path                                                # 경로 처리 라이브러리
from typing import Tuple, Optional, Dict, Any                           # 타입 힌트

# YAML 모듈 임포트 (예외 처리)
try:                                                                    # 예외 처리 시작
    import yaml                                                         # YAML 파일 처리 라이브러리
except ImportError:                                                     # 임포트 실패 시
    print("❌ PyYAML이 설치되지 않았습니다. 다음 명령어로 설치하세요:")        # 설치 안내 메시지
    print("   pip install PyYAML")                                      # 설치 명령어 출력
    sys.exit(1)                                                         # 프로그램 종료


# ==================== GPU 정보 및 권장사항 함수 ==================== #
# GPU 정보 및 권장 설정 반환 함수 정의
def get_gpu_info_and_recommendations() -> Optional[Dict[str, Any]]:
    """GPU 정보 및 권장 설정 반환"""
    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():                                   # CUDA가 사용 불가능한 경우
        return None                                                     # None 반환
    
    # GPU 정보 수집
    device = torch.cuda.current_device()                                # 현재 GPU 디바이스 번호
    props = torch.cuda.get_device_properties(device)                    # GPU 속성 정보
    device_name = props.name                                            # GPU 모델명
    total_memory = props.total_memory / (1024**3)                       # 총 메모리 (GB 단위)
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # 할당된 메모리 (GB)
    free_memory = total_memory - allocated_memory                       # 사용 가능한 메모리 (GB)
    
    # GPU별 권장 설정 (메모리 기반)
    gpu_profiles = {                                                    # GPU 프로필 딕셔너리
        # 고성능 GPU (20GB+)
        "high_end": {                                                   # 고성능 GPU 설정
            "memory_threshold": 20.0,                                   # 메모리 임계값 (GB)
            "batch_224": {"start": 64, "max": 256, "safety": 0.95},     # 224px 이미지 배치 설정 (최대 향상)
            "batch_384": {"start": 32, "max": 128, "safety": 0.95},     # 384px 이미지 배치 설정 (최대 향상)
            "batch_512": {"start": 16, "max": 64, "safety": 0.95},      # 512px 이미지 배치 설정 (최대 향상)
            "examples": ["RTX 4090", "RTX 3090", "A100", "V100"]        # 해당 GPU 예시
        },
        # 중급 GPU (10-20GB)
        "mid_range": {                                                  # 중급 GPU 설정
            "memory_threshold": 10.0,                                   # 메모리 임계값 (GB)
            "batch_224": {"start": 32, "max": 128, "safety": 0.90},     # 224px 이미지 배치 설정 (향상)
            "batch_384": {"start": 16, "max": 64, "safety": 0.90},      # 384px 이미지 배치 설정 (향상)
            "batch_512": {"start": 8, "max": 32, "safety": 0.90},       # 512px 이미지 배치 설정 (향상)
            "examples": ["RTX 3080", "RTX 3070", "RTX 2080 Ti"]         # 해당 GPU 예시
        },
        # 보급형 GPU (6-10GB)
        "budget": {                                                     # 보급형 GPU 설정
            "memory_threshold": 6.0,                                    # 메모리 임계값 (GB)
            "batch_224": {"start": 16, "max": 64, "safety": 0.90},      # 224px 이미지 배치 설정 (향상)
            "batch_384": {"start": 8, "max": 32, "safety": 0.90},       # 384px 이미지 배치 설정 (향상)
            "batch_512": {"start": 4, "max": 16, "safety": 0.90},       # 512px 이미지 배치 설정 (향상)
            "examples": ["RTX 3060", "RTX 2070", "GTX 1080 Ti"]         # 해당 GPU 예시
        },
        # 저사양 GPU (<6GB)
        "low_end": {                                                    # 저사양 GPU 설정
            "memory_threshold": 0.0,                                    # 메모리 임계값 (GB)
            "batch_224": {"start": 8, "max": 32, "safety": 0.85},       # 224px 이미지 배치 설정 (향상)
            "batch_384": {"start": 4, "max": 16, "safety": 0.85},       # 384px 이미지 배치 설정 (향상)
            "batch_512": {"start": 2, "max": 8, "safety": 0.85},        # 512px 이미지 배치 설정 (향상)
            "examples": ["GTX 1660", "GTX 1080", "RTX 2060"]            # 해당 GPU 예시
        }
    }
    
    # GPU 등급 결정 로직
    if total_memory >= 20.0:                    # 20GB 이상인 경우
        profile = gpu_profiles["high_end"]      # 고성능 프로필 선택
        tier = "고성능"                          # 등급 설정
    elif total_memory >= 10.0:                  # 10GB 이상인 경우
        profile = gpu_profiles["mid_range"]     # 중급 프로필 선택
        tier = "중급"                            # 등급 설정
    elif total_memory >= 6.0:                   # 6GB 이상인 경우
        profile = gpu_profiles["budget"]        # 보급형 프로필 선택
        tier = "보급형"                          # 등급 설정
    else:                                       # 6GB 미만인 경우
        profile = gpu_profiles["low_end"]       # 저사양 프로필 선택
        tier = "저사양"                          # 등급 설정

    # GPU 정보 딕셔너리 반환
    return {                                    # GPU 정보 딕셔너리 생성
        "name": device_name,                    # GPU 모델명
        "total_memory": total_memory,           # 총 메모리 (GB)
        "free_memory": free_memory,             # 사용 가능한 메모리 (GB)
        "tier": tier,                           # GPU 등급
        "profile": profile                      # 선택된 프로필
    }


# ==================== 배치 크기 테스트 함수 ==================== #
# 특정 배치 크기로 메모리 테스트 함수 정의
def test_batch_size(model_name: str, img_size: int, batch_size: int, device: str = "cuda") -> Tuple[bool, Optional[float]]:
    """특정 배치 크기로 메모리 테스트"""
    # 메모리 테스트 예외 처리
    try:
        # GPU 메모리 정리
        torch.cuda.empty_cache()                            # GPU 메모리 캐시 정리
        
        #------------------------ 가상 모델 생성 (실제 모델 크기 시뮬레이션) ------------------------#
        # Swin Transformer 모델인 경우
        if "swin" in model_name.lower():
            # Swin Transformer는 더 많은 메모리 사용
            dummy_model = torch.nn.Sequential(              # 가상 Swin 모델 생성
                torch.nn.Conv2d(3, 128, 7, 2, 3),           # 컨볼루션 레이어
                torch.nn.AdaptiveAvgPool2d(1),              # 적응형 평균 풀링
                torch.nn.Flatten(),                         # 평탄화 레이어
                torch.nn.Linear(128, 1000)                  # 분류 헤드
            ).to(device)                                    # GPU로 모델 이동
            memory_multiplier = 1.5                         # Swin은 더 많은 메모리 필요
            
        # EfficientNet 등 다른 모델
        else:
            dummy_model = torch.nn.Sequential(              # 가상 일반 모델 생성
                torch.nn.Conv2d(3, 64, 3, 1, 1),            # 컨볼루션 레이어
                torch.nn.AdaptiveAvgPool2d(1),              # 적응형 평균 풀링
                torch.nn.Flatten(),                         # 평탄화 레이어
                torch.nn.Linear(64, 1000)                   # 분류 헤드
            ).to(device)                                    # GPU로 모델 이동
            memory_multiplier = 1.0                         # 기본 메모리 계수
        
        # 가상 배치 데이터 생성
        dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)  # 입력 데이터 생성
        dummy_target = torch.randint(0, 17, (batch_size,)).to(device)  # 타겟 레이블 생성
        
        #-------------------------------- 테스트 실행 --------------------------------#
        # Forward pass 테스트
        with torch.cuda.amp.autocast():                                 # Mixed precision 사용
            output = dummy_model(dummy_input)                           # 순전파 실행
            loss = torch.nn.CrossEntropyLoss()(output, dummy_target)    # 손실 계산
        
        # Backward pass 테스트
        loss.backward()                                                 # 역전파 실행
        
        # 메모리 사용량 확인
        memory_used = torch.cuda.memory_allocated() / (1024**3)         # 사용된 메모리 (GB)
        
        # 메모리 정리
        del dummy_model, dummy_input, dummy_target, output, loss        # 변수 삭제
        torch.cuda.empty_cache()                                        # GPU 메모리 캐시 정리
        
        # 성공 및 메모리 사용량 반환
        return True, memory_used * memory_multiplier
        
    # CUDA 메모리 부족 예외 처리
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()                                        # 메모리 정리
        return False, None                                              # 실패 반환
    
    # 기타 예외 처리
    except Exception as e:
        print(f"⚠️ 테스트 중 오류: {e}")                                 # 에러 메시지 출력
        torch.cuda.empty_cache()                                        # 메모리 정리
        return False, None                                              # 실패 반환


# ==================== 최적 배치 크기 탐색 함수 ==================== #
# 최적의 배치 크기 찾기 함수 정의 (GPU 등급별 최적화)
def find_optimal_batch_size(model_name: str, img_size: int, gpu_info: Dict[str, Any]) -> int:
    print(f"🔍 {gpu_info['tier']} GPU 최적 배치 크기 탐색 중...")
    print(f"   GPU: {gpu_info['name']}")
    print(f"   메모리: {gpu_info['total_memory']:.1f} GB")
    print(f"   모델: {model_name}")
    print(f"   이미지 크기: {img_size}")

    #------------------------------- 이미지 크기별 배치 설정 선택 -------------------------------#
    # 이미지 크기별 배치 설정 선택
    if img_size <= 224:                                             # 224 이하인 경우
        batch_config = gpu_info['profile']['batch_224']             # 224px 배치 설정
    elif img_size <= 384:                                           # 224 초과 384 이하인 경우
        batch_config = gpu_info['profile']['batch_384']             # 384px 배치 설정
    else:                                                           # 384 초과인 경우
        batch_config = gpu_info['profile']['batch_512']             # 512px 배치 설정
    
    # 배치 설정 값 추출
    start_batch = batch_config['start']                             # 시작 배치 크기
    max_batch = batch_config['max']                                 # 최대 배치 크기
    safety_factor = batch_config['safety']                          # 안전 계수
    
    # 설정 정보 출력
    print(f"   📊 {gpu_info['tier']} GPU 권장 범위: {start_batch} ~ {max_batch}")  # 권장 범위 출력
    print(f"   🛡️ 안전 마진: {int((1-safety_factor)*100)}%") # 안전 마진 출력
    
    # 최적 배치 크기 초기화
    optimal_batch = start_batch
    
    # 이진 탐색으로 최적 배치 크기 찾기
    low, high = start_batch, max_batch                      # 탐색 범위 설정
    
    #------------------------------ 이진 탐색 반복 ------------------------------#
    # 이진 탐색 반복 (탐색 범위가 유효한 동안)
    while low <= high:
        mid = (low + high) // 2                             # 중간값 계산
        
        # 배치 크기 테스트 시작 메시지
        print(f"   배치 크기 {mid} 테스트 중...", end=" ")    # 테스트 진행 메시지 (줄바꿈 없음)
        
        # 배치 크기 테스트 실행
        success, memory_used = test_batch_size(model_name, img_size, mid)  # 메모리 테스트 실행
        
        # 테스트 성공 시 처리
        if success:
            optimal_batch = mid                             # 최적 배치 크기 업데이트
            
            # 메모리 사용량 정보 출력 (메모리 사용량 정보가 있는 경우)
            if memory_used:
                print(f"✅ (메모리: {memory_used:.2f} GB)")  # 성공 메시지와 메모리 사용량 출력
            # 메모리 사용량 정보가 없는 경우
            else:
                print("✅")                                 # 성공 메시지만 출력
            low = mid + 1                                   # 더 큰 배치 시도를 위해 하한값 증가
            
        # 테스트 실패 시 처리
        else:                                               # 메모리 테스트 실패 시
            print("❌ (메모리 부족)")                        # 실패 메시지 출력
            high = mid - 1                                  # 더 작은 배치로 시도를 위해 상한값 감소
    
    #------------------------------ 최종 배치 크기 계산 ------------------------------#
    # 안전 마진 적용
    final_batch = max(4, int(optimal_batch * safety_factor))  # 안전 계수 적용 (최소 4)
    
    # 4의 배수로 조정 (모든 GPU에서 효율적)
    final_batch = (final_batch // 4) * 4                    # 4의 배수로 반올림
    final_batch = max(4, final_batch)                       # 최소 4 보장
    
    # 최적 배치 크기 출력
    print(f"\n🎯 {gpu_info['tier']} GPU 최적 배치 크기: {final_batch}")  # 최종 결과 출력
    
    #------------------------------ GPU별 권장사항 출력 ------------------------------#
    # GPU별 추가 권장사항 생성
    recommendations = []                # 권장사항 리스트 초기화
    
    # 낮은 메모리 GPU 권장사항
    if gpu_info['total_memory'] < 8:    # 8GB 미만인 경우
        recommendations.append("💡 낮은 GPU 메모리: gradient_accumulation_steps 사용 권장")  # 그래디언트 누적 권장
    # 구형 GPU 권장사항
    if "GTX" in gpu_info['name']:       # GTX 시리즈인 경우
        recommendations.append("💡 구형 GPU: mixed precision (AMP) 비활성화 권장")          # AMP 비활성화 권장
    # 고성능 GPU 권장사항
    if gpu_info['total_memory'] >= 20:  # 20GB 이상인 경우
        recommendations.append("💡 고성능 GPU: 더 큰 모델이나 더 높은 해상도 고려 가능")        # 고급 옵션 권장
    
    # 권장사항 출력
    for rec in recommendations:
        print(f"   {rec}")      # 권장사항 출력
    
    # 최종 배치 크기 반환
    return final_batch


# ==================== 설정 파일 업데이트 함수 ==================== #
# 설정 파일의 배치 크기 업데이트 함수 정의
def update_config_file(config_path: str, batch_size: int):
    # 설정 파일 읽기
    with open(config_path, 'r', encoding='utf-8') as f:     # 설정 파일 열기
        config = yaml.safe_load(f)                          # YAML 파일 파싱
    
    # 배치 크기 업데이트
    # train 섹션이 있는 경우 우선 업데이트 (현재 프로젝트 표준)
    if 'train' in config:
        config['train']['batch_size'] = batch_size          # batch_size 값 업데이트
    # training 섹션이 있는 경우 업데이트 (레거시 지원)
    elif 'training' in config:
        config['training']['batch_size'] = batch_size       # batch_size 값 업데이트
    # 둘 다 없는 경우 train 섹션 생성
    else:
        config['train'] = {'batch_size': batch_size}        # train 섹션 생성
    
    # 백업 파일 생성
    backup_path = config_path + '.backup'                   # 백업 파일 경로 생성
    
    # 백업 파일 저장
    with open(backup_path, 'w', encoding='utf-8') as f:     # 백업 파일 열기
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)  # YAML 형식으로 저장
    
    # 원본 파일 업데이트
    with open(config_path, 'w', encoding='utf-8') as f:     # 원본 파일 열기
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)  # 업데이트된 설정 저장
    
    # 업데이트 완료 메시지 출력
    print(f"✅ 설정 파일 업데이트: {config_path}")          # 업데이트 완료 메시지
    print(f"📄 백업 파일 생성: {backup_path}")             # 백업 파일 생성 메시지


# ==================== 메인 함수 ==================== #
# 메인 함수 정의
def main():
    # CLI 인자 파서 생성
    parser = argparse.ArgumentParser(description="RTX 4090 최적화 자동 배치 크기 조정")
    
    # CLI 인자 정의
    parser.add_argument("--config", required=True, help="설정 파일 경로")                           # 필수 설정 파일 인자
    parser.add_argument("--test-only", action="store_true", help="테스트만 하고 파일 업데이트 안함")  # 테스트 전용 플래그
    parser.add_argument("--model", type=str, help="모델 이름 (자동 감지 안될 때)")                   # 모델명 수동 지정
    parser.add_argument("--img-size", type=int, help="이미지 크기 (자동 감지 안될 때)")              # 이미지 크기 수동 지정

    # CLI 인자 파싱
    args = parser.parse_args()                                  # 인자 파싱 실행
    
    # 설정 파일 존재 여부 확인 (설정 파일이 존재하지 않는 경우)
    if not os.path.exists(args.config):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")   # 에러 메시지 출력
        sys.exit(1)                                             # 프로그램 종료
    
    # GPU 확인 (CUDA 사용 불가능한 경우)
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다!")                    # 에러 메시지 출력
        sys.exit(1)                                             # 프로그램 종료
    
    # GPU 정보 출력
    device_name = torch.cuda.get_device_name()                 # GPU 모델명 가져오기
    gpu_info = get_gpu_info_and_recommendations()              # GPU 정보 및 권장사항 가져오기
    
    # GPU 정보 유효성 확인 (GPU 정보를 가져올 수 없는 경우)
    if gpu_info is None:
        print("❌ GPU 정보를 가져올 수 없습니다!")                # 에러 메시지 출력
        sys.exit(1)                                            # 프로그램 종료
    
    # GPU 정보 출력
    print("🚀 팀 협업용 GPU 최적화 배치 크기 조정기")               # 프로그램 제목
    print("=" * 55)                                             # 구분선 출력
    print(f"🔧 GPU: {device_name}")                            # GPU 모델명 출력
    print(f"💾 총 메모리: {gpu_info['total_memory']:.1f} GB")   # 총 메모리 출력
    print(f"🏆 GPU 등급: {gpu_info['tier']}")                  # GPU 등급 출력
    print(f"💡 권장 배치 범위: {gpu_info['profile']['batch_224']['start']} ~ {gpu_info['profile']['batch_224']['max']}")  # 권장 배치 범위 출력
    
    # 설정 파일 로드
    with open(args.config, 'r', encoding='utf-8') as f:    # 설정 파일 열기
        config = yaml.safe_load(f)                          # YAML 파일 파싱
    
    # 모델 및 이미지 크기 추출
    model_name = args.model or config.get('model', {}).get('name', 'swin_base_patch4_window7_224')  # 모델명 추출
    
    # CLI 인자에서 이미지 크기 찾기 (여러 경로 시도)
    img_size = args.img_size
    
    # CLI 인자에 이미지 크기가 없는 경우 설정 파일에서 찾기 (이미지 크기가 지정되지 않은 경우)
    if not img_size:
        # 설정 파일의 여러 위치에서 이미지 크기 탐색
        img_size = (config.get('model', {}).get('img_size') or     # model.img_size에서 찾기
                   config.get('train', {}).get('img_size') or      # train.img_size에서 찾기
                   config.get('training', {}).get('img_size') or   # training.img_size에서 찾기
                   384)                                            # 기본값 384
    
    # 모델 및 이미지 정보 출력
    print(f"📊 모델: {model_name}")                                 # 사용할 모델명 출력
    print(f"📏 이미지 크기: {img_size}")                             # 사용할 이미지 크기 출력

    # RTX 4090 특별 최적화
    if "RTX 4090" in device_name:                                   # RTX 4090인 경우
        print("\n🎯 RTX 4090 감지! 고성능 최적화 모드")                # 특별 최적화 모드 안내
        # 이미지 크기별 권장 배치 크기 설정
        if img_size <= 224:                                         # 224px 이하인 경우
            recommended_batch = 96                                  # RTX 4090에서 224px는 큰 배치 가능
        elif img_size <= 384:                                       # 384px 이하인 경우
            recommended_batch = 48                                  # 384px에서도 충분한 크기
        else:                                                       # 512px 이상인 경우
            recommended_batch = 24                                  # 512px 이상에서는 보수적으로

        print(f"💡 RTX 4090 권장 시작 배치: {recommended_batch}")    # 권장 배치 크기 출력
    
    # 최적 배치 크기 찾기
    optimal_batch = find_optimal_batch_size(model_name, img_size, gpu_info)  # 최적 배치 크기 탐색 실행
    
    # 최종 결과 출력
    print("\n" + "=" * 50)                                              # 결과 구분선
    print(f"🎉 최종 결과:")                                              # 최종 결과 제목
    print(f"   최적 배치 크기: {optimal_batch}")                          # 최적 배치 크기 출력
    print(f"   예상 메모리 절약: ~{((96/optimal_batch)-1)*100:.0f}%")     # 메모리 절약 예상치
    print(f"   예상 훈련 속도: {optimal_batch/32:.1f}x 기준")             # 훈련 속도 예상치
    
    # 설정 파일 업데이트 여부 확인 (테스트 전용 모드가 아닌 경우)
    if not args.test_only:
        # 설정 파일 업데이트
        update_config_file(args.config, optimal_batch)                  # 설정 파일에 최적 배치 크기 적용
        
        # 완료 안내 메시지
        print(f"\n✅ 완료! 이제 다음 명령어로 최적화된 훈련을 시작하세요:")    # 완료 메시지
        print(f"   python src/training/train_main.py --config configs/train_highperf.yaml --mode highperf")  # 실행 명령어 안내
    # 테스트 전용 모드인 경우
    else:
        # 테스트 모드 안내
        print(f"\n💡 테스트 모드: 설정 파일이 업데이트되지 않았습니다.")      # 테스트 모드 안내
        print(f"   수동으로 batch_size를 {optimal_batch}로 설정하세요.")   # 수동 설정 안내


# ==================== 스크립트 실행 진입점 ==================== #
# 스크립트 직접 실행 시 메인 함수 호출
if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()                  # 메인 함수 실행
