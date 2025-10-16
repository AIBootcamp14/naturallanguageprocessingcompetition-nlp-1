"""
팀 협업용 GPU 최적화 자동 배치 크기 찾기 도구
다양한 GPU 환경에서 최적의 배치 크기를 자동으로 찾아주는 도구

Author: AI Team
Date: 2025-01-05
"""

import os                                                       # 운영체제 파일 시스템 접근
import sys                                                      # 시스템 관련 기능
import torch                                                    # PyTorch 딥러닝 프레임워크
import gc                                                       # 가비지 컬렉터 메모리 관리
import argparse                                                 # CLI 인자 파싱
from pathlib import Path                                        # 경로 처리 라이브러리
from typing import Tuple, Optional, Dict, Any                   # 타입 힌트

# YAML 모듈 임포트 (예외 처리)
try:
    import yaml                                                 # YAML 파일 처리 라이브러리
# 임포트 실패 시 안내 메시지 출력 후 종료
except ImportError:
    print("❌ PyYAML이 설치되지 않았습니다. 다음 명령어로 설치하세요:")  # 설치 안내 메시지
    print("   pip install PyYAML")                              # 설치 명령어 출력
    sys.exit(1)                                                 # 프로그램 종료


# ==================== GPU 정보 및 권장사항 함수 ==================== #
# GPU 정보 확인 및 권장 설정 반환 함수
def get_gpu_info_and_recommendations() -> Dict[str, Any]:
    # CUDA 사용 가능 여부 확인 (CUDA가 사용 불가능한 경우)
    if not torch.cuda.is_available():
        return {                                                        # CPU 모드 설정 반환
            'name': 'CPU',                                              # 디바이스 이름
            'total_memory': 0,                                          # 메모리 용량 (CPU는 0)
            'tier': 'cpu',                                              # 등급: CPU
            'profile': {                                                # 배치 크기 프로필
                'batch_224': {'start': 4, 'max': 8, 'safety': 0.9},     # 224px 이미지 설정
                'batch_384': {'start': 2, 'max': 4, 'safety': 0.9},     # 384px 이미지 설정
                'batch_512': {'start': 1, 'max': 2, 'safety': 0.9}      # 512px 이미지 설정
            }
        }
    
    # GPU 정보 수집
    device_name = torch.cuda.get_device_name()                                      # GPU 디바이스 이름 조회
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)     # GPU 메모리 용량 (GB)
    
    #---------------- GPU 등급 및 권장 배치 크기 프로필 설정 ----------------#
    # 하이엔드 GPU 확인
    if any(gpu in device_name for gpu in ['RTX 4090', 'RTX 4080', 'RTX 3090', 'A100', 'V100']):
        tier = 'high_end'                                               # 등급: 하이엔드
        profile = {                                                     # 하이엔드 GPU 배치 설정
            'batch_224': {'start': 64, 'max': 256, 'safety': 0.95},    # 224px: 시작 64, 최대 256 (더 공격적)
            'batch_384': {'start': 32, 'max': 128, 'safety': 0.95},    # 384px: 시작 32, 최대 128 (기존 64→128)
            'batch_512': {'start': 16, 'max': 64, 'safety': 0.95}      # 512px: 시작 16, 최대 64 (기존 32→64)
        }
    # 미드레인지 GPU 확인
    elif any(gpu in device_name for gpu in ['RTX 3080', 'RTX 3070', 'RTX 4070']):
        tier = 'mid_range'                                              # 등급: 미드레인지
        profile = {                                                     # 미드레인지 GPU 배치 설정
            'batch_224': {'start': 32, 'max': 128, 'safety': 0.9},     # 224px: 시작 32, 최대 128 (기존 64→128)
            'batch_384': {'start': 16, 'max': 64, 'safety': 0.9},      # 384px: 시작 16, 최대 64 (기존 32→64)
            'batch_512': {'start': 8, 'max': 32, 'safety': 0.9}        # 512px: 시작 8, 최대 32 (기존 16→32)
        }
    # 보급형 GPU 확인
    elif any(gpu in device_name for gpu in ['RTX 3060', 'RTX 2070', 'RTX 2080']):
        tier = 'budget'                                                 # 등급: 보급형
        profile = {                                                     # 보급형 GPU 배치 설정
            'batch_224': {'start': 16, 'max': 64, 'safety': 0.9},      # 224px: 시작 16, 최대 64 (기존 32→64)
            'batch_384': {'start': 8, 'max': 32, 'safety': 0.9},       # 384px: 시작 8, 최대 32 (기존 16→32)
            'batch_512': {'start': 4, 'max': 16, 'safety': 0.9}        # 512px: 시작 4, 최대 16 (기존 8→16)
        }
    # GTX 1660, GTX 1080 등 구형 GPU
    else:
        tier = 'low_end'                                                # 등급: 로우엔드
        profile = {                                                     # 로우엔드 GPU 배치 설정
            'batch_224': {'start': 8, 'max': 16, 'safety': 0.9},       # 224px: 시작 8, 최대 16
            'batch_384': {'start': 4, 'max': 8, 'safety': 0.9},        # 384px: 시작 4, 최대 8
            'batch_512': {'start': 2, 'max': 4, 'safety': 0.9}         # 512px: 시작 2, 최대 4
        }
    
    # GPU 정보 및 설정 반환
    return {
        'name': device_name,                                            # GPU 디바이스 이름
        'total_memory': total_memory,                                   # GPU 메모리 용량
        'tier': tier,                                                   # GPU 등급
        'profile': profile                                              # 배치 크기 프로필
    }


# ==================== 배치 크기 테스트 함수 ==================== #
# 특정 배치 크기로 메모리 테스트 함수
def test_batch_size(model_name: str, img_size: int, batch_size: int) -> Tuple[bool, Optional[float]]:
    # 예외 처리 시작
    try:
        # GPU 메모리 정리
        torch.cuda.empty_cache()                                # CUDA 캐시 메모리 정리
        gc.collect()                                            # 가비지 컬렉션 실행
        
        device = torch.device('cuda')                           # CUDA 디바이스 설정
        
        # 간단한 모델로 테스트 (Swin Transformer 모델인 경우)
        if 'swin' in model_name.lower():                    
            # Swin Transformer 근사 모델
            model = torch.nn.Sequential(                        # 순차적 모델 생성
                torch.nn.Conv2d(3, 128, 3, padding=1),          # 컨볼루션 레이어 (3→128 채널)
                torch.nn.BatchNorm2d(128),                      # 배치 정규화
                torch.nn.ReLU(),                                # ReLU 활성화 함수
                torch.nn.AdaptiveAvgPool2d((7, 7)),             # 적응형 평균 풀링
                torch.nn.Flatten(),                             # 평탄화
                torch.nn.Linear(128 * 7 * 7, 1000),             # 완전연결층 1
                torch.nn.Linear(1000, 100)                      # 완전연결층 2 (출력)
            ).to(device)                                        # GPU로 모델 이동
        # ConvNext 모델인 경우
        elif 'convnext' in model_name.lower():
            # ConvNext 근사 모델
            model = torch.nn.Sequential(                        # 순차적 모델 생성
                torch.nn.Conv2d(3, 96, 4, stride=4),            # 컨볼루션 레이어 (stride=4)
                torch.nn.LayerNorm([96, img_size//4, img_size//4]),  # 레이어 정규화
                torch.nn.Conv2d(96, 192, 1),                    # 1x1 컨볼루션
                torch.nn.AdaptiveAvgPool2d((1, 1)),             # 글로벌 평균 풀링
                torch.nn.Flatten(),                             # 평탄화
                torch.nn.Linear(192, 100)                       # 완전연결층 (출력)
            ).to(device)                                        # GPU로 모델 이동
        # 기본 ResNet 스타일 모델
        else:
            # 기본 ResNet 스타일 모델
            model = torch.nn.Sequential(                        # 순차적 모델 생성
                torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),     # 초기 컨볼루션
                torch.nn.BatchNorm2d(64),                       # 배치 정규화
                torch.nn.ReLU(),                                # ReLU 활성화 함수
                torch.nn.MaxPool2d(3, stride=2, padding=1),     # 최대 풀링
                torch.nn.AdaptiveAvgPool2d((1, 1)),             # 글로벌 평균 풀링
                torch.nn.Flatten(),                             # 평탄화
                torch.nn.Linear(64, 100)                        # 완전연결층 (출력)
            ).to(device)                                        # GPU로 모델 이동
        
        # 테스트 데이터 생성
        test_input = torch.randn(batch_size, 3, img_size, img_size, device=device)  # 랜덤 입력 데이터
        test_target = torch.randint(0, 100, (batch_size,), device=device)           # 랜덤 타겟 레이블
        
        # Forward pass
        output = model(test_input)                                      # 순전파 실행
        loss = torch.nn.functional.cross_entropy(output, test_target)   # 교차엔트로피 손실 계산
        
        # 역전파 실행
        loss.backward()                                     
        
        # 메모리 사용량 측정
        memory_used = torch.cuda.memory_allocated() / (1024**3)  # GPU 메모리 사용량 (GB)
        
        # 정리
        del model, test_input, test_target, output, loss        # 메모리 해제
        torch.cuda.empty_cache()                                # CUDA 캐시 정리
        gc.collect()                                            # 가비지 컬렉션
        
        return True, memory_used                                # 성공 및 메모리 사용량 반환
    
    # 런타임 에러 처리    
    except RuntimeError as e:
        # 메모리 부족 에러이거나 CUDA 에러인 경우
        if "out of memory" in str(e) or "CUDA" in str(e):
            try:
                torch.cuda.empty_cache()                        # CUDA 캐시 정리
                gc.collect()                                    # 가비지 컬렉션
            except:
                pass                                            # 캐시 정리 실패해도 계속 진행
            return False, None                                  # 실패 반환
        # 다른 런타임 에러인 경우
        else:
            try:
                torch.cuda.empty_cache()                        # CUDA 캐시 정리
                gc.collect()                                    # 가비지 컬렉션
            except:
                pass
            return False, None                                  # 안전하게 실패 반환
    # 기타 예외 처리
    except Exception as e:
        try:
            torch.cuda.empty_cache()                            # CUDA 캐시 정리
            gc.collect()                                        # 가비지 컬렉션
        except:
            pass
        return False, None                                      # 실패 반환


# ==================== 최적 배치 크기 탐색 함수 ==================== #
def find_optimal_batch_size(model_name: str, img_size: int, gpu_info: Dict[str, Any]) -> int:
    """최적의 배치 크기 찾기 (GPU 등급별 최적화)"""
    print(f"🔍 {gpu_info['tier']} GPU 최적 배치 크기 탐색 중...")  # 탐색 시작 안내
    print(f"   GPU: {gpu_info['name']}")                        # GPU 이름 출력
    print(f"   메모리: {gpu_info['total_memory']:.1f} GB")       # GPU 메모리 용량 출력
    print(f"   모델: {model_name}")                              # 모델 이름 출력
    print(f"   이미지 크기: {img_size}")                          # 이미지 크기 출력
    
    # 이미지 크기별 프로필 선택
    if img_size <= 224:                                         # 224px 이하인 경우
        batch_config = gpu_info['profile']['batch_224']         # 224px 프로필 선택
    elif img_size <= 384:                                       # 384px 이하인 경우
        batch_config = gpu_info['profile']['batch_384']         # 384px 프로필 선택
    else:                                                       # 512px 이상인 경우
        batch_config = gpu_info['profile']['batch_512']             # 512px 프로필 선택
    
    start_batch = batch_config['start']                         # 시작 배치 크기
    max_batch = batch_config['max']                             # 최대 배치 크기
    safety_factor = batch_config['safety']                      # 안전 마진 계수
    
    print(f"   📊 {gpu_info['tier']} GPU 권장 범위: {start_batch} ~ {max_batch}")  # 권장 범위 출력
    print(f"   🛡️ 안전 마진: {int((1-safety_factor)*100)}%")    # 안전 마진 퍼센트 출력
    
    optimal_batch = start_batch                                 # 최적 배치 크기 초기값
    
    # 이진 탐색으로 최적 배치 크기 찾기
    low, high = start_batch, max_batch                          # 탐색 범위 설정
    
    # 이진 탐색 루프
    while low <= high:
        mid = (low + high) // 2                                 # 중간값 계산
        
        print(f"   배치 크기 {mid} 테스트 중...", end=" ")         # 테스트 중 메시지
        
        # 배치 크기 테스트
        success, memory_used = test_batch_size(model_name, img_size, mid)
        
        # 테스트 성공 시
        if success:
            optimal_batch = mid     # 최적 배치 크기 업데이트

            # 메모리 사용량 정보가 있는 경우
            if memory_used:
                print(f"✅ (메모리: {memory_used:.2f} GB)") # 성공 및 메모리 사용량 출력
            # 메모리 사용량 정보가 없는 경우
            else:
                print("✅")                 # 성공 메시지만 출력
            low = mid + 1                   # 더 큰 배치 크기 시도

        # 테스트 실패 시
        else:
            print("❌ (메모리 부족)")        # 실패 메시지 출력
            high = mid - 1                  # 더 작은 배치 크기로 시도
    
    # 안전 마진 적용 - 고성능 GPU는 더 공격적으로, 저성능 GPU는 보수적으로
    if gpu_info['tier'] == 'high_end':
        # 고성능 GPU: 안전 마진 최소화 (5%만 감소)
        final_batch = max(4, int(optimal_batch * safety_factor))
    elif gpu_info['tier'] == 'mid_range':
        # 중급 GPU: 적당한 안전 마진 (10% 감소)
        final_batch = max(4, int(optimal_batch * safety_factor))
    else:
        # 저성능 GPU: 보수적 안전 마진 (15% 감소)
        final_batch = max(4, int(optimal_batch * 0.85))
    
    # 4의 배수로 조정 (모든 GPU에서 효율적)
    final_batch = (final_batch // 4) * 4    # 4의 배수로 조정
    final_batch = max(4, final_batch)       # 최소값 4 보장
    
    # 최종 결과 출력
    print(f"\n🎯 {gpu_info['tier']} GPU 최적 배치 크기: {final_batch}")
    
    # GPU별 추가 권장사항
    recommendations = []                # 권장사항 리스트 초기화
    
    # 메모리가 8GB 미만인 경우
    if gpu_info['total_memory'] < 8:
        # 그래디언트 누적 권장
        recommendations.append("💡 낮은 GPU 메모리: gradient_accumulation_steps 사용 권장")
    # GTX 시리즈 GPU인 경우
    if "GTX" in gpu_info['name']:
        # AMP 비활성화 권장
        recommendations.append("💡 구형 GPU: mixed precision (AMP) 비활성화 권장")
    # 메모리가 20GB 이상인 경우
    if gpu_info['total_memory'] >= 20:
        # 고성능 활용 권장
        recommendations.append("💡 고성능 GPU: 더 큰 모델이나 더 높은 해상도 고려 가능")
    
    # 권장사항 순회
    for rec in recommendations:
        print(f"   {rec}")  # 권장사항 출력
    
    # 최종 배치 크기 반환
    return final_batch


# ==================== 설정 파일 업데이트 함수 ==================== #
def update_config_file(config_path: str, batch_size: int):
    """설정 파일의 배치 크기 업데이트 (백업 포함)"""
    try:                                                    # 예외 처리 시작
        # 백업 파일 생성
        import shutil
        from datetime import datetime
        import os
        
        # 백업 파일 경로 생성 (타임스탬프 포함)
        backup_path = f"{config_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 원본 파일 백업
        shutil.copy2(config_path, backup_path)
        print(f"📂 백업 파일 생성: {os.path.basename(backup_path)}")
        
        with open(config_path, 'r', encoding='utf-8') as f: # 설정 파일 읽기
            config = yaml.safe_load(f)                      # YAML 파일 로드
        
        if 'train' not in config:                          # train 섹션이 없는 경우
            config['train'] = {}                           # train 섹션 생성
        
        config['train']['batch_size'] = batch_size         # 배치 크기 설정
        
        # training 섹션도 확인하여 동일하게 업데이트
        if 'training' in config:                           # training 섹션이 있는 경우
            config['training']['batch_size'] = batch_size  # training 배치 크기도 동일하게 설정
            print(f"📝 training.batch_size도 {batch_size}로 동기화")
        
        with open(config_path, 'w', encoding='utf-8') as f: # 설정 파일 쓰기
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)  # YAML 파일 저장
        
        print(f"✅ 설정 파일 업데이트 완료: batch_size = {batch_size}")  # 업데이트 완료 메시지
        
    # 예외 발생 시
    except Exception as e:
        print(f"❌ 설정 파일 업데이트 실패: {e}")              # 에러 메시지 출력


# ==================== 메인 함수 ==================== #
def main():
    """메인 실행 함수"""
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description='팀 협업용 GPU 최적화 자동 배치 크기 찾기 도구')       # 인자 파서 생성
    parser.add_argument('--config', required=True, help='YAML 설정 파일 경로')                      # 필수: 설정 파일 경로
    parser.add_argument('--model', help='모델 이름 (설정 파일에서 자동 추출)')                         # 선택: 모델 이름
    parser.add_argument('--img-size', type=int, help='이미지 크기 (설정 파일에서 자동 추출)')          # 선택: 이미지 크기
    parser.add_argument('--test-only', action='store_true', help='테스트만 수행 (설정 파일 수정 안함)')# 선택: 테스트 전용 모드
    
    args = parser.parse_args()                                  # 인자 파싱 실행
    
    print("🚀 팀 협업용 GPU 최적화 자동 배치 크기 찾기 도구")        # 프로그램 제목
    print("=" * 55)                                             # 구분선 출력
    
    # 설정 파일 존재 확인
    if not os.path.exists(args.config):                         # 설정 파일이 존재하지 않는 경우
        print(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")   # 에러 메시지 출력
        sys.exit(1)                                             # 프로그램 종료
    
    # GPU 확인 (CUDA가 사용 불가능한 경우)
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다!")     # 에러 메시지 출력
        sys.exit(1)                             # 프로그램 종료
    
    # GPU 정보 및 권장사항 가져오기
    gpu_info = get_gpu_info_and_recommendations()           # GPU 정보 조회
    
    print(f"🔧 GPU: {gpu_info['name']}")                        # GPU 이름 출력
    print(f"💾 GPU 메모리: {gpu_info['total_memory']:.1f} GB")   # GPU 메모리 용량 출력
    print(f"🏆 GPU 등급: {gpu_info['tier']}")                    # GPU 등급 출력
    
    # 설정 파일 로드
    with open(args.config, 'r', encoding='utf-8') as f:     # 설정 파일 열기
        config = yaml.safe_load(f)                          # YAML 파일 로드
    
    # 모델 및 이미지 크기 추출
    model_name = args.model or config.get('model', {}).get('name', 'swin_base_patch4_window7_224')  # 모델 이름 추출
    
    # 이미지 크기 찾기 (여러 경로 시도)
    img_size = args.img_size                                # CLI 인자에서 이미지 크기 조회
    
    # CLI 인자에 없는 경우
    if not img_size:
        img_size = (config.get('model', {}).get('img_size') or      # 설정 파일의 여러 경로에서 시도
                   config.get('train', {}).get('img_size') or       # train 섹션
                   config.get('training', {}).get('img_size') or    # training 섹션
                   config.get('data', {}).get('img_size') or        # data 섹션
                   384)                                             # 기본값 384
    
    print(f"📊 모델: {model_name}")         # 모델 이름 출력
    print(f"📏 이미지 크기: {img_size}")     # 이미지 크기 출력
    
    # 실제 사용된 배치 설정 가져오기
    actual_profile_key = f"batch_{img_size}"  # 실제 사용된 이미지 크기에 맞는 프로필 키
    if actual_profile_key in gpu_info['profile']:
        actual_batch_range = gpu_info['profile'][actual_profile_key]
        print(f"💡 권장 배치 범위: {actual_batch_range['start']} ~ {actual_batch_range['max']}")
    else:
        # 기본값으로 224 사용 (호환성)
        actual_batch_range = gpu_info['profile']['batch_224']
        print(f"💡 권장 배치 범위: {actual_batch_range['start']} ~ {actual_batch_range['max']} (기본값)")
    
    # 최적 배치 크기 찾기
    optimal_batch = find_optimal_batch_size(model_name, img_size, gpu_info)  # 최적 배치 크기 탐색
    
    print("\n" + "=" * 55)                                                        # 구분선 출력
    print(f"🎉 최종 결과:")                                                        # 최종 결과 제목
    print(f"   최적 배치 크기: {optimal_batch}")                                    # 최적 배치 크기 출력
    print(f"   GPU 등급: {gpu_info['tier']}")                                      # GPU 등급 출력
    print(f"   예상 메모리 사용률: ~{(optimal_batch/actual_batch_range['max'])*100:.0f}%")  # 메모리 사용률 출력
    
    # 테스트 전용 모드가 아닌 경우
    if not args.test_only:
        # 설정 파일 업데이트
        update_config_file(args.config, optimal_batch)                          # 설정 파일에 배치 크기 업데이트
        
        print(f"\n✅ 완료! 이제 다음 명령어로 최적화된 훈련을 시작하세요:")           # 완료 메시지
        print(f"   python src/training/train_main.py --config configs/train_highperf.yaml --mode highperf")  # 실행 명령어 안내
        
        # GPU별 추가 권장사항
        print(f"\n💡 {gpu_info['tier']} GPU 추가 권장사항:")  # 추가 권장사항 제목
        
        # 메모리가 8GB 미만인 경우
        if gpu_info['total_memory'] < 8:
            print(f"   - gradient_accumulation_steps = 2-4 사용 권장 (낮은 메모리)")  # 그래디언트 누적 권장
            print(f"   - mixed precision 비활성화 고려")                             # AMP 비활성화 권장
        # 메모리가 20GB 이상인 경우
        elif gpu_info['total_memory'] >= 20:
            print(f"   - 더 큰 모델이나 ensemble 고려 가능")                # 큰 모델 권장
            print(f"   - Multi-GPU training 가능")                      # 멀티 GPU 권장
        
        print(f"   - 실제 훈련 시작 전에 작은 epoch로 테스트해보세요")        # 테스트 권장
        print(f"   - 모니터링: nvidia-smi -l 1 명령어로 GPU 사용량 확인")   # 모니터링 권장
        print(f"   - 배치 크기 설정 공유하여 일관성 유지")             # 팀 협업 권장
    
    # 테스트 전용 모드인 경우
    else:
        print(f"\n💡 테스트 모드: 설정 파일이 업데이트되지 않았습니다.")    # 테스트 모드 안내
        print(f"   수동으로 batch_size를 {optimal_batch}로 설정하세요.")  # 수동 설정 안내
    
    print(f"\n✨ {gpu_info['tier']} GPU 최적화 완료!")                  # 최종 완료 메시지
    print(f"🤝 다른 설정을 공유하여 협업하세요!")                  # 협업 권장 메시지


# ==================== 스크립트 실행 진입점 ==================== #
if __name__ == "__main__":  # 스크립트 직접 실행 시
    main()                  # 메인 함수 실행
