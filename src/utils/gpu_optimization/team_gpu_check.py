#!/usr/bin/env python3                              # Python3 실행 환경 지정
# -*- coding: utf-8 -*-                             # UTF-8 인코딩 지정
"""
GPU 호환성 빠른 체크 도구
Quick GPU compatibility check for team members
"""

# ------------------------- 라이브러리 Import ------------------------- #
import torch                                        # PyTorch GPU 관련 함수
import sys                                          # 시스템 종료 함수

# ---------------------- GPU 호환성 체크 함수 ---------------------- #
def check_gpu_compatibility():
    print("🔍 팀 GPU 호환성 체크")                  # 체크 시작 메시지
    print("=" * 40)                                # 구분선 출력
    
    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():             # CUDA 사용 불가능한 경우
        print("❌ CUDA가 사용 불가능합니다")        # 오류 메시지 출력
        print("💡 해결책:")                       # 해결 방법 제목
        print("   - NVIDIA 드라이버 설치 확인")     # 드라이버 체크 안내
        print("   - CUDA 설치 확인")               # CUDA 설치 체크 안내
        print("   - PyTorch CUDA 버전 확인")       # PyTorch 버전 체크 안내
        return False                              # False 반환하고 종료
    
    # GPU 정보 수집 및 출력
    device_count = torch.cuda.device_count()     # 사용 가능한 GPU 개수
    print(f"✅ CUDA 사용 가능")                   # CUDA 사용 가능 메시지
    print(f"🔧 GPU 개수: {device_count}")        # GPU 개수 출력
    
    #---------------- 각 GPU별 상세 정보 출력 ----------------#
    # GPU 개수만큼 반복
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)                                 # GPU 이름 가져오기
        memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)    # 메모리 GB 계산
        
        print(f"\n📊 GPU {i}: {device_name}")       # GPU 번호와 이름 출력
        print(f"💾 메모리: {memory_gb:.1f} GB")      # 메모리 크기 출력
        
        # GPU 성능 등급 분류 및 권장사항 제공
        # 고급 GPU
        if any(gpu in device_name for gpu in ['RTX 4090', 'RTX 4080', 'RTX 3090', 'A100', 'V100']):
            tier = "🏆 HIGH-END"                                                        # 고급 등급 설정
            batch_rec = "96-256 (224px), 80-128 (384px)"                                # 배치 크기 권장사항 (최적화됨)
            note = "최고 성능! Multi-GPU 훈련 가능, 95% 안전 계수 적용"                     # 성능 메모
        # 중급 GPU
        elif any(gpu in device_name for gpu in ['RTX 3080', 'RTX 3070', 'RTX 4070']):
            tier = "🥈 MID-RANGE"                                                       # 중급 등급 설정
            batch_rec = "64-128 (224px), 32-64 (384px)"                                 # 배치 크기 권장사항 (최적화됨)
            note = "우수한 성능! gradient_accumulation_steps=2 권장, 90% 안전 계수"        # 성능 메모
        # 보급형 GPU
        elif any(gpu in device_name for gpu in ['RTX 3060', 'RTX 2070', 'RTX 2080']):
            tier = "🥉 BUDGET"                                                          # 보급형 등급 설정
            batch_rec = "32-64 (224px), 16-32 (384px)"                                  # 배치 크기 권장사항 (최적화됨)
            note = "적절한 성능! gradient_accumulation_steps=3-4 권장, 90% 안전 계수"       # 성능 메모
        # 기타 GPU (저사양)
        else:
            tier = "⚠️ LOW-END"                                                         # 저사양 등급 설정
            batch_rec = "16-32 (224px), 8-16 (384px)"                                   # 배치 크기 권장사항 (최적화됨)
            note = "주의! mixed precision 비활성화, gradient_accumulation_steps=6-8 권장"  # 성능 메모
        
        print(f"🏷️ 등급: {tier}")               # GPU 등급 출력
        print(f"📏 권장 배치: {batch_rec}")      # 권장 배치 크기 출력
        print(f"💡 팁: {note}")                 # 사용 팁 출력
    
    # 사용자를 위한 권장 명령어 안내
    print(f"\n🚀 다음 단계:")
    print(f"   1. 자동 배치 크기 최적화:")                                                          # 1단계 안내
    print(f"      python src/utils/gpu_optimization/auto_batch_size.py --config configs/train.yaml --test-only")  # 테스트 명령어
    print(f"   2. 설정 파일 업데이트:")                                                             # 2단계 안내  
    print(f"      python src/utils/gpu_optimization/auto_batch_size.py --config configs/train.yaml")              # 업데이트 명령어
    print(f"   3. 훈련 시작:")                                                                     # 3단계 안내
    print(f"      python src/training/train_main.py --config configs/train_highperf.yaml --mode highperf")  # 훈련 시작 명령어
    
    # PyTorch 환경 정보 출력
    print(f"\n🐍 PyTorch 정보:")                                            # PyTorch 정보 제목
    print(f"   버전: {torch.__version__}")                                  # PyTorch 버전 출력
    print(f"   CUDA 지원: {'Yes' if torch.cuda.is_available() else 'No'}")  # CUDA 지원 여부
    
    # CUDA 사용 가능한 경우 추가 정보
    if torch.cuda.is_available():                                           # CUDA 사용 가능하면
        print(f"   CUDA 장치 개수: {torch.cuda.device_count()}")             # 장치 개수 출력
    
    # cuDNN 상태
    print(f"   cuDNN 사용 가능: {'Yes' if torch.backends.cudnn.enabled else 'No'}")
    
    # 성공 시 True 반환
    return True

# ---------------------- 메인 실행 부분 ---------------------- #
if __name__ == "__main__":                   # 스크립트 직접 실행 시
    print("팀 협업용 GPU 호환성 체크 도구")     # 프로그램 제목 (한글)
    print("Team GPU Compatibility Checker")  # 프로그램 제목 (영문)
    print()                                  # 빈 줄 출력
    
    # 예외 처리 시작
    try:
        # GPU 호환성 체크 실행
        success = check_gpu_compatibility()
        
        # 성공한 경우
        if success:
            print(f"\n✅ GPU 설정 완료! 팀 협업 준비 완료!")            # 성공 메시지
        # 실패한 경우
        else:
            print(f"\n❌ GPU 설정 문제 발견. 위의 해결책을 참고하세요.")  # 실패 메시지
    # 예외 발생 시
    except Exception as e:
        print(f"\n💥 오류 발생: {e}")                                # 오류 메시지 출력
        print(f"💡 Python 환경과 패키지 설치를 확인하세요.")            # 해결 방법 안내
