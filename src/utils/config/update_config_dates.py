"""
Config 날짜 자동 업데이트 유틸리티
추론 설정 파일들의 실험 날짜를 자동으로 업데이트합니다.

사용법:
    python src/utils/config/update_config_dates.py                     # 오늘 날짜로 업데이트
    python src/utils/config/update_config_dates.py --date 20250908     # 특정 날짜로 업데이트
    python src/utils/config/update_config_dates.py --lastest            # 가장 최신 실험 날짜로 업데이트
"""

import argparse                                                 # CLI 인자 파싱 라이브러리
import os                                                       # 운영체제 파일 시스템 접근
import re                                                       # 정규표현식 패턴 매칭
import yaml                                                     # YAML 파일 파싱 (사용되지 않지만 import 유지)
from datetime import datetime                                   # 현재 날짜 시간 처리
from pathlib import Path                                        # 경로 처리 (사용되지 않지만 import 유지)
from typing import List, Dict, Optional                         # 타입 힌트 지원


#--------------------------------- 가장 최신 날짜 찾기 ---------------------------------#
def find_lastest_experiment_date(experiments_dir: str = "experiments/train") -> Optional[str]:
    """
    experiments/train 디렉터리에서 가장 최신 실험 날짜를 찾습니다.
    
    Args:
        experiments_dir: 실험 결과가 저장된 루트 디렉터리 경로
    
    Returns:
        str: YYYYMMDD 형식의 날짜 (예: "20250907") 또는 None (찾지 못한 경우)
    """
    # 실험 디렉터리 존재 여부 확인 (디렉터리 존재하지 않는 경우)
    if not os.path.exists(experiments_dir):
        print(f"❌ 실험 디렉터리가 존재하지 않습니다: {experiments_dir}")
        return None     # None 반환하여 오류 표시
    
    # 날짜 형식의 디렉터리들 찾기 (YYYYMMDD 패턴)
    date_pattern = re.compile(r'^\d{8}$')                       # 8자리 숫자로만 구성된 패턴 (날짜 형식)
    date_dirs = []                                              # 발견된 날짜 디렉터리들을 저장할 리스트
    
    # 실험 디렉터리 내의 모든 항목 순회 (experiments/train 내 모든 파일/폴더 확인)
    for item in os.listdir(experiments_dir):
        item_path = os.path.join(experiments_dir, item)         # 전체 경로 생성
        
        # 디렉터리이면서 날짜 패턴에 맞는 경우만 수집 (폴더이면서 8자리 숫자인 경우)
        if os.path.isdir(item_path) and date_pattern.match(item):
            date_dirs.append(item)                              # 날짜 디렉터리 리스트에 추가
    
    # 날짜 디렉터리가 하나도 없는 경우 오류 처리 (날짜 형식 디렉터리를 찾지 못한 경우)
    if not date_dirs:
        print(f"❌ {experiments_dir}에서 날짜 형식의 디렉터리를 찾을 수 없습니다.")
        return None                                            # None 반환하여 오류 표시
    
    # 가장 최신 날짜 반환 (문자열 정렬로 가장 큰 값이 최신)
    lastest_date = max(date_dirs)                               # 문자열 비교로 가장 큰 날짜 선택
    print(f"📅 가장 최신 실험 날짜: {lastest_date}")
    return lastest_date                                         # 최신 날짜 반환


#---------------------------- 모델별 실험 폴더 찾기 ----------------------------#
def find_model_experiments(base_date: str, experiments_dir: str = "experiments/train") -> Dict[str, str]:
    """
    지정된 날짜의 실험 디렉터리에서 모델별 폴더를 찾습니다.
    
    Args:
        base_date: YYYYMMDD 형식의 날짜 (예: "20250907")
        experiments_dir: 실험 루트 디렉터리 경로
        
    Returns:
        dict: 모델명 -> 폴더명 매핑 (예: {"efficientnet": "efficientnet_b3", "swin": "swin-sighperf"})
    """
    # 지정된 날짜의 실험 디렉터리 경로 생성
    date_dir = os.path.join(experiments_dir, base_date)
    
    # experiments/train/20250907 형태의 경로 (해당 날짜 디렉터리가 존재하지 않는 경우)
    if not os.path.exists(date_dir):
        print(f"❌ 날짜 디렉터리가 존재하지 않습니다: {date_dir}")
        return {}                                              # 빈 딕셔너리 반환
    
    # 모델별 폴더 탐지를 위한 딕셔너리 초기화
    model_dirs = {}                                            # 모델명과 폴더명을 매핑할 딕셔너리
    
    # 날짜 디렉터리 내의 모든 항목 순회하여 모델 폴더 찾기 (날짜 디렉터리 내 모든 파일/폴더 확인)
    for item in os.listdir(date_dir):
        item_path = os.path.join(date_dir, item)               # 전체 경로 생성
        
        # 디렉터리인 경우만 처리
        if os.path.isdir(item_path):
            # 모델명 추정 (efficientnet, swin 등) - 폴더명에서 모델 타입 판별
            if "efficientnet" in item.lower():                 # 폴더명에 efficientnet이 포함된 경우
                model_dirs["efficientnet"] = item              # efficientnet 모델로 매핑
            # 폴더명에 swin이 포함된 경우
            elif "swin" in item.lower():
                model_dirs["swin"] = item                      # swin 모델로 매핑
    
    # 발견된 모델 실험들 출력 (사용자에게 정보 제공)
    print(f"📂 발견된 모델 실험들:")
    for model, folder in model_dirs.items():                   # 모델별로 폴더 정보 출력
        print(f"   - {model}: {folder}")                       # 모델명과 실제 폴더명 표시
    
    return model_dirs                                          # 모델 매핑 딕셔너리 반환


#----------------------------- YAML 파일 경로 업데이트 -----------------------------#
def update_yaml_paths(file_path: str, new_date: str, model_mapping: Dict[str, str]) -> bool:
    """
    YAML 파일의 날짜 경로를 업데이트합니다.
    
    Args:
        file_path: 업데이트할 YAML 파일 경로 (예: "configs/infer.yaml")
        new_date: 새로운 날짜 (YYYYMMDD 형식, 예: "20250907")
        model_mapping: 모델명 -> 폴더명 매핑 딕셔너리
        
    Returns:
        bool: 업데이트 성공 여부 (True: 성공, False: 실패)
    """
    try:
        # YAML 파일 읽기 - 전체 내용을 문자열로 로드
        with open(file_path, 'r', encoding='utf-8') as f:      # UTF-8 인코딩으로 파일 읽기
            content = f.read()                                 # 파일 전체 내용을 문자열로 저장
        
        # 백업 생성 - 원본 파일 손실 방지
        backup_path = f"{file_path}.backup"                    # 백업 파일 경로 생성
        with open(backup_path, 'w', encoding='utf-8') as f:    # 백업 파일 생성
            f.write(content)                                   # 원본 내용을 백업 파일에 저장
        
        updated = False                                        # 업데이트 수행 여부 추적 플래그
        
        # 날짜 패턴 찾기 및 교체 - 정규표현식으로 8자리 숫자 패턴 탐지
        date_pattern = r'(\d{8})'                              # YYYYMMDD 형식의 날짜 패턴
        
        # 더 정확한 경로 업데이트를 위한 라인별 처리 방식 사용
        lines = content.split('\n')                            # 파일 내용을 라인별로 분할
        updated_lines = []                                     # 업데이트된 라인들을 저장할 리스트
        
        # 각 라인별로 날짜 패턴 확인 및 업데이트 수행 (파일의 모든 라인 순회)
        for line in lines:
            updated_line = line                                # 기본적으로 원본 라인 유지
            
            # 체크포인트 경로 업데이트 - infer.yaml의 ckpt.path 처리 (ckpt path 라인이면서 날짜가 있는 경우)
            if 'ckpt' in line and 'path:' in line and re.search(r'\d{8}', line):
                # efficientnet 경로 처리 (efficientnet 모델 경로인 경우)
                if "efficientnet" in line.lower() and "efficientnet" in model_mapping:
                    updated_line = re.sub(r'\d{8}', new_date, line)                     # 날짜 부분을 새 날짜로 교체
                    updated_line = re.sub(r'efficientnet[^/]*', model_mapping["efficientnet"], updated_line)  # 폴더명도 실제 폴더명으로 교체
                    updated = True                                                      # 업데이트 수행됨 표시
                
            # fold_results 경로 업데이트 - infer_highperf.yaml의 fold_results_path 처리
            # fold_results_path 라인이면서 날짜가 있는 경우
            elif 'fold_results_path:' in line and re.search(r'\d{8}', line):
                # swin 모델 경로인 경우
                if "swin" in line.lower() and "swin" in model_mapping:
                    updated_line = re.sub(r'\d{8}', new_date, line)  # 날짜 부분을 새 날짜로 교체
                    updated_line = re.sub(r'swin[^/]*', model_mapping["swin"], updated_line)  # 폴더명도 실제 폴더명으로 교체
                    updated = True                              # 업데이트 수행됨 표시

            updated_lines.append(updated_line)                  # 처리된 라인을 결과 리스트에 추가
        
        #---------------------------- 업데이트가 수행된 경우 파일에 저장 ----------------------------#
        # 하나 이상의 라인이 업데이트된 경우
        if updated:
            # 업데이트된 내용 저장
            updated_content = '\n'.join(updated_lines)         # 라인들을 다시 합쳐서 파일 내용 생성
            with open(file_path, 'w', encoding='utf-8') as f:  # 원본 파일에 덮어쓰기
                f.write(updated_content)                       # 업데이트된 내용 저장
            
            print(f"✅ {file_path} 업데이트 완료")               # 성공 메시지 출력
            return True                                        # 성공 반환
        
        # 업데이트할 내용이 없는 경우
        else:
            print(f"⚠️  {file_path}에서 업데이트할 날짜를 찾지 못했습니다.")
            # 백업 파일 제거 - 변경사항이 없으므로 백업 불필요
            os.remove(backup_path)                             # 백업 파일 삭제
            return False                                       # 실패 반환
    
    # 파일 처리 중 예외 발생시
    except Exception as e:
        print(f"❌ {file_path} 업데이트 중 오류: {e}")          # 오류 메시지 출력
        return False                                          # 실패 반환


def main():
    """
    메인 실행 함수 - 명령행 인자를 처리하여 설정 파일 날짜 업데이트를 수행합니다.
    
    지원하는 실행 모드:
    1. --lastest: 가장 최신 실험 날짜로 자동 업데이트
    2. --date YYYYMMDD: 특정 날짜로 업데이트
    3. 인자 없음: 오늘 날짜로 업데이트
    
    명령행 사용 예시:
    - python update_config_dates.py --lastest
    - python update_config_dates.py --date 20250907
    - python update_config_dates.py --configs configs/infer.yaml
    """
    # 명령행 인자 파서 설정 - 사용자가 다양한 옵션으로 실행할 수 있도록 지원
    parser = argparse.ArgumentParser(description="추론 설정 파일의 실험 날짜 자동 업데이트")            # 프로그램 설명
    parser.add_argument("--date", type=str, help="업데이트할 날짜 (YYYYMMDD 형식, 예: 20250908)")     # 특정 날짜 지정 옵션
    parser.add_argument("--lastest", action="store_true", help="가장 최신 실험 날짜로 업데이트")        # 최신 날짜 자동 탐지 옵션
    parser.add_argument("--configs", nargs="+", default=["configs/infer.yaml", "configs/infer_highperf.yaml"],  # 업데이트할 설정 파일 목록
                       help="업데이트할 설정 파일들")                                                 # 기본값으로 두 개의 주요 설정 파일 지정
    
    args = parser.parse_args()                              # 명령행 인자 파싱 실행
    
    # 프로그램 시작 헤더 출력
    print("🔄 Config 날짜 업데이트 유틸리티")                  # 프로그램 제목 출력
    print("=" * 40)                                         # 구분선 출력
    
    # Step 1: 타겟 날짜 결정 - 사용자 입력에 따라 적절한 날짜 선택
    target_date = None                                      # 업데이트 대상 날짜 초기화
    
    #-------------------------- 최신 날짜 자동 탐지 모드 --------------------------#
    # --lastest 옵션이 지정된 경우
    if args.lastest:
        target_date = find_lastest_experiment_date()         # experiments/train에서 최신 날짜 탐지
        
        # 유효한 날짜를 찾지 못한 경우
        if not target_date:
            print("❌ 최신 실험 날짜를 찾을 수 없습니다.")      # 오류 메시지 출력
            return                                          # 프로그램 종료
    
    #-------------------------- 특정 날짜 지정 모드 --------------------------#
    # --date 옵션이 지정된 경우
    elif args.date:
        # 날짜 형식 검증 - YYYYMMDD 형식인지 확인 (정규표현식으로 8자리 숫자 형식 검증)
        if not re.match(r'^\d{8}$', args.date):
            print("❌ 날짜는 YYYYMMDD 형식이어야 합니다 (예: 20250908)")  # 형식 오류 메시지
            return                                          # 프로그램 종료
        target_date = args.date                             # 사용자 지정 날짜 사용
    
    #---------------------------- 기본 모드 (오늘 날짜 사용) ----------------------------#
    # 아무 옵션도 지정되지 않은 경우
    else:                                                  
        # 오늘 날짜 사용 - 현재 시스템 날짜를 YYYYMMDD 형식으로 변환
        target_date = datetime.now().strftime("%Y%m%d")     # datetime을 사용해 오늘 날짜 생성
    
    print(f"📅 타겟 날짜: {target_date}")                    # 결정된 타겟 날짜 출력
    
    #-------------------------- Step 2: 해당 날짜의 모델 실험들 탐지 및 매핑 구성 --------------------------#
    model_mapping = find_model_experiments(target_date)    # 타겟 날짜의 모델 폴더들 탐지
    
    # 해당 날짜의 실험을 찾지 못한 경우
    if not model_mapping:
        print(f"❌ {target_date} 날짜의 실험 결과를 찾을 수 없습니다.")     # 오류 메시지 출력
        print(f"💡 다음 명령어로 사용 가능한 날짜를 확인하세요:")            # 도움말 메시지
        print(f"   ls experiments/train/")                             # 실제 사용 가능한 날짜 확인 방법 안내
        return  # 프로그램 종료
    
    #-------------------------- Step 3: 설정 파일들 순차적 업데이트 수행 --------------------------#
    updated_count = 0                                      # 성공적으로 업데이트된 파일 카운터
    
    # 지정된 모든 설정 파일 순회
    for config_file in args.configs:
        # 파일이 존재하는 경우
        if os.path.exists(config_file):
            print(f"\n🔧 {config_file} 업데이트 중...")     # 현재 처리 중인 파일 알림
            
            # YAML 파일 업데이트 실행
            if update_yaml_paths(config_file, target_date, model_mapping):
                updated_count += 1  # 성공 카운터 증가
        
        # 파일이 존재하지 않는 경우
        else:
            print(f"⚠️  설정 파일이 존재하지 않습니다: {config_file}")          # 경고 메시지 출력
    
    # Step 4: 업데이트 결과 요약 및 안내 메시지 출력
    print(f"\n✅ 업데이트 완료! ({updated_count}/{len(args.configs)} 파일)")  # 전체 결과 요약
    print(f"💡 백업 파일이 생성되었습니다 (.backup 확장자)")                    # 백업 파일 생성 안내
    
    # Step 5: 후속 작업 안내 - 업데이트 완료 후 실행 가능한 명령어들 제시
    print(f"\n🚀 이제 다음 명령어로 추론을 실행할 수 있습니다:")    # 후속 작업 헤더
    print(f"   # 기본 추론")                                    # 기본 추론 명령어 섹션
    print(f"   python src/inference/infer_main.py --config configs/infer.yaml --mode basic")              # 기본 추론 실행 명령어
    print(f"   # 고성능 추론")                                  # 고성능 추론 명령어 섹션
    print(f"   python src/inference/infer_main.py --config configs/infer_highperf.yaml --mode highperf")  # 고성능 추론 실행 명령어


if __name__ == "__main__":
    main()
