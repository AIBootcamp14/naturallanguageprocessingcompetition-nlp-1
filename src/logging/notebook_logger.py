# src/logging/notebook_logger.py
"""
노트북용 범용 로깅 및 결과 저장 유틸리티

이 모듈은 Jupyter 노트북에서 실행되는 모든 작업의 출력을 
체계적으로 로깅하고 결과를 저장하는 기능을 제공합니다.
"""

import os                                                   # 운영체제 파일/디렉터리 조작
import sys                                                  # 시스템 관련 기능 (stdout/stderr 제어)
import io                                                   # 입출력 스트림 처리
import logging                                              # 파이썬 표준 로깅 모듈
import matplotlib.pyplot as plt                             # 그래프 시각화 라이브러리
import pandas as pd                                         # 데이터프레임 처리
import numpy as np                                          # 수치 계산 라이브러리
from datetime import datetime                               # 현재 시간 처리
from pathlib import Path                                    # 경로 처리 라이브러리
from contextlib import contextmanager                       # 컨텍스트 매니저 데코레이터
from typing import Optional, Union, Dict, Any               # 타입 힌트
import json                                                 # JSON 직렬화/역직렬화


# ==================== 노트북용 범용 로거 클래스 ==================== #
# 노트북용 범용 로거 클래스 정의
class NotebookLogger:
    # 초기화 함수 정의
    def __init__(self, file_name: str, base_log_dir: str = "notebooks", folder_name: str = "analysis"):
        """
        Args:
            file_name: 파일 이름 (예: "data_analysis", "model_comparison")
            base_log_dir: 기본 로그 디렉토리 (예: "notebooks")
            folder_name: 폴더명 (예: "unit_tests", "results_comparison", "analysis")
        """
        self.file_name = file_name                                  # 파일 이름 저장
        self.folder_name = folder_name                              # 폴더명 저장
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # 현재 시간 타임스탬프 생성
        
        # 로그 디렉토리 구조 생성 메서드 호출
        self._setup_directories(base_log_dir, folder_name, file_name)
        
        # 로거 설정
        self.logger = self._setup_logger()                          # 로거 초기화 메서드 호출
        self.test_results = {}                                      # 결과 저장 딕셔너리
        self.start_time = datetime.now()                            # 시작 시간 기록
        
        # 출력 캡처를 위한 설정
        self.original_stdout = sys.stdout                           # 원본 표준 출력 저장
        self.original_stderr = sys.stderr                           # 원본 표준 에러 저장
        
        self.log_info(f"노트북 작업 시작: {file_name}")               # 작업 시작 로그
        self.log_info(f"로그 디렉토리: {self.base_dir}")              # 로그 디렉터리 경로 로그
    
    # ---------------------- 디렉토리 설정 메서드 ---------------------- #
    def _setup_directories(self, base_log_dir: str, folder_name: str, file_name: str):
        """디렉토리 구조를 설정하는 메서드"""
        # 로그 디렉토리 구조 생성: base_log_dir/folder_name/file_name/timestamp
        self.base_dir = "notebooks" /Path(base_log_dir) / folder_name / file_name / self.timestamp
        self.log_dir = self.base_dir / "logs"               # 로그 파일 디렉터리
        self.image_dir = self.base_dir / "images"           # 이미지 저장 디렉터리
        self.data_dir = self.base_dir / "data"              # 데이터 저장 디렉터리
        self.results_dir = self.base_dir / "results"        # 결과 저장 디렉터리
        
        # 디렉토리 생성 처리
        for dir_path in [self.log_dir, self.image_dir, self.data_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)     # 디렉터리 생성 (중간 경로 포함)
    
    # ---------------------- 로거 초기화 함수 ---------------------- #
    # 로거 초기화 함수 정의
    def _setup_logger(self) -> logging.Logger:
        """로거 초기화"""
        logger = logging.getLogger(f"notebook_{self.file_name}")            # 파일별 로거 생성
        logger.setLevel(logging.DEBUG)                                      # 로그 레벨을 DEBUG로 설정
        
        # 기존 핸들러 제거 처리
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)   # 핸들러 제거
        
        # 파일 핸들러 추가
        log_file = self.log_dir / f"{self.file_name}_{self.timestamp}.log"  # 로그 파일 경로 생성
        file_handler = logging.FileHandler(log_file, encoding='utf-8')      # 파일 핸들러 생성
        file_handler.setLevel(logging.DEBUG)                                # 핸들러 로그 레벨 설정
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'          # 포맷 문자열
        )                                                                   # 포맷터 생성 완료
        file_handler.setFormatter(formatter)                                # 핸들러에 포맷터 적용
        logger.addHandler(file_handler)                                     # 로거에 핸들러 추가
        
        # 설정된 로거 반환
        return logger
    
    # ---------------------- 정보 로그 기록 함수 ---------------------- #
    # 정보 로그 기록 함수 정의
    def log_info(self, message: str):
        self.logger.info(message)    # 로거에 정보 메시지 기록
        print(f"📝 {message}")      # 콘솔에 정보 메시지 출력
    
    # ---------------------- 에러 로그 기록 함수 ---------------------- #
    # 에러 로그 기록 함수 정의
    def log_error(self, message: str, exception: Optional[Exception] = None):
        # 예외 객체가 있는 경우 처리
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True)    # 상세 예외 정보 포함 로그
            print(f"❌ {message}: {str(exception)}")                           # 콘솔에 에러 메시지 출력
        # 예외 객체가 없는 경우 처리
        else:
            self.logger.error(message)      # 기본 에러 메시지 로그
            print(f"❌ {message}")         # 콘솔에 에러 메시지 출력
    
    # ---------------------- 경고 로그 기록 함수 ---------------------- #
    # 경고 로그 기록 함수 정의
    def log_warning(self, message: str):
        self.logger.warning(message)                # 로거에 경고 메시지 기록
        print(f"⚠️ {message}")                     # 콘솔에 경고 메시지 출력
    
    # ---------------------- 성공 로그 기록 함수 ---------------------- #
    # 성공 로그 기록 함수 정의
    def log_success(self, message: str):
        self.logger.info(f"SUCCESS: {message}")     # 로거에 성공 메시지 기록
        print(f"✅ {message}")                     # 콘솔에 성공 메시지 출력
    
    @contextmanager # 컨텍스트 매니저 데코레이터
    # 출력 캡처 컨텍스트 매니저 정의
    def capture_output(self, section_name: str):
        """출력 캡처 컨텍스트 매니저"""
        captured_output = io.StringIO()                     # 표준 출력 캡처용 스트림
        captured_error = io.StringIO()                      # 표준 에러 캡처용 스트림
        
        # stdout, stderr 리다이렉트
        sys.stdout = captured_output                        # 표준 출력을 캡처 스트림으로 변경
        sys.stderr = captured_error                         # 표준 에러를 캡처 스트림으로 변경
        
        # 컨텍스트 실행 블록
        try:                                                # 예외 처리 시작
            yield captured_output, captured_error           # 캡처 스트림 반환
        # 원래 출력으로 복원 처리
        finally:                                            # 최종 정리 작업
            # 원래 출력으로 복원
            sys.stdout = self.original_stdout               # 표준 출력 원상 복구
            sys.stderr = self.original_stderr               # 표준 에러 원상 복구
            
            # 캡처된 내용 저장
            output_content = captured_output.getvalue()     # 캡처된 표준 출력 내용 가져오기
            error_content = captured_error.getvalue()       # 캡처된 표준 에러 내용 가져오기
            
            # 표준 출력 내용이 있는 경우 처리
            if output_content:
                output_file = self.log_dir / f"{section_name}_output.txt"  # 출력 파일 경로 생성
                # 출력 내용을 파일에 저장
                with open(output_file, 'w', encoding='utf-8') as f:  # 파일 열기
                    f.write(output_content)                 # 출력 내용 파일에 쓰기
                self.log_info(f"출력 저장: {output_file}")   # 저장 완료 로그
            
            # 에러 출력 내용이 있는 경우 처리
            if error_content:
                error_file = self.log_dir / f"{section_name}_error.txt"  # 에러 파일 경로 생성
                # 에러 내용을 파일에 저장
                with open(error_file, 'w', encoding='utf-8') as f:  # 파일 열기
                    f.write(error_content)                  # 에러 내용 파일에 쓰기
                self.log_warning(f"에러 출력 저장: {error_file}")  # 저장 완료 경고 로그
            
            # 표준 출력을 콘솔에 재출력
            if output_content:                              # 출력 내용이 있으면
                print(output_content)                       # 콘솔에 출력
            # 에러 출력을 콘솔에 재출력
            if error_content:                               # 에러 내용이 있으면
                print(error_content, file=sys.stderr)       # 표준 에러로 출력
    
    # ---------------------- matplotlib 그림 저장 함수 ---------------------- #
    # 그림 저장 함수 정의
    def save_figure(self, fig, filename: str, title: Optional[str] = None, dpi: int = 300):
        # 예외 처리 블록 시작
        try:
            # 파일 확장자가 없으면 .png 추가
            # 파일 확장자 확인 및 추가
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):  # 확장자 없는 경우
                filename += '.png'                          # 기본 확장자 추가
            
            filepath = self.image_dir / filename            # 이미지 파일 전체 경로 생성
            
            # 제목 설정 처리
            if title:                                       # 제목이 지정된 경우
                fig.suptitle(f"{title} - {self.timestamp}", fontsize=12)  # 그림 제목 설정
            
            # 저장
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')  # 그림 파일 저장
            self.log_success(f"그림 저장: {filepath}")       # 저장 완료 로그
            
            # 메타데이터 저장
            meta_file = self.image_dir / f"{filename.split('.')[0]}_meta.json"  # 메타데이터 파일 경로
            metadata = {                                    # 메타데이터 딕셔너리 생성
                "filename": filename,                       # 파일명
                "title": title,                             # 제목
                "timestamp": self.timestamp,                # 타임스탬프
                "dpi": dpi,                                 # 해상도
                "file_name": self.file_name                 # 파일명
            } 
            
            # 메타데이터 파일 저장
            with open(meta_file, 'w', encoding='utf-8') as f:  # 메타데이터 파일 열기
                json.dump(metadata, f, indent=2, ensure_ascii=False)  # JSON 형태로 저장
            
            return filepath                                 # 저장된 파일 경로 반환
            
        # 예외 발생 시 처리
        except Exception as e:
            self.log_error(f"그림 저장 실패: {filename}", e) # 에러 로그 기록
            return None                                     # None 반환
    
    # ---------------------- 데이터 저장 함수들 ---------------------- #
    # 데이터프레임 저장 함수 정의
    def save_dataframe(self, df: pd.DataFrame, filename: str, description: Optional[str] = None):
        # 예외 처리 블록 시작
        try:
            # 파일 지원 확장자가 없는 경우 파일 확장자 확인 및 추가
            if not filename.endswith(('.csv', '.xlsx', '.json')):
                filename += '.csv'                          # 기본 .csv 확장자 추가
            
            filepath = self.data_dir / filename             # 데이터 파일 전체 경로 생성

            #-------------- 파일 형식에 따른 저장 방식 선택 --------------#
            # CSV 파일인 경우 CSV 형식으로 저장
            if filename.endswith('.csv'):
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
            # Excel 파일인 경우 Excel 형식으로 저장
            elif filename.endswith('.xlsx'):
                df.to_excel(filepath, index=False)
            # JSON 파일인 경우 JSON 형식으로 저장
            elif filename.endswith('.json'):
                df.to_json(filepath, orient='records', indent=2, force_ascii=False)

            # 저장 완료 로그
            self.log_success(f"데이터프레임 저장: {filepath} ({len(df)} 행)")
            
            #-------------- 메타데이터 생성 및 저장 --------------#
            # 메타데이터 파일 경로
            meta_file = self.data_dir / f"{filename.split('.')[0]}_meta.json"
            
            # 메타데이터 딕셔너리 생성
            metadata = {
                "filename": filename,                       # 파일명
                "description": description,                 # 설명
                "shape": list(df.shape),                    # 데이터프레임 크기 (행, 열)
                "columns": list(df.columns),                # 컬럼명 리스트
                "timestamp": self.timestamp,                # 타임스탬프
                "file_name": self.file_name                 # 파일명
            }
            
            # 메타데이터 파일 저장
            with open(meta_file, 'w', encoding='utf-8') as f:           # 메타데이터 파일 열기
                json.dump(metadata, f, indent=2, ensure_ascii=False)    # JSON 형태로 저장
            
            # 저장된 파일 경로 반환
            return filepath
            
        # 예외 발생 시 처리
        except Exception as e:
            self.log_error(f"데이터프레임 저장 실패: {filename}", e)    # 에러 로그 기록
            return None                                             # None 반환
    
    # ---------------------- NumPy 배열 저장 함수 ---------------------- #
    # NumPy 배열 저장 함수 정의
    def save_numpy_array(self, arr: np.ndarray, filename: str, description: Optional[str] = None):
        # 예외 처리 블록 시작
        try:
            # 파일 확장자 확인 및 추가
            if not filename.endswith('.npy'):               # .npy 확장자가 없는 경우
                filename += '.npy'                          # .npy 확장자 추가
            
            filepath = self.data_dir / filename             # NumPy 파일 전체 경로 생성
            np.save(filepath, arr)                          # NumPy 배열을 파일에 저장
            
            self.log_success(f"NumPy 배열 저장: {filepath} {arr.shape}")  # 저장 완료 로그
            
            #-------------- 메타데이터 생성 및 저장 --------------#
            # 메타데이터 파일 경로
            meta_file = self.data_dir / f"{filename.split('.')[0]}_meta.json"

            # 메타데이터 딕셔너리 생성
            metadata = {
                "filename": filename,                       # 파일명
                "description": description,                 # 설명
                "shape": list(arr.shape),                   # 배열 모양
                "dtype": str(arr.dtype),                    # 데이터 타입
                "timestamp": self.timestamp,                # 타임스탬프
                "file_name": self.file_name                 # 파일명
            }
            
            # 메타데이터 파일 저장
            with open(meta_file, 'w', encoding='utf-8') as f:           # 메타데이터 파일 열기
                json.dump(metadata, f, indent=2, ensure_ascii=False)    # JSON 형태로 저장
            
            # 저장된 파일 경로 반환
            return filepath
            
        # 예외 발생 시 처리
        except Exception as e:
            self.log_error(f"NumPy 배열 저장 실패: {filename}", e)    # 에러 로그 기록
            return None                                             # None 반환
    
    # ---------------------- 테스트 결과 저장 함수 ---------------------- #
    # 결과 저장 함수 정의
    def save_test_result(self, section: str, result: dict):
        # 결과 딕셔너리에 저장
        self.test_results[section] = {
            **result,                                       # 기존 결과 데이터 병합
            "timestamp": datetime.now().isoformat(),        # 현재 시간 타임스탬프 추가
            "section": section                              # 섹션명 추가
        }
        
        #------------------------ 결과 파일 저장 -------------------------#
        # 결과 파일 경로 생성
        result_file = self.results_dir / f"{section}_result.json"
        # 결과 파일 저장
        with open(result_file, 'w', encoding='utf-8') as f:  # 결과 파일 열기
            json.dump(self.test_results[section], f, indent=2, ensure_ascii=False)  # JSON 형태로 저장
        
        self.log_info(f"결과 저장: {section}")               # 저장 완료 로그
    
    # ---------------------- 성능 메트릭 저장 함수 ---------------------- #
    # 성능 메트릭 저장 함수 정의
    def save_performance_metrics(self, metrics: dict, section: str = "performance"):
        # 메타데이터 포함 메트릭 딕셔너리 생성
        metrics_with_meta = {                               
            "metrics": metrics,                             # 성능 메트릭 데이터
            "file_name": self.file_name,                    # 파일명
            "section": section,                             # 섹션명
            "timestamp": datetime.now().isoformat()         # 현재 시간 타임스탬프
        }
        
        # 메트릭 파일 경로 생성
        metrics_file = self.results_dir / f"{section}_metrics.json"
        
        # 메트릭 파일 저장
        with open(metrics_file, 'w', encoding='utf-8') as f:                # 메트릭 파일 열기
            json.dump(metrics_with_meta, f, indent=2, ensure_ascii=False)   # JSON 형태로 저장
        
        self.log_success(f"성능 메트릭 저장: {metrics_file}")                 # 저장 완료 로그
    
    # ---------------------- 작업 완료 및 정리 함수 ---------------------- #
    # 작업 완료 및 최종 결과 저장 함수 정의
    def finalize_test(self):
        end_time = datetime.now()                                           # 작업 종료 시간 기록
        duration = (end_time - self.start_time).total_seconds()             # 작업 소요 시간 계산
        
        # 전체 작업 요약 딕셔너리 생성
        summary = {
            "file_name": self.file_name,                    # 파일명
            "folder_name": self.folder_name,                # 폴더명
            "start_time": self.start_time.isoformat(),      # 시작 시간
            "end_time": end_time.isoformat(),               # 종료 시간
            "duration_seconds": duration,                   # 소요 시간(초)
            "total_sections": len(self.test_results),       # 총 섹션 수
            "results": self.test_results,                   # 결과들
            "log_directory": str(self.base_dir)             # 로그 디렉터리 경로
        }
        
        # 최종 요약 파일 저장
        summary_file = self.base_dir / "summary.json"       # 요약 파일 경로 생성
        
        # 요약 파일 저장
        with open(summary_file, 'w', encoding='utf-8') as f:  # 요약 파일 열기
            json.dump(summary, f, indent=2, ensure_ascii=False)  # JSON 형태로 저장
        
        self.log_success(f"작업 완료! 총 소요 시간: {duration:.2f}초")       # 완료 로그
        self.log_success(f"결과 요약: {summary_file}")          # 요약 파일 경로 로그
        
        # 최종 요약을 콘솔에 출력
        print("\n" + "="*50)                                   # 구분선 출력
        print(f"🏁 노트북 작업 완료: {self.file_name}")          # 작업 완료 메시지
        print("="*50)                                          # 구분선 출력
        print(f"📁 결과 디렉토리: {self.base_dir}")             # 결과 디렉터리 출력
        print(f"⏱️ 소요 시간: {duration:.2f}초")                # 소요 시간 출력
        print(f"📊 섹션 수: {len(self.test_results)}")           # 섹션 수 출력
        print("="*50)                                          # 구분선 출력
        
        # 작업 요약 반환
        return summary


# ==================== 편의 함수들 ==================== #
# 노트북 로거 생성 편의 함수 정의
def create_notebook_logger(file_name: str, folder_name: str = "analysis", base_log_dir: str = "notebooks") -> NotebookLogger:
    # NotebookLogger 인스턴스 반환
    return NotebookLogger(file_name, base_log_dir, folder_name)


# 작업 섹션 시작 로깅 데코레이터 정의
def log_section(logger: NotebookLogger, section_name: str):
    # 데코레이터 함수 정의
    def decorator(func):
        # 래퍼 함수 정의
        def wrapper(*args, **kwargs):
            logger.log_info(f"=== {section_name} 시작 ===")  # 섹션 시작 로그
            # 함수 실행 예외 처리
            try:
                result = func(*args, **kwargs)                      # 원본 함수 실행
                logger.log_success(f"=== {section_name} 완료 ===")   # 섹션 완료 로그
                return result                                       # 함수 결과 반환
            # 예외 발생 시 처리
            except Exception as e:
                logger.log_error(f"=== {section_name} 실패 ===", e)  # 섹션 실패 로그
                raise                                               # 예외 재발생
        return wrapper                                              # 래퍼 함수 반환
    return decorator                                                # 데코레이터 반환


# ---------------------- 메인 실행 블록 ---------------------- #
# 스크립트 직접 실행 시 테스트
if __name__ == "__main__":
    # 테스트 예제
    logger = create_notebook_logger("example_analysis", "test_folder")  # 예제 로거 생성
    
    # 기본 로깅 테스트
    logger.log_info("노트북 로거 예제 시작")                              # 테스트 시작 로그
    logger.log_success("성공 메시지 테스트")                             # 성공 메시지 테스트
    logger.log_warning("경고 메시지 테스트")                             # 경고 메시지 테스트
    
    # 그림 저장 테스트
    fig, ax = plt.subplots()                                            # matplotlib 그림 생성
    ax.plot([1, 2, 3], [1, 4, 2])                                      # 예제 데이터 플롯
    ax.set_title("예제 그래프")                                          # 그래프 제목 설정
    logger.save_figure(fig, "example_plot", "예제 플롯")                 # 그림 저장
    plt.close(fig)                                                      # 그림 메모리 해제
    
    # 데이터 저장 테스트
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})                # 예제 데이터프레임 생성
    logger.save_dataframe(df, "example_data", "예제 데이터프레임")        # 데이터프레임 저장
    
    # 결과 저장
    logger.save_test_result("example_section", {                        # 결과 저장
        "status": "success",                                            # 상태
        "score": 0.95,                                                  # 점수
        "details": "예제 작업 완료"                                      # 상세 내용
    })
    
    # 작업 완료
    logger.finalize_test()