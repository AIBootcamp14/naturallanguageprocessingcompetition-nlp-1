"""
간단한 노트북용 로거
"""

from datetime import datetime
from typing import Optional
from pathlib import Path


class NotebookLogger:
    """간단한 노트북 로거 클래스"""
    
    def __init__(self, log_path: Optional[str] = None, print_also: bool = True):
        """
        Args:
            log_path: 로그 파일 경로
            print_also: 콘솔에도 출력할지 여부
        """
        self.print_also = print_also
        self.log_path = log_path
        
        # 로그 파일 설정
        if log_path:
            # 디렉토리 생성
            log_dir = Path(log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, 'w', encoding='utf-8')
        else:
            self.log_file = None
    
    def write(self, message: str):
        """메시지를 로그 파일과 콘솔에 출력"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # 파일에 쓰기
        if self.log_file:
            self.log_file.write(f"{log_message}\n")
            self.log_file.flush()
        
        # 콘솔에 출력
        if self.print_also:
            print(message)
    
    def __del__(self):
        """소멸자: 파일 닫기"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()


# 편의 함수
def create_notebook_logger(log_path: Optional[str] = None, print_also: bool = True):
    """NotebookLogger 인스턴스 생성 편의 함수"""
    return NotebookLogger(log_path, print_also)
