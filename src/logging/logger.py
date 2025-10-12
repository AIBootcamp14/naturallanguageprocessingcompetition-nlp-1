# ---------------------- 로그 기록 모듈 ---------------------- #

import sys                                       # 시스템 관련 기능 모듈
from datetime import datetime                    # 현재 시간 가져오기 모듈
from tqdm import tqdm                            # 진행률 표시 모듈

# ---------------------- Logger 클래스 정의 ---------------------- #
class Logger:                                    # Logger 클래스 정의
    """
    로그를 파일에 저장하고, 표준 출력(stdout)과 표준 에러(stderr)를
    로그 파일로 리디렉션하는 기능이 추가된 Logger 클래스
    진행률 표시줄(tqdm)의 중복 출력을 방지하는 필터링 기능 포함
    """
    # 초기화 함수 정의
    def __init__(self, log_path: str, print_also: bool = True):
        self.log_path = log_path                 # 로그 파일 경로 저장
        self.print_also = print_also             # 콘솔 출력 여부 저장
        self.encoding = 'utf-8'                  # 인코딩 명시 (sys.stdout 리디렉션용)
        # 원본 표준 출력을 저장해 둡니다.
        self.original_stdout = sys.stdout        # 원본 표준 출력 저장
        self.original_stderr = sys.stderr        # 원본 표준 에러 저장
        # 로그 파일을 열고, 라인 버퍼링을 사용합니다.
        self.log_file = open(log_path, 'a', encoding='utf-8', buffering=1)  # 로그 파일 열기
        # 진행률 표시줄 중복 방지를 위한 변수
        self.last_progress_line = None           # 마지막 진행률 라인 저장
        self.last_progress_percent = None        # 마지막 진행률 퍼센티지 저장

    
    # ---------------------- 진행률 라인 확인 함수 ---------------------- #
    def _is_progress_line(self, message: str) -> bool:
        """
        메시지가 tqdm 진행률 표시줄인지 확인
        퍼센티지(%), it/s, s/it 등이 포함된 라인을 감지
        """
        # ANSI 이스케이프 시퀀스 제거
        import re
        clean_message = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', message)

        progress_indicators = ['%|', 'it/s', 's/it', '/s]', '|', '[00:']
        # 진행률 지표가 포함되어 있고, 숫자가 포함된 경우만 진행률로 간주
        has_indicator = any(indicator in clean_message for indicator in progress_indicators)
        has_digit = any(c.isdigit() for c in clean_message)
        return has_indicator and has_digit

    # ---------------------- 퍼센티지 추출 함수 ---------------------- #
    def _extract_percentage(self, message: str) -> float:
        """
        메시지에서 진행률 퍼센티지를 추출
        예: "Training:  50%|█████     | 100/200" -> 50.0
        """
        import re
        # ANSI 이스케이프 시퀀스 제거
        clean_message = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', message)

        # 퍼센티지 패턴 찾기: 숫자 + %
        match = re.search(r'(\d+)%', clean_message)
        if match:
            return float(match.group(1))
        return None

    # 로그 기록 함수 정의
    def write(self, message: str, print_also: bool = True, print_error: bool = False):
        """
        로그 메시지를 파일에 기록하고,
        print_also=True일 경우 콘솔에도 출력합니다.
        진행률 표시줄의 중복 출력을 방지합니다.
        """
        # 메시지 앞뒤 공백을 제거하고, 개행 문자가 없으면 추가합니다.
        message = message.strip()                # 메시지 공백 제거
        if not message:                          # 메시지가 비어있으면
            return                               # 함수 종료

        # ---------------------- 진행률 라인 중복 방지 (1% 단위) ---------------------- #
        # 진행률 라인인지 확인
        is_progress = self._is_progress_line(message)

        if is_progress:
            # 현재 진행률 퍼센티지 추출
            current_percent = self._extract_percentage(message)

            # 퍼센티지를 추출할 수 있는 경우
            if current_percent is not None:
                # 이전 퍼센티지가 있고, 1% 미만 차이인 경우 건너뛰기
                if self.last_progress_percent is not None:
                    percent_diff = abs(current_percent - self.last_progress_percent)
                    if percent_diff < 1.0:
                        return               # 1% 미만 차이는 기록하지 않음

                # 1% 이상 차이가 나거나 첫 진행률인 경우 기록
                self.last_progress_percent = current_percent
                self.last_progress_line = message
            else:
                # 퍼센티지를 추출할 수 없지만 진행률 라인인 경우 (예: "0/100")
                # 이전 라인과 완전히 동일하면 건너뛰기
                if self.last_progress_line == message:
                    return
                self.last_progress_line = message
        else:
            # 진행률이 아닌 일반 메시지가 오면 진행률 상태 초기화
            self.last_progress_line = None
            self.last_progress_percent = None

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 현재 시간 타임스탬프 생성
        line = f"{timestamp} | {message}\n"      # 타임스탬프와 메시지 결합

        self.log_file.write(line)                # 로그 파일에 기록

        # 콘솔 출력 옵션이 활성화된 경우
        if self.print_also and print_also:
            # 에러 메시지인 경우
            if print_error:
                # 재귀 호출을 피하기 위해 원본 표준 출력을 사용합니다.
                self.original_stdout.write(f"\033[91m{line}\033[0m")  # 빨간색으로 에러 출력
            # 일반 메시지인 경우
            else:
                self.original_stdout.write(line) # 일반 출력
    
    
    # 플러시 함수 정의
    def flush(self):
        """
        스트림 인터페이스에 필요한 flush 메서드입니다.
        """
        self.log_file.flush()   # 로그 파일 버퍼 플러시


    # 리다이렉션 시작 함수 정의
    def start_redirect(self):
        """
        표준 출력(stdout)과 표준 에러(stderr)를 이 로거 인스턴스로 리다이렉션합니다.
        """
        self.write(">> 표준 출력 및 오류를 로그 파일로 리디렉션 시작", print_also=True)  # 리다이렉션 시작 로그
        sys.stdout = self   # 표준 출력을 로거로 리디렉션
        sys.stderr = self   # 표준 에러를 로거로 리디렉션


    # 리다이렉션 중지 함수 정의
    def stop_redirect(self):
        """
        표준 출력(stdout)과 표준 에러(stderr)를 원상 복구합니다.
        """
        self.write(">> 로그 리디렉션 중료.", print_also=True)  # 리다이렉션 중지 로그
        sys.stdout = self.original_stdout        # 표준 출력 원상 복구
        sys.stderr = self.original_stderr        # 표준 에러 원상 복구
    
    
    # tqdm 리다이렉션 함수 정의
    def tqdm_redirect(self):
        """tqdm.write를 이 로거 인스턴스로 리다이렉션합니다."""
        # tqdm 호환 래퍼 함수 정의
        def tqdm_write_wrapper(s, file=None, end="\n", nolock=False):
            self.write(s)                        # 메시지를 로거로 전달
        tqdm.write = tqdm_write_wrapper          # tqdm 출력을 래퍼 함수로 리다이렉션

    
    # 로거 종료 함수 정의
    def close(self):
        """
        로그 파일을 닫습니다.
        """
        self.log_file.close() # 로그 파일 닫기