from typing import Optional

# 표준 라이브러리 임포트
import os, json, yaml, time, hashlib            # os: 파일/디렉터리 조작
                                                # json: JSON 직렬화/역직렬화
                                                # yaml: YAML 파일 처리
                                                # time: 시간 유틸
                                                # hashlib: 해시 함수 (sha1 등)
from datetime import datetime                   # 현재 시각 가져오기


# ---------------------- 디렉토리 보장 ---------------------- #
def ensure_dir(p: str):
    """경로 p에 해당하는 디렉토리를 생성(이미 있으면 무시) 후 경로 반환"""
    os.makedirs(p, exist_ok=True)                   # 디렉토리 생성 (이미 있으면 무시)
    return p                                        # 경로 반환


# ---------------------- YAML 로드 ---------------------- #
def load_yaml(path: str):
    """YAML 파일을 읽어 Python 객체(dict 등)로 반환"""
    with open(path, 'r', encoding='utf-8') as f:    # 파일 열기
        return yaml.safe_load(f)                    # 안전하게 YAML 파싱


# ---------------------- YAML 덤프 ---------------------- #
def dump_yaml(obj, path: str):
    """Python 객체를 YAML 파일로 저장 (한글/키순서 유지)"""
    ensure_dir(os.path.dirname(path))               # 상위 디렉토리 보장
    with open(path, 'w', encoding='utf-8') as f:    # 파일 열기 (쓰기 모드)
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False) # YAML 저장


# ---------------------- JSONL append ---------------------- #
def jsonl_append(path: str, rec: dict):
    """JSON Lines 포맷으로 레코드를 파일 끝에 추가"""
    ensure_dir(os.path.dirname(path))               # 상위 디렉토리 보장
    with open(path, 'a', encoding='utf-8') as f:    # 파일 열기 (추가 모드)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # JSON 직렬화 후 한 줄 저장


# ---------------------- 현재 시각 문자열 ---------------------- #
def now(fmt="%Y%m%d-%H%M%S"):
    """현재 시각을 주어진 포맷 문자열로 반환"""
    return datetime.now().strftime(fmt)             # datetime → 문자열 변환


# ---------------------- 짧은 UID 생성 ---------------------- #
def short_uid(n=6):
    """현재 시각 기반 sha1 해시에서 앞 n글자 추출"""
    return hashlib.sha1(now().encode()).hexdigest()[:n]


# ---------------------- 경로 해석 ---------------------- #
def resolve_path(base_dir: str, p: str) -> str:
    """
    config 파일이 있는 디렉토리를 기준으로
    상대경로를 절대경로로 변환
    """
    if os.path.isabs(p):
        return p
    
    # ./로 시작하는 경우 프로젝트 루트 기준으로 처리
    if p.startswith('./'):
        # 현재 작업 디렉토리(프로젝트 루트) 기준
        return os.path.normpath(os.path.join(os.getcwd(), p[2:]))
    
    # 기존 로직: config 디렉토리 기준
    return os.path.normpath(os.path.join(base_dir, p))


# ---------------------- 파일 필수 확인 ---------------------- #
def require_file(path: str, hint: str = ""):
    """
    파일이 없으면 FileNotFoundError 발생
    - hint: 추가 힌트 메시지 (에러 디버깅 도움)
    """
    if not os.path.isfile(path):                  # 파일 존재 여부 확인
        cwd = os.getcwd()                         # 현재 작업 디렉토리 확인
        msg = f"필수 파일이 없습니다: {path} (cwd={cwd})"  # 기본 에러 메시지
        if hint:                                  # 힌트가 있으면 추가
            msg += f"\n힌트: {hint}"
        raise FileNotFoundError(msg)              # 예외 발생


# ---------------------- 디렉토리 필수 확인 ---------------------- #
def require_dir(path: str, hint: str = ""):
    """
    디렉토리가 없으면 FileNotFoundError 발생
    - hint: 추가 힌트 메시지 (에러 디버깅 도움)
    """
    if not os.path.isdir(path):                   # 디렉토리 존재 여부 확인
        cwd = os.getcwd()                         # 현재 작업 디렉토리 확인
        msg = f"필수 디렉토리가 없습니다: {path} (cwd={cwd})"  # 기본 에러 메시지
        if hint:                                  # 힌트가 있으면 추가
            msg += f"\n힌트: {hint}"
        raise FileNotFoundError(msg)              # 예외 발생


# ---------------------- 날짜별 로그 경로 생성 ---------------------- #
def create_log_path(log_type: str, filename: Optional[str] = None) -> str:
    """
    날짜별 로그 경로 생성
    
    Args:
        log_type: 로그 타입 (train, infer, optimization, pipeline)
        filename: 파일명 (None이면 자동 생성)
    
    Returns:
        전체 로그 파일 경로
    """
    # 현재 날짜 생성 (YYYYMMDD 형식)
    today = time.strftime("%Y%m%d")
    
    # 날짜별 로그 디렉토리 경로 생성
    log_dir = ensure_dir(f"logs/{today}/{log_type}")
    
    # 파일명이 없으면 현재 시각으로 생성
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M")
        filename = f"{log_type}_{timestamp}.log"
    
    # 전체 경로 반환
    return os.path.join(log_dir, filename)