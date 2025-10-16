"""
경로 자동 해석 유틸리티

experiments/[날짜]/... 형태의 경로에서 실제 날짜 폴더를 자동으로 찾아주는 기능
"""

import re
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime


def resolve_date_placeholder(path: str, verbose: bool = True) -> str:
    """
    경로에서 [날짜] 플레이스홀더를 실제 날짜로 치환

    Args:
        path: 경로 (예: "experiments/[날짜]/kobart_ultimate/final_model")
        verbose: 로그 출력 여부

    Returns:
        str: 해석된 경로

    Examples:
        >>> resolve_date_placeholder("experiments/[날짜]/kobart_ultimate/final_model")
        "experiments/20251014/kobart_ultimate/final_model"

        >>> resolve_date_placeholder("experiments/[latest]/kobart_ultimate/final_model")
        "experiments/20251014/kobart_ultimate/final_model"
    """
    # [날짜] 또는 [latest] 패턴 찾기
    pattern = r'\[(?:날짜|date|latest)\]'

    if not re.search(pattern, path):
        return path  # 플레이스홀더가 없으면 그대로 반환

    # experiments 폴더에서 날짜 폴더 찾기
    experiments_dir = Path("experiments")

    if not experiments_dir.exists():
        if verbose:
            print(f"⚠️  experiments 폴더가 없습니다: {experiments_dir}")
        return path

    # 날짜 형식 폴더 찾기 (YYYYMMDD)
    date_folders = []
    for item in experiments_dir.iterdir():
        if item.is_dir() and re.match(r'^\d{8}$', item.name):
            date_folders.append(item.name)

    if not date_folders:
        if verbose:
            print(f"⚠️  experiments 폴더에 날짜 폴더가 없습니다")
        return path

    # 가장 최근 날짜 선택
    latest_date = sorted(date_folders)[-1]

    # 경로 치환
    resolved_path = re.sub(pattern, latest_date, path)

    if verbose:
        print(f"📁 경로 자동 해석:")
        print(f"   원본: {path}")
        print(f"   해석: {resolved_path}")
        print(f"   (사용 가능한 날짜: {', '.join(sorted(date_folders))})")

    return resolved_path


def find_latest_experiment(
    pattern: str,
    date: Optional[str] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    특정 패턴에 맞는 가장 최근 실험 폴더 찾기

    Args:
        pattern: 실험 폴더명 패턴 (예: "kobart_ultimate*")
        date: 특정 날짜 (None이면 가장 최근 날짜)
        verbose: 로그 출력 여부

    Returns:
        Optional[str]: 실험 폴더 경로 (없으면 None)

    Examples:
        >>> find_latest_experiment("kobart_ultimate*")
        "experiments/20251014/20251014_154616_kobart_ultimate_optuna"
    """
    experiments_dir = Path("experiments")

    if not experiments_dir.exists():
        if verbose:
            print(f"⚠️  experiments 폴더가 없습니다")
        return None

    # 날짜 폴더 찾기
    if date is None:
        date_folders = sorted([
            item.name for item in experiments_dir.iterdir()
            if item.is_dir() and re.match(r'^\d{8}$', item.name)
        ])
        if not date_folders:
            if verbose:
                print(f"⚠️  날짜 폴더가 없습니다")
            return None
        date = date_folders[-1]

    date_dir = experiments_dir / date
    if not date_dir.exists():
        if verbose:
            print(f"⚠️  날짜 폴더가 없습니다: {date_dir}")
        return None

    # 패턴에 맞는 실험 폴더 찾기
    from fnmatch import fnmatch
    matching_folders = [
        folder for folder in date_dir.iterdir()
        if folder.is_dir() and fnmatch(folder.name, pattern)
    ]

    if not matching_folders:
        if verbose:
            print(f"⚠️  패턴 '{pattern}'에 맞는 실험 폴더가 없습니다")
        return None

    # 가장 최근 폴더 선택 (타임스탬프 기준)
    latest_folder = sorted(matching_folders, key=lambda x: x.name)[-1]
    result = str(latest_folder)

    if verbose:
        print(f"📁 실험 폴더 찾기:")
        print(f"   패턴: {pattern}")
        print(f"   날짜: {date}")
        print(f"   결과: {result}")
        if len(matching_folders) > 1:
            print(f"   (총 {len(matching_folders)}개 중 가장 최근 선택)")

    return result


def resolve_model_path(path: str, verbose: bool = True) -> str:
    """
    모델 경로 자동 해석 (날짜 + 실험 폴더 자동 탐색)

    Args:
        path: 모델 경로
        verbose: 로그 출력 여부

    Returns:
        str: 해석된 경로

    Examples:
        >>> resolve_model_path("experiments/[날짜]/kobart_ultimate*/kobart/final_model")
        "experiments/20251014/20251014_154616_kobart_ultimate_optuna/kobart/final_model"
    """
    # 1단계: [날짜] 플레이스홀더 치환
    path = resolve_date_placeholder(path, verbose=False)

    # 2단계: 와일드카드(*) 패턴 해석
    if '*' in path or '?' in path:
        parts = Path(path).parts
        current = Path(parts[0])

        for i, part in enumerate(parts[1:], 1):
            if '*' in part or '?' in part:
                # 와일드카드가 있는 부분 처리
                from fnmatch import fnmatch
                if not current.exists():
                    if verbose:
                        print(f"⚠️  경로가 존재하지 않습니다: {current}")
                    return path

                matching = [
                    item for item in current.iterdir()
                    if item.is_dir() and fnmatch(item.name, part)
                ]

                if not matching:
                    if verbose:
                        print(f"⚠️  패턴 '{part}'에 맞는 폴더가 없습니다: {current}")
                    return path

                # 가장 최근 폴더 선택
                latest = sorted(matching, key=lambda x: x.name)[-1]
                current = latest
            else:
                current = current / part

        resolved = str(current)

        if verbose:
            print(f"📁 모델 경로 자동 해석:")
            print(f"   원본: {path}")
            print(f"   해석: {resolved}")

        return resolved

    return path


def save_command_to_experiment(
    output_dir: str,
    command: Optional[List[str]] = None,
    verbose: bool = True
) -> None:
    """
    실행 명령어를 실험 폴더에 저장

    Args:
        output_dir: 실험 출력 디렉토리
        command: 명령어 리스트 (None이면 sys.argv 사용)
        verbose: 로그 출력 여부
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 명령어 가져오기
    if command is None:
        command = sys.argv

    # 타임스탬프
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 명령어 파일 저장
    command_file = output_path / "command.txt"

    with open(command_file, 'w', encoding='utf-8') as f:
        f.write(f"# 실행 시각: {timestamp}\n")
        f.write(f"# 실행 경로: {Path.cwd()}\n")
        f.write(f"\n# 실행 명령어:\n")

        # 명령어를 보기 좋게 포맷
        if len(command) == 1:
            f.write(command[0] + "\n")
        else:
            f.write(command[0] + " \\\n")
            for arg in command[1:-1]:
                f.write(f"  {arg} \\\n")
            f.write(f"  {command[-1]}\n")

    if verbose:
        print(f"💾 실행 명령어 저장: {command_file}")


def get_latest_checkpoint(
    experiment_pattern: str,
    date: Optional[str] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    특정 실험의 가장 최근 체크포인트 폴더 찾기

    Args:
        experiment_pattern: 실험 폴더명 패턴
        date: 특정 날짜 (None이면 가장 최근)
        verbose: 로그 출력 여부

    Returns:
        Optional[str]: 체크포인트 폴더 경로

    Examples:
        >>> get_latest_checkpoint("*kobart_ultimate_optuna*")
        "experiments/20251014/20251014_154616_kobart_ultimate_optuna/checkpoints"
    """
    experiment_dir = find_latest_experiment(experiment_pattern, date, verbose=False)

    if experiment_dir is None:
        if verbose:
            print(f"⚠️  실험 폴더를 찾을 수 없습니다: {experiment_pattern}")
        return None

    checkpoint_dir = Path(experiment_dir) / "checkpoints"

    if not checkpoint_dir.exists():
        if verbose:
            print(f"⚠️  체크포인트 폴더가 없습니다: {checkpoint_dir}")
        return None

    result = str(checkpoint_dir)

    if verbose:
        print(f"📁 체크포인트 폴더:")
        print(f"   실험: {experiment_pattern}")
        print(f"   경로: {result}")

    return result
