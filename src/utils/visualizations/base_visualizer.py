#!/usr/bin/env python3
"""
베이스 시각화 클래스 및 폰트 설정
시각화에 필요한 기본 설정 및 공통 클래스 제공
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# ------------------------- 서드파티 라이브러리 ------------------------- #
import numpy as np
import pandas as pd

# matplotlib 백엔드를 Agg로 설정
import matplotlib
matplotlib.use('Agg')  # tkinter 오류 방지

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


# ==================== 폰트 설정 함수 ==================== #
# ---------------------- 한글 폰트 설정 함수 ---------------------- #
def setup_korean_font():
    """한글 폰트 설정

    나눔고딕 폰트를 로드하여 matplotlib에서 한글 표시가 가능하도록 설정

    Returns:
        bool: 폰트 로드 성공 여부
    """
    # -------------- 폰트 로드 시도 -------------- #
    # 폰트 설정 시도
    try:
        # -------------- 폰트 경로 설정 -------------- #
        # 나눔고딕 폰트 경로
        font_path = './font/NanumGothic.ttf'

        # -------------- 절대 경로 변환 -------------- #
        # 상대 경로인 경우 절대 경로로 변환
        if not os.path.isabs(font_path):
            base_dir = Path(__file__).parent.parent.parent.parent  # 프로젝트 루트로 이동
            font_path = str(base_dir / 'font' / 'NanumGothic.ttf')  # 절대 경로 생성

        # -------------- 폰트 파일 존재 확인 -------------- #
        # 폰트 파일이 존재하는 경우
        if os.path.exists(font_path):
            # -------------- 폰트 등록 -------------- #
            # FontProperties 객체 생성
            fontprop = fm.FontProperties(fname=font_path)

            # FontEntry 생성 및 등록
            fe = fm.FontEntry(fname=font_path, name='NanumGothic')
            fm.fontManager.ttflist.insert(0, fe)  # 폰트 매니저에 등록

            # -------------- matplotlib 설정 -------------- #
            # 한글과 영문 호환성을 위한 폰트 패밀리 설정
            plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']  # 폰트 패밀리
            plt.rcParams['font.size'] = 10                                 # 기본 글자 크기
            plt.rcParams['axes.unicode_minus'] = False                     # 마이너스 기호 깨짐 방지

            # -------------- 레이아웃 설정 -------------- #
            # 글자 겹침 방지를 위한 설정
            plt.rcParams['figure.autolayout'] = True  # 자동 레이아웃 조정
            plt.rcParams['axes.titlepad'] = 20        # 제목과 축 사이 여백

            print("✅ 나눔고딕 폰트 로드 성공")
            return True

        # -------------- 폰트 파일 없음 -------------- #
        # 폰트 파일을 찾을 수 없는 경우
        else:
            print(f"❌ 폰트 파일을 찾을 수 없습니다: {font_path}")
            return False

    # -------------- 예외 발생 시 처리 -------------- #
    # 폰트 로드 실패 시
    except Exception as e:
        print(f"❌ 폰트 로드 실패: {e}")

        # -------------- 폴백 설정 -------------- #
        # 기본 폰트로 설정
        plt.rcParams['font.family'] = ['DejaVu Sans']  # 기본 폰트
        plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

        return False


# 폰트 설정 실행
setup_korean_font()


# ==================== 베이스 시각화 클래스 ==================== #
# ---------------------- 간단한 시각화 클래스 ---------------------- #
class SimpleVisualizer:
    """간단한 시각화 클래스

    실험 결과 시각화를 위한 기본 기능 제공
    """

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(self, output_dir: str, model_name: str):
        """
        Args:
            output_dir: 출력 디렉토리 경로
            model_name: 모델 이름
        """
        # -------------- 디렉토리 설정 -------------- #
        # 출력 디렉토리 및 이미지 디렉토리 설정
        self.output_dir = Path(output_dir)          # 출력 디렉토리
        self.model_name = model_name                # 모델 이름
        self.images_dir = self.output_dir / "images"  # 이미지 저장 디렉토리

        # -------------- 디렉토리 생성 -------------- #
        # 이미지 디렉토리 생성
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # -------------- 색상 팔레트 설정 -------------- #
        # 시각화에 사용할 색상 팔레트
        self.colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD', '#17A2B8']

    # ---------------------- 플롯 저장 함수 ---------------------- #
    def save_plot(self, filename: str):
        """플롯 저장

        현재 matplotlib 플롯을 파일로 저장

        Args:
            filename: 저장할 파일명
        """
        # -------------- 파일 경로 생성 -------------- #
        # 이미지 파일 경로
        path = self.images_dir / filename

        # -------------- 플롯 저장 -------------- #
        # 고해상도로 저장
        plt.savefig(path, dpi=300, bbox_inches='tight')  # DPI 300, 여백 제거
        plt.close()  # 플롯 닫기

        # 저장 완료 메시지
        print(f"📊 Saved visualization: {path}")


# ==================== 헬퍼 함수들 ==================== #
# ---------------------- 정리된 출력 구조 생성 함수 ---------------------- #
def create_organized_output_structure(base_dir: str, pipeline_type: str, model_name: str) -> Path:
    """정리된 출력 구조 생성

    날짜별, 파이프라인별, 모델별 디렉토리 구조 생성

    Args:
        base_dir: 기본 디렉토리 경로
        pipeline_type: 파이프라인 타입 (train, infer, optimization)
        model_name: 모델 이름

    Returns:
        Path: 생성된 출력 디렉토리 경로
    """
    # -------------- 날짜 문자열 생성 -------------- #
    # 현재 날짜를 YYYYMMDD 형식으로 생성
    date_str = datetime.now().strftime('%Y%m%d')

    # -------------- 출력 디렉토리 경로 생성 -------------- #
    # base_dir/pipeline_type/YYYYMMDD/model_name 구조
    output_dir = Path(base_dir) / pipeline_type / date_str / model_name

    # -------------- 디렉토리 생성 -------------- #
    # 상위 디렉토리까지 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------- images 폴더 생성 -------------- #
    # 시각화 파일 저장용 폴더
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    return output_dir
