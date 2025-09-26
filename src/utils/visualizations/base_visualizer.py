#!/usr/bin/env python3
"""
베이스 시각화 클래스 및 폰트 설정
"""

import os
import numpy as np
import pandas as pd

# matplotlib 백엔드를 Agg로 설정 (tkinter 오류 방지)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # 나눔고딕 폰트 경로 및 설정
        font_path = './font/NanumGothic.ttf'
        
        # 절대 경로로 변환
        if not os.path.isabs(font_path):
            base_dir = Path(__file__).parent.parent.parent.parent  # src/utils/visualization/에서 프로젝트 루트로
            font_path = str(base_dir / 'font' / 'NanumGothic.ttf')
        
        if os.path.exists(font_path):
            # 폰트 등록 및 설정 (한글 텍스트 표시를 위함)
            fontprop = fm.FontProperties(fname=font_path)
            fe = fm.FontEntry(fname=font_path, name='NanumGothic')
            fm.fontManager.ttflist.insert(0, fe)
            
            # matplotlib 설정 - 한글과 영문 호환성을 위한 폰트 패밀리 설정
            plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']
            plt.rcParams['font.size'] = 10                   # 기본 글자 크기 설정
            plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
            
            # 글자 겹침 방지를 위한 레이아웃 설정
            plt.rcParams['figure.autolayout'] = True         # 자동 레이아웃 조정
            plt.rcParams['axes.titlepad'] = 20               # 제목과 축 사이 여백
            
            print("✅ 나눔고딕 폰트 로드 성공")
            return True
        else:
            print(f"❌ 폰트 파일을 찾을 수 없습니다: {font_path}")
            return False
    except Exception as e:
        print(f"❌ 폰트 로드 실패: {e}")
        # 폴백 설정
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 폰트 설정 실행
setup_korean_font()

class SimpleVisualizer:
    """간단한 시각화 클래스"""
    
    def __init__(self, output_dir: str, model_name: str):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # 색상 팔레트
        self.colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD', '#17A2B8']
    
    def save_plot(self, filename: str):
        """플롯 저장"""
        path = self.images_dir / filename
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved visualization: {path}")

def create_organized_output_structure(base_dir: str, pipeline_type: str, model_name: str) -> Path:
    """정리된 출력 구조 생성"""
    date_str = datetime.now().strftime('%Y%m%d')
    output_dir = Path(base_dir) / pipeline_type / date_str / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # images 폴더 생성
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    return output_dir
