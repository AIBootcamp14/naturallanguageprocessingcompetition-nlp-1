#!/usr/bin/env python3
"""
간단한 시각화 시스템 - 새로운 모듈 구조를 사용하는 호환성 래퍼
"""

# 호환성을 위한 전체 import
from . import (
    SimpleVisualizer,
    setup_korean_font,
    create_training_visualizations,
    create_inference_visualizations,
    create_optimization_visualizations,
    visualize_training_pipeline,
    visualize_inference_pipeline,
    visualize_optimization_pipeline,
    create_organized_output_structure
)

# 전역적으로 사용되는 함수들
__all__ = [
    'SimpleVisualizer',
    'setup_korean_font',
    'create_training_visualizations',
    'create_inference_visualizations', 
    'create_optimization_visualizations',
    'visualize_training_pipeline',
    'visualize_inference_pipeline',
    'visualize_optimization_pipeline',
    'create_organized_output_structure'
]
