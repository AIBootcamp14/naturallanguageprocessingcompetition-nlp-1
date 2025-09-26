#!/usr/bin/env python3
"""
추론 시각화 모듈
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .base_visualizer import SimpleVisualizer

def create_inference_visualizations(predictions: np.ndarray, model_name: str, output_dir: str,
                                  confidence_scores: Optional[np.ndarray] = None):
    """추론 결과 시각화 - 7개의 다양한 시각화 생성"""
    viz = SimpleVisualizer(output_dir, model_name)
    
    try:
        # 예측값이 확률 형태인 경우 클래스로 변환
        if predictions.ndim == 2:
            pred_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            class_probs = predictions
        else:
            pred_classes = predictions
            confidences = confidence_scores if confidence_scores is not None else np.ones_like(predictions)
            class_probs = None
        
        unique, counts = np.unique(pred_classes, return_counts=True)
        
        # 1. 클래스별 예측 분포 (막대그래프)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color=viz.colors[:len(unique)], alpha=0.7)
        
        plt.title(f'클래스별 예측 분포 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('클래스')
        plt.ylabel('예측 개수')
        
        # 백분율 표시
        total = len(pred_classes)
        for i, (cls, count) in enumerate(zip(unique, counts)):
            percentage = (count / total) * 100
            plt.text(cls, count + total*0.01, f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        viz.save_plot('01_class_distribution.png')
        
        # 2. 신뢰도 분포 히스토그램
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        
        # 통계선 표시
        mean_conf = np.mean(confidences)
        median_conf = np.median(confidences)
        plt.axvline(float(mean_conf), color='red', linestyle='--', alpha=0.8, 
                   label=f'평균: {mean_conf:.3f}')
        plt.axvline(float(median_conf), color='green', linestyle='--', alpha=0.8, 
                   label=f'중간값: {median_conf:.3f}')
        
        plt.title(f'신뢰도 분포 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('신뢰도 점수')
        plt.ylabel('빈도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        viz.save_plot('02_confidence_distribution.png')
        
        # 3. 클래스별 평균 신뢰도 비교
        plt.figure(figsize=(10, 6))
        class_confidences = []
        class_labels = []
        class_stds = []
        
        for cls in unique:
            mask = pred_classes == cls
            avg_conf = np.mean(confidences[mask])
            std_conf = np.std(confidences[mask])
            class_confidences.append(avg_conf)
            class_stds.append(std_conf)
            class_labels.append(f'클래스 {cls}')
        
        bars = plt.bar(class_labels, class_confidences, 
                      color=viz.colors[:len(class_labels)], alpha=0.7,
                      yerr=class_stds, capsize=5)
        plt.title(f'클래스별 평균 신뢰도 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('클래스')
        plt.ylabel('평균 신뢰도')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for bar, conf, std in zip(bars, class_confidences, class_stds):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                    f'{conf:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        viz.save_plot('03_class_confidence_comparison.png')
        
        # 4. 신뢰도 구간별 예측 분포
        plt.figure(figsize=(12, 6))
        
        # 신뢰도 구간 정의
        confidence_bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        bin_labels = ['매우낮음\n(0-0.5)', '낮음\n(0.5-0.7)', '보통\n(0.7-0.8)', 
                     '높음\n(0.8-0.9)', '매우높음\n(0.9-0.95)', '확실\n(0.95-1.0)']
        
        # 각 구간별 개수 계산
        bin_counts = []
        for i in range(len(confidence_bins)-1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if i == len(confidence_bins)-2:  # 마지막 구간은 1.0 포함
                mask = (confidences >= confidence_bins[i]) & (confidences <= confidence_bins[i+1])
            bin_counts.append(np.sum(mask))
        
        colors = ['#FF6B6B', '#FFA726', '#FFCC02', '#66BB6A', '#42A5F5', '#AB47BC']
        bars = plt.bar(bin_labels, bin_counts, color=colors, alpha=0.7)
        
        plt.title(f'신뢰도 구간별 예측 분포 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('신뢰도 구간')
        plt.ylabel('예측 개수')
        
        # 백분율 표시
        for bar, count in zip(bars, bin_counts):
            percentage = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + total*0.01,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        viz.save_plot('04_confidence_bins.png')
        
        # 5. 클래스별 신뢰도 분포 (박스플롯)
        plt.figure(figsize=(12, 6))
        
        confidence_by_class = []
        for cls in unique:
            mask = pred_classes == cls
            confidence_by_class.append(confidences[mask])
        
        bp = plt.boxplot(confidence_by_class, patch_artist=True)
        plt.xticks(range(1, len(unique)+1), [f'클래스 {cls}' for cls in unique])
        plt.title(f'클래스별 신뢰도 분포 (박스플롯) - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('클래스')
        plt.ylabel('신뢰도')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        viz.save_plot('05_confidence_boxplot.png')
        
        # 6. 종합 추론 분석 (2x2 레이아웃)
        plt.figure(figsize=(15, 10))
        
        # 좌상단: 클래스 분포 파이차트
        plt.subplot(2, 2, 1)
        plt.pie(counts, labels=[f'클래스 {cls}' for cls in unique], autopct='%1.1f%%',
               colors=viz.colors[:len(unique)], startangle=90)
        plt.title('클래스 비율')
        
        # 우상단: 신뢰도 히스토그램 (간소화)
        plt.subplot(2, 2, 2)
        plt.hist(confidences, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(mean_conf, color='red', linestyle='--', label=f'평균: {mean_conf:.3f}')
        plt.title('신뢰도 분포')
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        plt.legend()
        
        # 좌하단: 클래스별 평균 신뢰도
        plt.subplot(2, 2, 3)
        plt.bar(range(len(unique)), class_confidences, 
               color=viz.colors[:len(unique)], alpha=0.7)
        plt.title('클래스별 평균 신뢰도')
        plt.xlabel('클래스')
        plt.ylabel('평균 신뢰도')
        plt.xticks(range(len(unique)), [f'C{cls}' for cls in unique])
        
        # 우하단: 통계 요약
        plt.subplot(2, 2, 4)
        stats_text = f"""추론 통계 요약:
총 예측 샘플: {len(pred_classes):,}개
고유 클래스: {len(unique)}개
평균 신뢰도: {mean_conf:.3f}
신뢰도 표준편차: {np.std(confidences):.3f}
높은 신뢰도(>0.9): {np.sum(confidences > 0.9):,}개 ({np.sum(confidences > 0.9)/len(confidences)*100:.1f}%)
낮은 신뢰도(<0.5): {np.sum(confidences < 0.5):,}개 ({np.sum(confidences < 0.5)/len(confidences)*100:.1f}%)"""
        
        plt.text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
                transform=plt.gca().transAxes)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.suptitle(f'종합 추론 분석 - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        viz.save_plot('06_inference_summary.png')
        
        # 7. 클래스 확률 분포 히트맵 (확률 예측의 경우)
        if class_probs is not None and class_probs.shape[1] > 1:
            plt.figure(figsize=(12, 8))
            
            # 각 클래스별 확률 분포 샘플링 (시각화를 위해 최대 1000개)
            sample_size = min(1000, class_probs.shape[0])
            sample_indices = np.random.choice(class_probs.shape[0], sample_size, replace=False)
            sample_probs = class_probs[sample_indices]
            
            # 히트맵 생성
            plt.imshow(sample_probs.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            plt.colorbar(label='확률')
            
            plt.title(f'클래스별 확률 분포 히트맵 - {model_name}', fontsize=16, fontweight='bold')
            plt.xlabel('샘플 인덱스')
            plt.ylabel('클래스')
            plt.yticks(range(class_probs.shape[1]), [f'클래스 {i}' for i in range(class_probs.shape[1])])
            
            viz.save_plot('07_probability_heatmap.png')
        
        print(f"✅ Inference visualizations completed: {viz.images_dir}")
        print(f"📊 Generated {len(list(viz.images_dir.glob('*.png')))} inference visualization images")
        
    except Exception as e:
        print(f"❌ Inference visualization failed: {str(e)}")

def visualize_inference_pipeline(predictions: np.ndarray, model_name: str, output_dir: str,
                               confidence_scores: Optional[np.ndarray] = None):
    """추론 파이프라인 시각화 호출"""
    create_inference_visualizations(predictions, model_name, output_dir, confidence_scores)
