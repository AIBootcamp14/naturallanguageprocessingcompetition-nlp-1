#!/usr/bin/env python3
"""
학습 시각화 모듈
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from .base_visualizer import SimpleVisualizer

def create_training_visualizations(fold_results: Dict, model_name: str, output_dir: str, 
                                 history_data: Optional[Dict] = None):
    """학습 결과 시각화 - 7개의 다양한 시각화 생성"""
    viz = SimpleVisualizer(output_dir, model_name)
    
    try:
        # 폴드 데이터 추출
        if 'fold_results' in fold_results:
            fold_data = fold_results['fold_results']
            folds = [f"Fold {f['fold']}" for f in fold_data]
            f1_scores = [f.get('best_f1', f.get('f1', 0)) for f in fold_data]
            accuracies = [f.get('best_accuracy', f.get('accuracy', 0)) for f in fold_data]
        else:
            folds = list(fold_results.keys())
            f1_scores = list(fold_results.values())
            accuracies = f1_scores  # 기본값으로 F1 사용
        
        # 1. 폴드별 F1 성능 비교
        plt.figure(figsize=(10, 6))
        bars = plt.bar(folds, f1_scores, color=viz.colors[:len(folds)], alpha=0.7)
        
        # 값 표시
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 평균선 표시
        avg_f1 = np.mean(f1_scores)
        plt.axhline(y=float(avg_f1), color='red', linestyle='--', alpha=0.7, 
                   label=f'평균: {avg_f1:.3f}')
        
        plt.title(f'폴드별 F1 성능 비교 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('폴드')
        plt.ylabel('F1 점수')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        viz.save_plot('01_fold_f1_performance.png')
        
        # 2. 폴드별 정확도 비교
        plt.figure(figsize=(10, 6))
        bars = plt.bar(folds, accuracies, color=viz.colors[1:len(folds)+1], alpha=0.7)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        avg_acc = np.mean(accuracies)
        plt.axhline(y=float(avg_acc), color='green', linestyle='--', alpha=0.7, 
                   label=f'평균: {avg_acc:.3f}')
        
        plt.title(f'폴드별 정확도 비교 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('폴드')
        plt.ylabel('정확도')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        viz.save_plot('02_fold_accuracy_comparison.png')
        
        # 3. F1 vs 정확도 산점도
        plt.figure(figsize=(10, 6))
        plt.scatter(f1_scores, accuracies, c=range(len(folds)), 
                   s=100, alpha=0.7, cmap='viridis')
        
        # 폴드 라벨 추가
        for i, fold in enumerate(folds):
            plt.annotate(fold, (f1_scores[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('F1 점수')
        plt.ylabel('정확도')
        plt.title(f'F1 점수 vs 정확도 상관관계 - {model_name}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 대각선 표시 (이상적인 관계)
        min_val = min(min(f1_scores), min(accuracies)) - 0.02
        max_val = max(max(f1_scores), max(accuracies)) + 0.02
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='이상적 관계')
        plt.legend()
        viz.save_plot('03_f1_vs_accuracy_scatter.png')
        
        # 4. 성능 분포 히스토그램
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(f1_scores, bins=10, alpha=0.7, color=viz.colors[0], edgecolor='black')
        plt.axvline(float(avg_f1), color='red', linestyle='--', label=f'평균: {avg_f1:.3f}')
        plt.title('F1 점수 분포')
        plt.xlabel('F1 점수')
        plt.ylabel('빈도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(accuracies, bins=10, alpha=0.7, color=viz.colors[1], edgecolor='black')
        plt.axvline(float(avg_acc), color='green', linestyle='--', label=f'평균: {avg_acc:.3f}')
        plt.title('정확도 분포')
        plt.xlabel('정확도')
        plt.ylabel('빈도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'성능 분포 히스토그램 - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        viz.save_plot('04_performance_distribution.png')
        
        # 5. 성능 통계 요약 차트
        plt.figure(figsize=(10, 8))
        
        metrics = ['F1 점수', '정확도']
        means = [avg_f1, avg_acc]
        stds = [np.std(f1_scores), np.std(accuracies)]
        maxs = [max(f1_scores), max(accuracies)]
        mins = [min(f1_scores), min(accuracies)]
        
        x = np.arange(len(metrics))
        width = 0.2
        
        plt.bar(x - width*1.5, means, width, label='평균', color=viz.colors[0], alpha=0.7)
        plt.bar(x - width*0.5, maxs, width, label='최대', color=viz.colors[1], alpha=0.7)
        plt.bar(x + width*0.5, mins, width, label='최소', color=viz.colors[2], alpha=0.7)
        plt.bar(x + width*1.5, stds, width, label='표준편차', color=viz.colors[3], alpha=0.7)
        
        plt.xlabel('성능 지표')
        plt.ylabel('점수')
        plt.title(f'성능 통계 요약 - {model_name}', fontsize=16, fontweight='bold')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 값 표시
        for i, metric in enumerate(metrics):
            plt.text(i-width*1.5, float(means[i])+0.01, f'{means[i]:.3f}', ha='center', va='bottom', fontsize=9)
            plt.text(i-width*0.5, float(maxs[i])+0.01, f'{maxs[i]:.3f}', ha='center', va='bottom', fontsize=9)
            plt.text(i+width*0.5, float(mins[i])+0.01, f'{mins[i]:.3f}', ha='center', va='bottom', fontsize=9)
            plt.text(i+width*1.5, float(stds[i])+0.01, f'{stds[i]:.3f}', ha='center', va='bottom', fontsize=9)
        
        viz.save_plot('05_performance_statistics.png')
        
        # 6. 학습 곡선 (히스토리 데이터가 있는 경우)
        if history_data and 'train_loss' in history_data and len(history_data['train_loss']) > 0:
            plt.figure(figsize=(12, 8))
            
            epochs = history_data.get('epochs', range(1, len(history_data['train_loss']) + 1))
            
            # 데이터 길이 검증
            if len(epochs) != len(history_data['train_loss']):
                epochs = range(1, len(history_data['train_loss']) + 1)
            
            # 2x2 서브플롯
            plt.subplot(2, 2, 1)
            plt.plot(epochs, history_data['train_loss'], 'o-', color=viz.colors[0], alpha=0.7, linewidth=2)
            plt.title('학습 손실')
            plt.xlabel('에포크')
            plt.ylabel('손실')
            plt.grid(True, alpha=0.3)
            
            if 'val_loss' in history_data:
                plt.subplot(2, 2, 2)
                plt.plot(epochs, history_data['val_loss'], 's-', color=viz.colors[1], alpha=0.7, linewidth=2)
                plt.title('검증 손실')
                plt.xlabel('에포크')
                plt.ylabel('손실')
                plt.grid(True, alpha=0.3)
            
            if 'val_f1' in history_data:
                plt.subplot(2, 2, 3)
                plt.plot(epochs, history_data['val_f1'], '^-', color=viz.colors[2], alpha=0.7, linewidth=2)
                plt.title('검증 F1 점수')
                plt.xlabel('에포크')
                plt.ylabel('F1 점수')
                plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            # 통계 정보
            stats_text = f"""학습 요약:
모델: {model_name}
평균 F1: {avg_f1:.4f}
최고 F1: {max(f1_scores) if f1_scores else 0:.4f}
표준편차: {np.std(f1_scores) if f1_scores else 0:.4f}
변동계수: {(np.std(f1_scores)/avg_f1)*100 if avg_f1 > 0 else 0:.2f}%"""
            plt.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            
            plt.suptitle(f'학습 기록 - {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            viz.save_plot('06_training_history.png')
            
            # 7. 손실 비교 차트 (학습 vs 검증)
            if 'val_loss' in history_data:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, history_data['train_loss'], 'o-', label='학습 손실', 
                        color=viz.colors[0], linewidth=2)
                plt.plot(epochs, history_data['val_loss'], 's-', label='검증 손실', 
                        color=viz.colors[1], linewidth=2)
                
                plt.title(f'학습 vs 검증 손실 비교 - {model_name}', fontsize=16, fontweight='bold')
                plt.xlabel('에포크')
                plt.ylabel('손실')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 과적합 감지 영역 표시
                if len(history_data['train_loss']) > 3:
                    train_trend = np.polyfit(epochs[-3:], history_data['train_loss'][-3:], 1)[0]
                    val_trend = np.polyfit(epochs[-3:], history_data['val_loss'][-3:], 1)[0]
                    
                    if train_trend < 0 and val_trend > 0:  # 과적합 징후
                        plt.axvspan(epochs[-3], epochs[-1], alpha=0.2, color='red', 
                                   label='과적합 위험 구간')
                        plt.legend()
                
                viz.save_plot('07_loss_comparison.png')
        
        print(f"✅ Training visualizations completed: {viz.images_dir}")
        print(f"📊 Generated {len(list(viz.images_dir.glob('*.png')))} training visualization images")
        
    except Exception as e:
        import traceback
        print(f"❌ Training visualization failed: {str(e)}")
        print(f"❌ Error details: {traceback.format_exc()}")

def visualize_training_pipeline(fold_results: Dict, model_name: str, output_dir: str, 
                               history_data: Optional[Dict] = None):
    """학습 파이프라인 시각화 통합 함수"""
    try:
        print(f"🎯 Starting training visualization for {model_name}")
        print(f"📊 fold_results keys: {list(fold_results.keys()) if isinstance(fold_results, dict) else 'not dict'}")
        print(f"📁 output_dir: {output_dir}")
        
        create_training_visualizations(fold_results, model_name, output_dir, history_data)
        
    except Exception as e:
        import traceback
        print(f"❌ visualize_training_pipeline failed: {str(e)}")
        print(f"❌ Full traceback: {traceback.format_exc()}")
