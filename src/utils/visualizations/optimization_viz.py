#!/usr/bin/env python3
"""
최적화 시각화 모듈
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from .base_visualizer import SimpleVisualizer

def create_optimization_visualizations(study_path: str, model_name: str, output_dir: str):
    """최적화 결과 시각화 - 6개의 다양한 시각화 생성"""
    viz = SimpleVisualizer(output_dir, model_name)
    
    try:
        import pickle
        import optuna
        
        # Study 로드
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        
        trials = study.trials
        values = [t.value for t in trials if t.value is not None]
        
        if len(values) < 3:
            print("Not enough trials for visualization")
            return
        
        # 1. 최적화 진행 히스토리
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(values)), values, 'o-', color=viz.colors[0], alpha=0.7, linewidth=2)
        
        # 최고 성능 표시
        best_idx = np.argmax(values)
        plt.scatter(best_idx, values[best_idx], color='red', s=150, zorder=5, 
                   label=f'최고 성능: {values[best_idx]:.4f}')
        
        # 추세선 추가
        z = np.polyfit(range(len(values)), values, 1)
        p = np.poly1d(z)
        plt.plot(range(len(values)), p(range(len(values))), "--", alpha=0.8, color='gray', 
                label=f'추세: {"상승" if z[0] > 0 else "하락"}')
        
        plt.title(f'최적화 진행 히스토리 - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('시행 번호')
        plt.ylabel('목적함수 값 (F1 점수)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 성능 개선 구간 하이라이트
        if len(values) > 10:
            improvement_points = []
            current_max = values[0]
            for i, val in enumerate(values):
                if val > current_max:
                    improvement_points.append(i)
                    current_max = val
            
            if improvement_points:
                plt.scatter(improvement_points, [values[i] for i in improvement_points], 
                           color='green', s=80, alpha=0.7, marker='^', label='개선 지점')
                plt.legend()
        
        plt.subplot(1, 2, 2)
        # 성능 분포
        plt.hist(values, bins=20, color=viz.colors[1], alpha=0.7, edgecolor='black')
        mean_val = np.mean(values)
        max_val = np.max(values)
        plt.axvline(float(mean_val), color='red', linestyle='--', alpha=0.8, 
                   label=f'평균: {mean_val:.4f}')
        plt.axvline(float(max_val), color='green', linestyle='--', alpha=0.8, 
                   label=f'최고: {max_val:.4f}')
        plt.title('성능 분포')
        plt.xlabel('F1 점수')
        plt.ylabel('빈도')
        plt.legend()
        
        plt.tight_layout()
        viz.save_plot('01_optimization_progress.png')
        
        # 2. 누적 최고 성능 기록
        plt.figure(figsize=(10, 6))
        best_values = []
        current_best = -float('inf')
        for val in values:
            if val > current_best:
                current_best = val
            best_values.append(current_best)
        
        plt.plot(range(len(best_values)), best_values, 'o-', color=viz.colors[2], 
                alpha=0.7, linewidth=2, markersize=4)
        
        # 개선 구간 표시
        improvements = np.diff(best_values)
        improvement_indices = np.where(improvements > 0)[0] + 1
        
        if len(improvement_indices) > 0:
            plt.scatter(improvement_indices, [best_values[i] for i in improvement_indices], 
                       color='red', s=80, zorder=5, label='성능 개선')
        
        plt.title(f'누적 최고 성능 기록 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('시행 번호')
        plt.ylabel('현재까지 최고 F1 점수')
        plt.grid(True, alpha=0.3)
        
        # 최종 개선폭 표시
        total_improvement = best_values[-1] - best_values[0]
        plt.text(0.02, 0.98, f'총 개선폭: {total_improvement:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                verticalalignment='top')
        
        if len(improvement_indices) > 0:
            plt.legend()
        viz.save_plot('02_cumulative_best.png')
        
        # 3. 하이퍼파라미터 중요도 (가능한 경우)
        try:
            importance = optuna.importance.get_param_importances(study)
            if importance:
                plt.figure(figsize=(10, 6))
                
                params = list(importance.keys())
                importances = list(importance.values())
                
                # 중요도순으로 정렬
                sorted_indices = np.argsort(importances)[::-1]
                params = [params[i] for i in sorted_indices]
                importances = [importances[i] for i in sorted_indices]
                
                bars = plt.barh(params, importances, color=viz.colors[:len(params)], alpha=0.7)
                
                # 값 표시
                for bar, imp in zip(bars, importances):
                    plt.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2,
                            f'{imp:.3f}', ha='left', va='center', fontweight='bold')
                
                plt.title(f'하이퍼파라미터 중요도 - {model_name}', fontsize=16, fontweight='bold')
                plt.xlabel('중요도')
                plt.ylabel('파라미터')
                plt.grid(axis='x', alpha=0.3)
                viz.save_plot('03_parameter_importance.png')
        except Exception as e:
            print(f"Parameter importance visualization skipped: {e}")
        
        # 4. 상위 성능 시행 비교
        plt.figure(figsize=(12, 6))
        
        # 상위 5개 시행 선택
        top_n = min(5, len(values))
        top_indices = np.argsort(values)[-top_n:][::-1]
        
        plt.subplot(1, 2, 1)
        top_values = [values[i] for i in top_indices]
        trial_labels = [f'Trial {i}' for i in top_indices]
        
        bars = plt.bar(trial_labels, top_values, color=viz.colors[:top_n], alpha=0.7)
        
        # 값 표시
        for bar, val in zip(bars, top_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(top_values)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'상위 {top_n}개 시행 성능')
        plt.ylabel('F1 점수')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # 성능 범위별 분포
        performance_ranges = []
        range_labels = []
        
        min_val, max_val = min(values), max(values)
        range_size = (max_val - min_val) / 5
        
        for i in range(5):
            lower = min_val + i * range_size
            upper = min_val + (i + 1) * range_size
            count = sum(1 for v in values if lower <= v < upper)
            if i == 4:  # 마지막 구간은 상한 포함
                count = sum(1 for v in values if lower <= v <= upper)
            performance_ranges.append(count)
            range_labels.append(f'{lower:.3f}-{upper:.3f}')
        
        plt.bar(range_labels, performance_ranges, color=viz.colors[:5], alpha=0.7)
        plt.title('성능 구간별 분포')
        plt.xlabel('F1 점수 구간')
        plt.ylabel('시행 수')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'상위 성능 분석 - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        viz.save_plot('04_top_trials_analysis.png')
        
        # 5. 종합 최적화 통계
        plt.figure(figsize=(14, 10))
        
        # 좌상단: 시행별 성능 변화
        plt.subplot(2, 2, 1)
        plt.plot(range(len(values)), values, 'o-', color=viz.colors[0], alpha=0.7, markersize=3)
        plt.axhline(float(mean_val), color='red', linestyle='--', alpha=0.7, label='평균')
        plt.title('시행별 성능 변화')
        plt.xlabel('시행 번호')
        plt.ylabel('F1 점수')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 우상단: 성능 분포 히스토그램
        plt.subplot(2, 2, 2)
        plt.hist(values, bins=15, color=viz.colors[1], alpha=0.7, edgecolor='black')
        plt.axvline(float(mean_val), color='red', linestyle='--', label=f'평균: {mean_val:.4f}')
        plt.title('성능 분포')
        plt.xlabel('F1 점수')
        plt.ylabel('빈도')
        plt.legend()
        
        # 좌하단: 누적 최고 성능
        plt.subplot(2, 2, 3)
        plt.plot(range(len(best_values)), best_values, 's-', color=viz.colors[2], 
                alpha=0.7, markersize=3)
        plt.title('누적 최고 성능')
        plt.xlabel('시행 번호')
        plt.ylabel('최고 F1 점수')
        plt.grid(True, alpha=0.3)
        
        # 우하단: 통계 요약
        plt.subplot(2, 2, 4)
        stats_text = f"""최적화 통계 요약:
총 시행 수: {len(trials)}
완료된 시행: {len(values)}
최고 F1: {max(values):.4f}
평균 F1: {mean_val:.4f}
표준편차: {np.std(values):.4f}
성능 향상: {total_improvement:.4f}
개선 횟수: {len(improvement_indices)}회
수렴도: {(np.std(values[-10:]) if len(values) >= 10 else np.std(values)):.4f}"""
        
        plt.text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                transform=plt.gca().transAxes)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.suptitle(f'종합 최적화 결과 - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        viz.save_plot('05_optimization_summary.png')
        
        # 6. 최적화 수렴 분석
        if len(values) >= 10:
            plt.figure(figsize=(12, 6))
            
            # 이동평균으로 수렴 패턴 분석
            window_size = min(10, len(values) // 3)
            moving_avg = []
            moving_std = []
            
            for i in range(window_size, len(values) + 1):
                window_values = values[i-window_size:i]
                moving_avg.append(np.mean(window_values))
                moving_std.append(np.std(window_values))
            
            plt.subplot(1, 2, 1)
            x_coords = range(window_size, len(values) + 1)
            plt.plot(x_coords, moving_avg, 'o-', color=viz.colors[0], alpha=0.7, 
                    label=f'이동평균 (window={window_size})')
            plt.fill_between(x_coords, 
                           np.array(moving_avg) - np.array(moving_std),
                           np.array(moving_avg) + np.array(moving_std),
                           alpha=0.3, color=viz.colors[0])
            plt.title('성능 수렴 패턴')
            plt.xlabel('시행 번호')
            plt.ylabel('이동평균 F1 점수')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(x_coords, moving_std, 's-', color=viz.colors[1], alpha=0.7)
            plt.title('성능 변동성 변화')
            plt.xlabel('시행 번호')
            plt.ylabel('이동 표준편차')
            plt.grid(True, alpha=0.3)
            
            # 수렴 판정
            recent_std = moving_std[-3:] if len(moving_std) >= 3 else moving_std
            convergence_status = "수렴" if np.mean(recent_std) < 0.01 else "진행중"
            plt.text(0.02, 0.98, f'수렴 상태: {convergence_status}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor="lightgreen" if convergence_status == "수렴" else "lightyellow", 
                             alpha=0.7),
                    verticalalignment='top')
            
            plt.suptitle(f'최적화 수렴 분석 - {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            viz.save_plot('06_convergence_analysis.png')
        
        print(f"✅ Optimization visualizations completed: {viz.images_dir}")
        print(f"📊 Generated {len(list(viz.images_dir.glob('*.png')))} optimization visualization images")
        
    except Exception as e:
        print(f"❌ Optimization visualization failed: {str(e)}")

def visualize_optimization_pipeline(study_path: str, model_name: str, output_dir: str,
                                  experiment_name: Optional[str] = None):
    """최적화 파이프라인 시각화 호출"""
    create_optimization_visualizations(study_path, model_name, output_dir)
