"""
학습 시각화 유틸리티
대화 요약 대회를 위한 학습 과정 및 결과 시각화
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class TrainingVisualizer:
    """학습 과정 및 결과 시각화 클래스"""

    def __init__(self, style='seaborn-v0_8', figsize=(12, 6)):
        """
        시각화 클래스 초기화

        Args:
            style: matplotlib 스타일
            figsize: 기본 figure 크기
        """
        # matplotlib 스타일 안전하게 설정
        try:
            plt.style.use(style)
        except OSError:
            # 스타일이 없으면 기본 스타일 또는 대체 스타일 사용
            try:
                plt.style.use('seaborn-v0_8')
            except:
                # seaborn 스타일이 없으면 기본 스타일 사용
                print("Using default matplotlib style")
                pass

        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """
        학습 히스토리 플롯

        Args:
            history: 학습 히스토리 딕셔너리
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Loss 플롯
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss', color=self.colors[0])
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss', color=self.colors[1])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ROUGE 스코어 플롯
        rouge_metrics = [k for k in history.keys() if 'rouge' in k.lower()]
        for i, metric in enumerate(rouge_metrics):
            axes[1].plot(history[metric], label=metric, color=self.colors[i % len(self.colors)])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ROUGE Score')
        axes[1].set_title('ROUGE Scores')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_rouge_comparison(self, results: Dict, save_path: Optional[str] = None):
        """
        모델별 ROUGE 스코어 비교

        Args:
            results: 모델별 ROUGE 스코어 딕셔너리
            save_path: 저장 경로
        """
        models = list(results.keys())
        rouge_1 = [results[m].get('rouge-1', 0) for m in models]
        rouge_2 = [results[m].get('rouge-2', 0) for m in models]
        rouge_l = [results[m].get('rouge-l', 0) for m in models]

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=self.figsize)

        bars1 = ax.bar(x - width, rouge_1, width, label='ROUGE-1', color=self.colors[0])
        bars2 = ax.bar(x, rouge_2, width, label='ROUGE-2', color=self.colors[1])
        bars3 = ax.bar(x + width, rouge_l, width, label='ROUGE-L', color=self.colors[2])

        ax.set_xlabel('Models')
        ax.set_ylabel('ROUGE Score')
        ax.set_title('ROUGE Scores Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 값 표시
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_kfold_results(self, fold_results: List[Dict], save_path: Optional[str] = None):
        """
        K-Fold 교차 검증 결과 시각화

        Args:
            fold_results: 각 fold의 결과 리스트
            save_path: 저장 경로
        """
        n_folds = len(fold_results)
        metrics = list(fold_results[0].keys())

        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))

        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            values = [fold[metric] for fold in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)

            # Box plot
            bp = axes[idx].boxplot(values, labels=['K-Fold'])
            axes[idx].scatter([1] * n_folds, values, alpha=0.6, s=50)

            # Mean line
            axes[idx].axhline(y=mean_val, color='r', linestyle='--',
                             label=f'Mean: {mean_val:.4f} ± {std_val:.4f}')

            axes[idx].set_title(f'{metric} across {n_folds} folds')
            axes[idx].set_ylabel(metric)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('K-Fold Cross Validation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_ensemble_weights(self, weights: Dict, save_path: Optional[str] = None):
        """
        앙상블 가중치 시각화

        Args:
            weights: 모델별 가중치 딕셔너리
            save_path: 저장 경로
        """
        models = list(weights.keys())
        values = list(weights.values())

        # 파이 차트
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 파이 차트
        colors_pie = self.colors[:len(models)]
        wedges, texts, autotexts = ax1.pie(values, labels=models, colors=colors_pie,
                                           autopct='%1.1f%%', startangle=90)
        ax1.set_title('Ensemble Weight Distribution (Pie)', fontweight='bold')

        # 바 차트
        bars = ax2.bar(models, values, color=colors_pie)
        ax2.set_ylabel('Weight')
        ax2.set_title('Ensemble Weight Distribution (Bar)', fontweight='bold')
        ax2.set_ylim(0, max(values) * 1.2)

        # 값 표시
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_learning_curves(self, train_scores: List, val_scores: List,
                           save_path: Optional[str] = None):
        """
        학습 곡선 플롯

        Args:
            train_scores: 학습 스코어 리스트
            val_scores: 검증 스코어 리스트
            save_path: 저장 경로
        """
        epochs = range(1, len(train_scores) + 1)

        plt.figure(figsize=self.figsize)

        plt.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
        plt.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)

        # 최고 성능 지점 표시
        best_val_epoch = np.argmax(val_scores) + 1
        best_val_score = max(val_scores)
        plt.scatter(best_val_epoch, best_val_score, color='red', s=100, zorder=5)
        plt.annotate(f'Best: {best_val_score:.4f}\n(Epoch {best_val_epoch})',
                    xy=(best_val_epoch, best_val_score),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true: List, y_pred: List,
                            labels: Optional[List] = None,
                            save_path: Optional[str] = None):
        """
        혼동 행렬 시각화

        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            labels: 레이블 이름
            save_path: 저장 경로
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_results_json(self, results: Dict, save_path: str):
        """
        결과를 JSON 파일로 저장

        Args:
            results: 결과 딕셔너리
            save_path: 저장 경로
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {save_path}")

    def create_report_figures(self, results: Dict, output_dir: str):
        """
        전체 리포트용 그림 생성

        Args:
            results: 전체 결과 딕셔너리
            output_dir: 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 학습 히스토리
        if 'history' in results:
            self.plot_training_history(
                results['history'],
                save_path=output_dir / 'training_history.png'
            )

        # 2. ROUGE 비교
        if 'model_scores' in results:
            self.plot_rouge_comparison(
                results['model_scores'],
                save_path=output_dir / 'rouge_comparison.png'
            )

        # 3. K-Fold 결과
        if 'kfold_results' in results:
            self.plot_kfold_results(
                results['kfold_results'],
                save_path=output_dir / 'kfold_results.png'
            )

        # 4. 앙상블 가중치
        if 'ensemble_weights' in results:
            self.plot_ensemble_weights(
                results['ensemble_weights'],
                save_path=output_dir / 'ensemble_weights.png'
            )

        print(f"All figures saved to {output_dir}")
