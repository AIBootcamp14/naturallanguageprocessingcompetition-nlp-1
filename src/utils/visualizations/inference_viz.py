#!/usr/bin/env python3
"""
추론 시각화 모듈
모델 추론 결과를 다양한 형태로 시각화
"""

# ------------------------- 표준 라이브러리 ------------------------- #
from typing import Optional

# ------------------------- 서드파티 라이브러리 ------------------------- #
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- 프로젝트 모듈 ------------------------- #
from .base_visualizer import SimpleVisualizer


# ==================== 추론 시각화 함수들 ==================== #
# ---------------------- 추론 결과 시각화 생성 ---------------------- #
def create_inference_visualizations(predictions: np.ndarray, model_name: str, output_dir: str,
                                  confidence_scores: Optional[np.ndarray] = None):
    """추론 결과 시각화 생성

    7개의 다양한 시각화를 생성하여 추론 결과를 분석함

    Args:
        predictions: 예측 결과 배열
        model_name: 모델 이름
        output_dir: 출력 디렉토리 경로
        confidence_scores: 신뢰도 점수 배열 (선택)
    """
    # 시각화 객체 생성
    viz = SimpleVisualizer(output_dir, model_name)

    # -------------- 예측 결과 처리 및 시각화 -------------- #
    # 시각화 실행 시도
    try:
        # -------------- 예측값 형태 확인 및 변환 -------------- #
        # 2차원 배열인 경우 (확률 형태)
        if predictions.ndim == 2:
            pred_classes = np.argmax(predictions, axis=1)  # 최대 확률 클래스 추출
            confidences = np.max(predictions, axis=1)      # 최대 확률값 추출
            class_probs = predictions                      # 전체 확률 분포 저장

        # 1차원 배열인 경우 (클래스 인덱스)
        else:
            pred_classes = predictions                                                       # 클래스 그대로 사용
            confidences = confidence_scores if confidence_scores is not None else np.ones_like(predictions)  # 신뢰도 설정
            class_probs = None                                                               # 확률 분포 없음

        # -------------- 클래스별 예측 빈도 계산 -------------- #
        # 고유 클래스 및 개수 추출
        unique, counts = np.unique(pred_classes, return_counts=True)

        # ==================== 시각화 1: 클래스별 예측 분포 ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(10, 6))

        # -------------- 막대 그래프 그리기 -------------- #
        # 클래스별 예측 수 막대 그래프
        bars = plt.bar(unique, counts, color=viz.colors[:len(unique)], alpha=0.7)

        # 그래프 제목 및 축 라벨
        plt.title(f'클래스별 예측 분포 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('클래스')
        plt.ylabel('예측 개수')

        # -------------- 백분율 표시 -------------- #
        # 각 막대 위에 개수와 백분율 표시
        total = len(pred_classes)  # 전체 예측 수

        # 각 클래스별 표시
        for i, (cls, count) in enumerate(zip(unique, counts)):
            percentage = (count / total) * 100  # 백분율 계산
            plt.text(cls, count + total*0.01, f'{count}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

        # 그리드 및 저장
        plt.grid(axis='y', alpha=0.3)
        viz.save_plot('01_class_distribution.png')

        # ==================== 시각화 2: 신뢰도 분포 히스토그램 ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(10, 6))

        # -------------- 히스토그램 그리기 -------------- #
        # 신뢰도 분포 히스토그램
        plt.hist(confidences, bins=30, color='skyblue', alpha=0.7, edgecolor='black')

        # -------------- 통계선 표시 -------------- #
        # 평균 및 중간값 계산
        mean_conf = np.mean(confidences)      # 평균 신뢰도
        median_conf = np.median(confidences)  # 중간값 신뢰도

        # 통계선 그리기
        plt.axvline(float(mean_conf), color='red', linestyle='--', alpha=0.8,
                   label=f'평균: {mean_conf:.3f}')  # 평균선
        plt.axvline(float(median_conf), color='green', linestyle='--', alpha=0.8,
                   label=f'중간값: {median_conf:.3f}')  # 중간값선

        # 그래프 꾸미기
        plt.title(f'신뢰도 분포 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('신뢰도 점수')
        plt.ylabel('빈도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        viz.save_plot('02_confidence_distribution.png')

        # ==================== 시각화 3: 클래스별 평균 신뢰도 비교 ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(10, 6))

        # -------------- 클래스별 신뢰도 통계 계산 -------------- #
        # 클래스별 데이터 저장
        class_confidences = []  # 평균 신뢰도
        class_labels = []       # 클래스 라벨
        class_stds = []         # 표준편차

        # 각 클래스별 통계 계산
        for cls in unique:
            mask = pred_classes == cls                # 해당 클래스 마스크
            avg_conf = np.mean(confidences[mask])     # 평균 신뢰도
            std_conf = np.std(confidences[mask])      # 표준편차
            class_confidences.append(avg_conf)        # 저장
            class_stds.append(std_conf)               # 저장
            class_labels.append(f'클래스 {cls}')      # 라벨 생성

        # -------------- 막대 그래프 그리기 -------------- #
        # 평균 신뢰도 막대 그래프 (에러바 포함)
        bars = plt.bar(class_labels, class_confidences,
                      color=viz.colors[:len(class_labels)], alpha=0.7,
                      yerr=class_stds, capsize=5)  # 에러바 표시

        # 그래프 꾸미기
        plt.title(f'클래스별 평균 신뢰도 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('클래스')
        plt.ylabel('평균 신뢰도')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # -------------- 값 표시 -------------- #
        # 각 막대 위에 평균±표준편차 표시
        for bar, conf, std in zip(bars, class_confidences, class_stds):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                    f'{conf:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

        # Figure 저장
        viz.save_plot('03_class_confidence_comparison.png')

        # ==================== 시각화 4: 신뢰도 구간별 예측 분포 ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(12, 6))

        # -------------- 신뢰도 구간 정의 -------------- #
        # 구간 경계 및 라벨 정의
        confidence_bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]  # 구간 경계
        bin_labels = ['매우낮음\n(0-0.5)', '낮음\n(0.5-0.7)', '보통\n(0.7-0.8)',
                     '높음\n(0.8-0.9)', '매우높음\n(0.9-0.95)', '확실\n(0.95-1.0)']  # 라벨

        # -------------- 각 구간별 개수 계산 -------------- #
        # 구간별 개수 저장
        bin_counts = []

        # 각 구간 순회
        for i in range(len(confidence_bins)-1):
            # 구간 범위 마스크
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])

            # 마지막 구간은 상한 포함
            if i == len(confidence_bins)-2:
                mask = (confidences >= confidence_bins[i]) & (confidences <= confidence_bins[i+1])

            # 개수 저장
            bin_counts.append(np.sum(mask))

        # -------------- 막대 그래프 그리기 -------------- #
        # 구간별 색상 정의
        colors = ['#FF6B6B', '#FFA726', '#FFCC02', '#66BB6A', '#42A5F5', '#AB47BC']

        # 막대 그래프 생성
        bars = plt.bar(bin_labels, bin_counts, color=colors, alpha=0.7)

        # 그래프 꾸미기
        plt.title(f'신뢰도 구간별 예측 분포 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('신뢰도 구간')
        plt.ylabel('예측 개수')

        # -------------- 백분율 표시 -------------- #
        # 각 막대 위에 개수와 백분율 표시
        for bar, count in zip(bars, bin_counts):
            percentage = (count / total) * 100  # 백분율 계산
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + total*0.01,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')

        # 그리드 및 저장
        plt.grid(axis='y', alpha=0.3)
        viz.save_plot('04_confidence_bins.png')

        # ==================== 시각화 5: 클래스별 신뢰도 분포 (박스플롯) ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(12, 6))

        # -------------- 클래스별 신뢰도 데이터 추출 -------------- #
        # 각 클래스별 신뢰도 배열 수집
        confidence_by_class = []

        # 각 클래스별 신뢰도 추출
        for cls in unique:
            mask = pred_classes == cls           # 해당 클래스 마스크
            confidence_by_class.append(confidences[mask])  # 신뢰도 배열 추가

        # -------------- 박스플롯 그리기 -------------- #
        # 박스플롯 생성
        bp = plt.boxplot(confidence_by_class, patch_artist=True)

        # x축 라벨 설정
        plt.xticks(range(1, len(unique)+1), [f'클래스 {cls}' for cls in unique])

        # 그래프 꾸미기
        plt.title(f'클래스별 신뢰도 분포 (박스플롯) - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('클래스')
        plt.ylabel('신뢰도')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Figure 저장
        viz.save_plot('05_confidence_boxplot.png')

        # ==================== 시각화 6: 종합 추론 분석 ==================== #
        # -------------- Figure 생성 (2x2 레이아웃) -------------- #
        plt.figure(figsize=(15, 10))

        # -------------- 좌상단: 클래스 분포 파이차트 -------------- #
        plt.subplot(2, 2, 1)

        # 파이차트 그리기
        plt.pie(counts, labels=[f'클래스 {cls}' for cls in unique], autopct='%1.1f%%',
               colors=viz.colors[:len(unique)], startangle=90)

        # 제목
        plt.title('클래스 비율')

        # -------------- 우상단: 신뢰도 히스토그램 (간소화) -------------- #
        plt.subplot(2, 2, 2)

        # 히스토그램 그리기
        plt.hist(confidences, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(mean_conf, color='red', linestyle='--', label=f'평균: {mean_conf:.3f}')  # 평균선

        # 그래프 꾸미기
        plt.title('신뢰도 분포')
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        plt.legend()

        # -------------- 좌하단: 클래스별 평균 신뢰도 -------------- #
        plt.subplot(2, 2, 3)

        # 막대 그래프 그리기
        plt.bar(range(len(unique)), class_confidences,
               color=viz.colors[:len(unique)], alpha=0.7)

        # 그래프 꾸미기
        plt.title('클래스별 평균 신뢰도')
        plt.xlabel('클래스')
        plt.ylabel('평균 신뢰도')
        plt.xticks(range(len(unique)), [f'C{cls}' for cls in unique])

        # -------------- 우하단: 통계 요약 -------------- #
        plt.subplot(2, 2, 4)

        # -------------- 통계 텍스트 생성 -------------- #
        # 통계 요약 문자열 생성
        stats_text = f"""추론 통계 요약:
총 예측 샘플: {len(pred_classes):,}개
고유 클래스: {len(unique)}개
평균 신뢰도: {mean_conf:.3f}
신뢰도 표준편차: {np.std(confidences):.3f}
높은 신뢰도(>0.9): {np.sum(confidences > 0.9):,}개 ({np.sum(confidences > 0.9)/len(confidences)*100:.1f}%)
낮은 신뢰도(<0.5): {np.sum(confidences < 0.5):,}개 ({np.sum(confidences < 0.5)/len(confidences)*100:.1f}%)"""

        # 텍스트 박스로 표시
        plt.text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
                transform=plt.gca().transAxes)

        # 축 숨기기
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        # 전체 제목 및 저장
        plt.suptitle(f'종합 추론 분석 - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        viz.save_plot('06_inference_summary.png')

        # ==================== 시각화 7: 클래스 확률 분포 히트맵 ==================== #
        # -------------- 확률 예측인 경우 히트맵 생성 -------------- #
        # 확률 분포가 있고 다중 클래스인 경우
        if class_probs is not None and class_probs.shape[1] > 1:
            # Figure 생성
            plt.figure(figsize=(12, 8))

            # -------------- 샘플링 (시각화 최적화) -------------- #
            # 최대 1000개 샘플로 제한
            sample_size = min(1000, class_probs.shape[0])  # 샘플 크기
            sample_indices = np.random.choice(class_probs.shape[0], sample_size, replace=False)  # 무작위 샘플링
            sample_probs = class_probs[sample_indices]  # 샘플 확률

            # -------------- 히트맵 생성 -------------- #
            # 히트맵 그리기 (전치하여 클래스를 y축으로)
            plt.imshow(sample_probs.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            plt.colorbar(label='확률')  # 컬러바 추가

            # 그래프 꾸미기
            plt.title(f'클래스별 확률 분포 히트맵 - {model_name}', fontsize=16, fontweight='bold')
            plt.xlabel('샘플 인덱스')
            plt.ylabel('클래스')
            plt.yticks(range(class_probs.shape[1]), [f'클래스 {i}' for i in range(class_probs.shape[1])])

            # Figure 저장
            viz.save_plot('07_probability_heatmap.png')

        # -------------- 시각화 완료 메시지 -------------- #
        # 완료 정보 출력
        print(f"✅ Inference visualizations completed: {viz.images_dir}")
        print(f"📊 Generated {len(list(viz.images_dir.glob('*.png')))} inference visualization images")

    # -------------- 예외 발생 시 에러 처리 -------------- #
    # 시각화 실패 시 에러 메시지 출력
    except Exception as e:
        print(f"❌ Inference visualization failed: {str(e)}")


# ==================== 파이프라인 호출 함수 ==================== #
# ---------------------- 추론 파이프라인 시각화 호출 ---------------------- #
def visualize_inference_pipeline(predictions: np.ndarray, model_name: str, output_dir: str,
                               confidence_scores: Optional[np.ndarray] = None):
    """추론 파이프라인 시각화 호출

    Args:
        predictions: 예측 결과 배열
        model_name: 모델 이름
        output_dir: 출력 디렉토리 경로
        confidence_scores: 신뢰도 점수 배열 (선택)
    """
    # 추론 시각화 함수 호출
    create_inference_visualizations(predictions, model_name, output_dir, confidence_scores)
