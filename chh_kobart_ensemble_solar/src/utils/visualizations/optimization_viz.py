#!/usr/bin/env python3
"""
최적화 시각화 모듈
Optuna 최적화 결과를 다양한 형태로 시각화
"""

# ------------------------- 표준 라이브러리 ------------------------- #
from typing import Dict, Optional

# ------------------------- 서드파티 라이브러리 ------------------------- #
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- 프로젝트 모듈 ------------------------- #
from .base_visualizer import SimpleVisualizer


# ==================== 최적화 시각화 함수들 ==================== #
# ---------------------- 최적화 결과 시각화 생성 ---------------------- #
def create_optimization_visualizations(study_path: str, model_name: str, output_dir: str):
    """최적화 결과 시각화 생성

    6개의 다양한 시각화를 생성하여 최적화 과정을 분석함

    Args:
        study_path: Optuna Study 파일 경로
        model_name: 모델 이름
        output_dir: 출력 디렉토리 경로
    """
    # 시각화 객체 생성
    viz = SimpleVisualizer(output_dir, model_name)

    # -------------- Study 파일 로드 및 데이터 추출 -------------- #
    # Study 로드 시도
    try:
        import pickle
        import optuna

        # -------------- Study 파일 로드 -------------- #
        # pickle 파일에서 Study 로드
        with open(study_path, 'rb') as f:
            study = pickle.load(f)

        # -------------- 시행 데이터 추출 -------------- #
        # 모든 시행 및 성능값 추출
        trials = study.trials
        values = [t.value for t in trials if t.value is not None]  # None이 아닌 값만 추출

        # -------------- 데이터 검증 -------------- #
        # 시각화에 충분한 데이터 확인
        if len(values) < 3:  # 최소 3개 이상 필요
            print("Not enough trials for visualization")
            return

        # ==================== 시각화 1: 최적화 진행 히스토리 ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(12, 6))

        # -------------- 좌측: 시행별 성능 변화 -------------- #
        plt.subplot(1, 2, 1)

        # 성능 변화 그래프 그리기
        plt.plot(range(len(values)), values, 'o-', color=viz.colors[0], alpha=0.7, linewidth=2)  # 시행별 성능

        # -------------- 최고 성능 표시 -------------- #
        # 최고 성능 지점 찾기
        best_idx = np.argmax(values)  # 최고 성능 인덱스
        plt.scatter(best_idx, values[best_idx], color='red', s=150, zorder=5,
                   label=f'최고 성능: {values[best_idx]:.4f}')  # 최고 성능 강조

        # -------------- 추세선 추가 -------------- #
        # 1차 추세선 계산 및 표시
        z = np.polyfit(range(len(values)), values, 1)  # 1차 다항식 피팅
        p = np.poly1d(z)  # 다항식 객체 생성
        plt.plot(range(len(values)), p(range(len(values))), "--", alpha=0.8, color='gray',
                label=f'추세: {"상승" if z[0] > 0 else "하락"}')  # 추세선 표시

        # 그래프 꾸미기
        plt.title(f'최적화 진행 히스토리 - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('시행 번호')
        plt.ylabel('목적함수 값 (F1 점수)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # -------------- 성능 개선 구간 하이라이트 -------------- #
        # 성능 개선 지점 찾기 (10개 이상 시행 시)
        if len(values) > 10:
            # 개선 지점 탐색
            improvement_points = []  # 개선 지점 인덱스 저장
            current_max = values[0]  # 현재까지 최고 성능

            # 각 시행별 개선 여부 확인
            for i, val in enumerate(values):
                # 현재 성능이 이전 최고보다 높으면
                if val > current_max:
                    improvement_points.append(i)  # 개선 지점 추가
                    current_max = val  # 최고 성능 갱신

            # 개선 지점 표시
            if improvement_points:
                plt.scatter(improvement_points, [values[i] for i in improvement_points],
                           color='green', s=80, alpha=0.7, marker='^', label='개선 지점')
                plt.legend()

        # -------------- 우측: 성능 분포 히스토그램 -------------- #
        plt.subplot(1, 2, 2)

        # 성능 분포 히스토그램
        plt.hist(values, bins=20, color=viz.colors[1], alpha=0.7, edgecolor='black')  # 분포 표시

        # -------------- 통계선 표시 -------------- #
        # 평균 및 최고 성능선
        mean_val = np.mean(values)  # 평균 성능
        max_val = np.max(values)    # 최고 성능
        plt.axvline(float(mean_val), color='red', linestyle='--', alpha=0.8,
                   label=f'평균: {mean_val:.4f}')  # 평균선
        plt.axvline(float(max_val), color='green', linestyle='--', alpha=0.8,
                   label=f'최고: {max_val:.4f}')  # 최고선

        # 그래프 꾸미기
        plt.title('성능 분포')
        plt.xlabel('F1 점수')
        plt.ylabel('빈도')
        plt.legend()

        # Figure 저장
        plt.tight_layout()
        viz.save_plot('01_optimization_progress.png')

        # ==================== 시각화 2: 누적 최고 성능 기록 ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(10, 6))

        # -------------- 누적 최고 성능 계산 -------------- #
        # 각 시행까지의 최고 성능 추적
        best_values = []  # 누적 최고 성능 저장
        current_best = -float('inf')  # 현재 최고 성능 초기화

        # 각 시행별 누적 최고 성능 계산
        for val in values:
            # 더 높은 성능 발견 시 갱신
            if val > current_best:
                current_best = val
            best_values.append(current_best)  # 현재까지 최고 성능 저장

        # -------------- 누적 최고 성능 그래프 -------------- #
        # 누적 최고 성능 선 그래프
        plt.plot(range(len(best_values)), best_values, 'o-', color=viz.colors[2],
                alpha=0.7, linewidth=2, markersize=4)

        # -------------- 개선 구간 표시 -------------- #
        # 성능 개선 지점 계산
        improvements = np.diff(best_values)  # 성능 차이 계산
        improvement_indices = np.where(improvements > 0)[0] + 1  # 개선 지점 인덱스

        # 개선 지점 강조
        if len(improvement_indices) > 0:
            plt.scatter(improvement_indices, [best_values[i] for i in improvement_indices],
                       color='red', s=80, zorder=5, label='성능 개선')

        # 그래프 꾸미기
        plt.title(f'누적 최고 성능 기록 - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('시행 번호')
        plt.ylabel('현재까지 최고 F1 점수')
        plt.grid(True, alpha=0.3)

        # -------------- 총 개선폭 표시 -------------- #
        # 처음과 마지막 성능 차이 계산
        total_improvement = best_values[-1] - best_values[0]  # 총 개선폭
        plt.text(0.02, 0.98, f'총 개선폭: {total_improvement:.4f}',
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                verticalalignment='top')

        # 범례 표시
        if len(improvement_indices) > 0:
            plt.legend()

        # Figure 저장
        viz.save_plot('02_cumulative_best.png')

        # ==================== 시각화 3: 하이퍼파라미터 중요도 ==================== #
        # -------------- 하이퍼파라미터 중요도 계산 시도 -------------- #
        # 중요도 분석 시도
        try:
            # Optuna 중요도 분석
            importance = optuna.importance.get_param_importances(study)

            # -------------- 중요도 시각화 -------------- #
            # 중요도 데이터가 있는 경우
            if importance:
                # Figure 생성
                plt.figure(figsize=(10, 6))

                # -------------- 중요도 데이터 정렬 -------------- #
                # 파라미터 및 중요도 추출
                params = list(importance.keys())          # 파라미터 이름
                importances = list(importance.values())   # 중요도 값

                # 중요도 순으로 정렬
                sorted_indices = np.argsort(importances)[::-1]  # 내림차순 인덱스
                params = [params[i] for i in sorted_indices]          # 파라미터 정렬
                importances = [importances[i] for i in sorted_indices]  # 중요도 정렬

                # -------------- 가로 막대 그래프 -------------- #
                # 중요도 막대 그래프
                bars = plt.barh(params, importances, color=viz.colors[:len(params)], alpha=0.7)

                # -------------- 값 표시 -------------- #
                # 각 막대에 중요도 값 표시
                for bar, imp in zip(bars, importances):
                    plt.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2,
                            f'{imp:.3f}', ha='left', va='center', fontweight='bold')

                # 그래프 꾸미기
                plt.title(f'하이퍼파라미터 중요도 - {model_name}', fontsize=16, fontweight='bold')
                plt.xlabel('중요도')
                plt.ylabel('파라미터')
                plt.grid(axis='x', alpha=0.3)

                # Figure 저장
                viz.save_plot('03_parameter_importance.png')

        # 중요도 계산 실패 시 처리
        except Exception as e:
            print(f"Parameter importance visualization skipped: {e}")

        # ==================== 시각화 4: 상위 성능 시행 비교 ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(12, 6))

        # -------------- 상위 시행 선택 -------------- #
        # 상위 5개 시행 인덱스 추출
        top_n = min(5, len(values))  # 최대 5개
        top_indices = np.argsort(values)[-top_n:][::-1]  # 상위 N개 인덱스

        # -------------- 좌측: 상위 시행 막대 그래프 -------------- #
        plt.subplot(1, 2, 1)

        # 상위 시행 데이터 추출
        top_values = [values[i] for i in top_indices]      # 상위 시행 성능
        trial_labels = [f'Trial {i}' for i in top_indices]  # 시행 라벨

        # 막대 그래프 그리기
        bars = plt.bar(trial_labels, top_values, color=viz.colors[:top_n], alpha=0.7)

        # -------------- 값 표시 -------------- #
        # 각 막대 위에 성능 값 표시
        for bar, val in zip(bars, top_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(top_values)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

        # 그래프 꾸미기
        plt.title(f'상위 {top_n}개 시행 성능')
        plt.ylabel('F1 점수')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # -------------- 우측: 성능 구간별 분포 -------------- #
        plt.subplot(1, 2, 2)

        # -------------- 성능 구간 정의 -------------- #
        # 구간별 시행 수 계산
        performance_ranges = []  # 각 구간 시행 수
        range_labels = []        # 구간 라벨

        # 최소/최대 성능 및 구간 크기
        min_val, max_val = min(values), max(values)  # 성능 범위
        range_size = (max_val - min_val) / 5         # 5개 구간으로 분할

        # -------------- 각 구간별 시행 수 계산 -------------- #
        # 5개 구간 순회
        for i in range(5):
            # 구간 범위 계산
            lower = min_val + i * range_size          # 하한
            upper = min_val + (i + 1) * range_size    # 상한

            # 구간 내 시행 수 계산
            if i == 4:  # 마지막 구간은 상한 포함
                count = sum(1 for v in values if lower <= v <= upper)
            else:
                count = sum(1 for v in values if lower <= v < upper)

            # 데이터 저장
            performance_ranges.append(count)                       # 시행 수 저장
            range_labels.append(f'{lower:.3f}-{upper:.3f}')  # 라벨 저장

        # -------------- 구간별 분포 막대 그래프 -------------- #
        # 막대 그래프 그리기
        plt.bar(range_labels, performance_ranges, color=viz.colors[:5], alpha=0.7)

        # 그래프 꾸미기
        plt.title('성능 구간별 분포')
        plt.xlabel('F1 점수 구간')
        plt.ylabel('시행 수')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # 전체 제목 및 저장
        plt.suptitle(f'상위 성능 분석 - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        viz.save_plot('04_top_trials_analysis.png')

        # ==================== 시각화 5: 종합 최적화 통계 ==================== #
        # -------------- Figure 생성 -------------- #
        plt.figure(figsize=(14, 10))

        # -------------- 좌상단: 시행별 성능 변화 -------------- #
        plt.subplot(2, 2, 1)

        # 성능 변화 그래프
        plt.plot(range(len(values)), values, 'o-', color=viz.colors[0], alpha=0.7, markersize=3)
        plt.axhline(float(mean_val), color='red', linestyle='--', alpha=0.7, label='평균')  # 평균선

        # 그래프 꾸미기
        plt.title('시행별 성능 변화')
        plt.xlabel('시행 번호')
        plt.ylabel('F1 점수')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # -------------- 우상단: 성능 분포 히스토그램 -------------- #
        plt.subplot(2, 2, 2)

        # 히스토그램 그리기
        plt.hist(values, bins=15, color=viz.colors[1], alpha=0.7, edgecolor='black')
        plt.axvline(float(mean_val), color='red', linestyle='--', label=f'평균: {mean_val:.4f}')  # 평균선

        # 그래프 꾸미기
        plt.title('성능 분포')
        plt.xlabel('F1 점수')
        plt.ylabel('빈도')
        plt.legend()

        # -------------- 좌하단: 누적 최고 성능 -------------- #
        plt.subplot(2, 2, 3)

        # 누적 최고 성능 그래프
        plt.plot(range(len(best_values)), best_values, 's-', color=viz.colors[2],
                alpha=0.7, markersize=3)

        # 그래프 꾸미기
        plt.title('누적 최고 성능')
        plt.xlabel('시행 번호')
        plt.ylabel('최고 F1 점수')
        plt.grid(True, alpha=0.3)

        # -------------- 우하단: 통계 요약 -------------- #
        plt.subplot(2, 2, 4)

        # -------------- 통계 텍스트 생성 -------------- #
        # 통계 요약 문자열 생성
        stats_text = f"""최적화 통계 요약:
총 시행 수: {len(trials)}
완료된 시행: {len(values)}
최고 F1: {max(values):.4f}
평균 F1: {mean_val:.4f}
표준편차: {np.std(values):.4f}
성능 향상: {total_improvement:.4f}
개선 횟수: {len(improvement_indices)}회
수렴도: {(np.std(values[-10:]) if len(values) >= 10 else np.std(values)):.4f}"""

        # 텍스트 박스로 표시
        plt.text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                transform=plt.gca().transAxes)

        # 축 숨기기
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        # 전체 제목 및 저장
        plt.suptitle(f'종합 최적화 결과 - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        viz.save_plot('05_optimization_summary.png')

        # ==================== 시각화 6: 최적화 수렴 분석 ==================== #
        # -------------- 충분한 데이터가 있는 경우 수렴 분석 -------------- #
        # 10개 이상 시행 시 수렴 분석
        if len(values) >= 10:
            # Figure 생성
            plt.figure(figsize=(12, 6))

            # -------------- 이동평균 및 이동 표준편차 계산 -------------- #
            # 윈도우 크기 설정
            window_size = min(10, len(values) // 3)  # 최대 10, 최소 전체의 1/3

            # 이동 통계 저장
            moving_avg = []  # 이동평균
            moving_std = []  # 이동 표준편차

            # 윈도우별 통계 계산
            for i in range(window_size, len(values) + 1):
                window_values = values[i-window_size:i]  # 윈도우 데이터
                moving_avg.append(np.mean(window_values))  # 평균 계산
                moving_std.append(np.std(window_values))   # 표준편차 계산

            # -------------- 좌측: 성능 수렴 패턴 -------------- #
            plt.subplot(1, 2, 1)

            # x축 좌표 (윈도우 시작점)
            x_coords = range(window_size, len(values) + 1)

            # 이동평균 선 그래프
            plt.plot(x_coords, moving_avg, 'o-', color=viz.colors[0], alpha=0.7,
                    label=f'이동평균 (window={window_size})')

            # 신뢰구간 표시 (평균 ± 표준편차)
            plt.fill_between(x_coords,
                           np.array(moving_avg) - np.array(moving_std),
                           np.array(moving_avg) + np.array(moving_std),
                           alpha=0.3, color=viz.colors[0])

            # 그래프 꾸미기
            plt.title('성능 수렴 패턴')
            plt.xlabel('시행 번호')
            plt.ylabel('이동평균 F1 점수')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # -------------- 우측: 성능 변동성 변화 -------------- #
            plt.subplot(1, 2, 2)

            # 이동 표준편차 그래프
            plt.plot(x_coords, moving_std, 's-', color=viz.colors[1], alpha=0.7)

            # 그래프 꾸미기
            plt.title('성능 변동성 변화')
            plt.xlabel('시행 번호')
            plt.ylabel('이동 표준편차')
            plt.grid(True, alpha=0.3)

            # -------------- 수렴 판정 -------------- #
            # 최근 표준편차로 수렴 여부 판단
            recent_std = moving_std[-3:] if len(moving_std) >= 3 else moving_std  # 최근 3개
            convergence_status = "수렴" if np.mean(recent_std) < 0.01 else "진행중"  # 수렴 판정

            # 수렴 상태 표시
            plt.text(0.02, 0.98, f'수렴 상태: {convergence_status}',
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3",
                             facecolor="lightgreen" if convergence_status == "수렴" else "lightyellow",
                             alpha=0.7),
                    verticalalignment='top')

            # 전체 제목 및 저장
            plt.suptitle(f'최적화 수렴 분석 - {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            viz.save_plot('06_convergence_analysis.png')

        # -------------- 시각화 완료 메시지 -------------- #
        # 완료 정보 출력
        print(f"✅ Optimization visualizations completed: {viz.images_dir}")
        print(f"📊 Generated {len(list(viz.images_dir.glob('*.png')))} optimization visualization images")

    # -------------- 예외 발생 시 에러 처리 -------------- #
    # 시각화 실패 시 에러 메시지 출력
    except Exception as e:
        print(f"❌ Optimization visualization failed: {str(e)}")


# ==================== 파이프라인 호출 함수 ==================== #
# ---------------------- 최적화 파이프라인 시각화 호출 ---------------------- #
def visualize_optimization_pipeline(study_path: str, model_name: str, output_dir: str,
                                  experiment_name: Optional[str] = None):
    """최적화 파이프라인 시각화 호출

    Args:
        study_path: Optuna Study 파일 경로
        model_name: 모델 이름
        output_dir: 출력 디렉토리 경로
        experiment_name: 실험 이름 (선택, 현재 미사용)
    """
    # 최적화 시각화 함수 호출
    create_optimization_visualizations(study_path, model_name, output_dir)
