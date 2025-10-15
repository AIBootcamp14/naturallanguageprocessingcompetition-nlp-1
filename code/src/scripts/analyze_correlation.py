#!/usr/bin/env python3
"""
Dev/Test 상관관계 분석

Phase 0 결과를 분석하여 Dev score가 Test score를 얼마나 예측하는지 확인
"""

import sys
sys.path.append('../scripts')

import numpy as np
import matplotlib.pyplot as plt
from wandb_utils import init_wandb, log_correlation_analysis, finish_run

print("="*80)
print("📊 Dev/Test 상관관계 분석")
print("="*80)
print()

# Phase 0 결과 데이터
experiments = {
    "Baseline": {
        "dev": 12.36,
        "test": 46.95
    },
    "Exp #4": {
        "dev": 11.50,
        "test": 47.44
    }
}

print("**실험 결과**:")
print()
print("| 실험 | Dev Score | Test Score | Gap | Ratio (Test/Dev) |")
print("|------|-----------|------------|-----|------------------|")

for name, scores in experiments.items():
    dev = scores["dev"]
    test = scores["test"]
    gap = abs(test - dev)
    ratio = test / dev
    print(f"| {name:12s} | {dev:9.2f} | {test:10.2f} | {gap:5.2f} | {ratio:16.2f} |")

# 변화 분석
baseline_dev = experiments["Baseline"]["dev"]
baseline_test = experiments["Baseline"]["test"]
exp4_dev = experiments["Exp #4"]["dev"]
exp4_test = experiments["Exp #4"]["test"]

dev_delta = exp4_dev - baseline_dev
test_delta = exp4_test - baseline_test

print()
print("**변화 분석**:")
print(f"- Dev 변화:  {dev_delta:+.2f} ({'⬇️ 감소' if dev_delta < 0 else '⬆️ 증가'})")
print(f"- Test 변화: {test_delta:+.2f} ({'⬇️ 감소' if test_delta < 0 else '⬆️ 증가'})")
print()

# 역전 현상 감지
if (dev_delta < 0 and test_delta > 0) or (dev_delta > 0 and test_delta < 0):
    print("⚠️  **역전 현상 발생!**")
    print("   Dev에서의 변화 방향과 Test에서의 변화 방향이 반대입니다.")
    print("   Dev score는 Test score를 예측하지 못합니다!")
    correlation_quality = "매우 낮음"
else:
    print("✅ **일관성 있음**")
    print("   Dev와 Test의 변화 방향이 일치합니다.")

    # 상관관계 강도 평가
    dev_change_pct = abs(dev_delta / baseline_dev * 100)
    test_change_pct = abs(test_delta / baseline_test * 100)
    ratio_similarity = min(dev_change_pct, test_change_pct) / max(dev_change_pct, test_change_pct)

    if ratio_similarity > 0.8:
        correlation_quality = "높음"
    elif ratio_similarity > 0.5:
        correlation_quality = "보통"
    else:
        correlation_quality = "낮음"

print()
print(f"**상관관계 품질**: {correlation_quality}")
print()

# 통계 계산
dev_scores = [baseline_dev, exp4_dev]
test_scores = [baseline_test, exp4_test]
exp_names = ["Baseline", "Exp #4"]

# Pearson 상관계수 (2개 점이라 완벽한 선형)
correlation = np.corrcoef(dev_scores, test_scores)[0, 1]
mae = np.mean(np.abs(np.array(dev_scores) - np.array(test_scores)))
mean_gap = np.mean([baseline_test - baseline_dev, exp4_test - exp4_dev])

print("**통계**:")
print(f"- Pearson 상관계수: {correlation:.4f}")
print(f"- 평균 절대 오차 (MAE): {mae:.2f}")
print(f"- 평균 Gap: {mean_gap:.2f}")
print()

# Wandb 초기화
wandb_run = init_wandb(
    experiment_name="correlation-analysis",
    config={
        "num_experiments": len(experiments),
        "baseline_dev": baseline_dev,
        "baseline_test": baseline_test,
        "exp4_dev": exp4_dev,
        "exp4_test": exp4_test,
        "dev_delta": dev_delta,
        "test_delta": test_delta,
        "correlation": correlation,
        "mae": mae,
        "mean_gap": mean_gap
    },
    tags=["correlation", "phase-0", "analysis"],
    group="phase-0-correlation",
    notes="Dev/Test 상관관계 분석 결과"
)

# Wandb에 상관관계 분석 기록
stats = log_correlation_analysis(
    dev_scores=dev_scores,
    test_scores=test_scores,
    exp_names=exp_names
)

# 제출 전략 제안
print("="*80)
print("**제출 전략 제안**")
print("="*80)
print()

if correlation_quality in ["매우 낮음", "낮음"]:
    print("❌ **Dev score를 신뢰할 수 없습니다!**")
    print()
    print("**권장 전략**:")
    print("1. **직접 Test 제출**: Dev sweep 결과를 무시하고 모든 후보를 Test에 제출")
    print("2. **다양성 우선**: Dev 1등이 아니라 다양한 설정을 Test")
    print("3. **제출 횟수 최대 활용**: 남은 제출 횟수를 적극 사용")
    print()
    print("**Length Penalty Sweep 전략**:")
    print("- LP=0.5, 0.6, 0.7, 0.8 모두 Dev에서 테스트")
    print("- Dev 순위와 무관하게 2-3개를 Test에 제출 (다양한 값 선택)")
    print("- 예: Dev 1등, 3등, 4등 제출 (극단값 포함)")
else:
    print("✅ **Dev score를 어느 정도 신뢰할 수 있습니다!**")
    print()
    print("**권장 전략**:")
    print("1. **Dev 기준 선택**: Dev에서 가장 좋은 1-2개를 Test에 제출")
    print("2. **보수적 접근**: 제출 횟수 절약")
    print()
    print("**Length Penalty Sweep 전략**:")
    print("- LP=0.5, 0.6, 0.7, 0.8 모두 Dev에서 테스트")
    print("- Dev 1등만 Test 제출 (1회)")
    print("- 실패 시 Dev 2등 제출 (추가 1회)")

print()
print("="*80)

# Summary 저장
finish_run(summary_metrics={
    "correlation": correlation,
    "mae": mae,
    "mean_gap": mean_gap,
    "correlation_quality": correlation_quality,
    "baseline_dev": baseline_dev,
    "baseline_test": baseline_test,
    "exp4_dev": exp4_dev,
    "exp4_test": exp4_test,
    "dev_delta": dev_delta,
    "test_delta": test_delta,
    "reversal_detected": (dev_delta < 0 and test_delta > 0) or (dev_delta > 0 and test_delta < 0)
})

print("\n✅ 분석 완료!")
print(f"   Wandb: {wandb_run.url}")
print("="*80)