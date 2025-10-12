#!/usr/bin/env python3
"""
증강 데이터 품질 EDA 스크립트
- 원본 vs 증강 데이터 샘플 비교
- ROUGE 기반 유사도 측정
- 길이 분포 분석
- 품질 평가 및 결론 도출
"""

import pandas as pd
import numpy as np
from rouge import Rouge
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """증강 데이터 로드 및 원본/증강 분리"""
    print("=" * 80)
    print("📂 데이터 로드 중...")
    print("=" * 80)

    df = pd.read_csv(data_path)
    print(f"전체 데이터: {len(df):,}개")
    print(f"컬럼: {list(df.columns)}")

    # 원본과 증강 분리
    original = df[df['is_augmented'] == False].reset_index(drop=True)
    augmented = df[df['is_augmented'] == True].reset_index(drop=True)

    print(f"\n✅ 원본 데이터: {len(original):,}개")
    print(f"✅ 증강 데이터: {len(augmented):,}개")
    print(f"✅ 증강 비율: {len(augmented) / len(original):.2f}x")

    return df, original, augmented


def compare_samples(original: pd.DataFrame, augmented: pd.DataFrame, n_samples: int = 5):
    """원본과 증강 샘플 비교"""
    print("\n" + "=" * 80)
    print(f"📊 샘플 비교 (상위 {n_samples}개)")
    print("=" * 80)

    for i in range(n_samples):
        orig = original.iloc[i]
        aug = augmented.iloc[i]

        print(f"\n{'─' * 80}")
        print(f"샘플 #{i+1}: {orig['fname']}")
        print(f"{'─' * 80}")

        print(f"\n[원본 대화] (길이: {len(orig['dialogue'])})")
        print(orig['dialogue'][:300] + "..." if len(orig['dialogue']) > 300 else orig['dialogue'])

        print(f"\n[증강 대화] (길이: {len(aug['dialogue'])})")
        print(aug['dialogue'][:300] + "..." if len(aug['dialogue']) > 300 else aug['dialogue'])

        print(f"\n[요약] (원본 길이: {len(orig['summary'])}, 증강 길이: {len(aug['summary'])})")
        print(f"원본: {orig['summary']}")
        print(f"증강: {aug['summary']}")
        print(f"요약 동일: {'✅ YES' if orig['summary'] == aug['summary'] else '❌ NO'}")

        print(f"\n[토픽]")
        print(f"원본: {orig['topic']}")
        print(f"증강: {aug['topic']}")
        print(f"토픽 동일: {'✅ YES' if orig['topic'] == aug['topic'] else '❌ NO'}")


def calculate_rouge_scores(original: pd.DataFrame, augmented: pd.DataFrame, n_samples: int = 500) -> Dict:
    """ROUGE 점수 계산 (원본 dialogue vs 증강 dialogue)"""
    print("\n" + "=" * 80)
    print(f"📈 ROUGE 점수 계산 (샘플: {n_samples}개)")
    print("=" * 80)

    rouge = Rouge()
    scores = {
        'rouge-1-f': [],
        'rouge-2-f': [],
        'rouge-l-f': [],
    }

    valid_count = 0

    for i in range(min(n_samples, len(original))):
        orig_dialogue = str(original.iloc[i]['dialogue']).strip()
        aug_dialogue = str(augmented.iloc[i]['dialogue']).strip()

        # 빈 문자열 체크
        if not orig_dialogue or not aug_dialogue:
            continue

        try:
            score = rouge.get_scores(aug_dialogue, orig_dialogue)[0]
            scores['rouge-1-f'].append(score['rouge-1']['f'])
            scores['rouge-2-f'].append(score['rouge-2']['f'])
            scores['rouge-l-f'].append(score['rouge-l']['f'])
            valid_count += 1
        except Exception as e:
            continue

    print(f"\n✅ 유효한 샘플: {valid_count}/{n_samples}")

    # 통계 계산
    stats = {}
    for key, values in scores.items():
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
        }

    # 결과 출력
    print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("─" * 80)
    for key, stat in stats.items():
        print(f"{key:<15} {stat['mean']:<10.4f} {stat['std']:<10.4f} {stat['min']:<10.4f} {stat['max']:<10.4f} {stat['median']:<10.4f}")

    return stats


def analyze_length_distribution(original: pd.DataFrame, augmented: pd.DataFrame):
    """길이 분포 분석"""
    print("\n" + "=" * 80)
    print("📏 길이 분포 분석")
    print("=" * 80)

    orig_dialogue_len = original['dialogue'].apply(len)
    aug_dialogue_len = augmented['dialogue'].apply(len)
    orig_summary_len = original['summary'].apply(len)
    aug_summary_len = augmented['summary'].apply(len)

    print(f"\n[Dialogue 길이]")
    print(f"{'':>15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("─" * 60)
    print(f"{'원본':>15} {orig_dialogue_len.mean():<10.1f} {orig_dialogue_len.std():<10.1f} {orig_dialogue_len.min():<10} {orig_dialogue_len.max():<10}")
    print(f"{'증강':>15} {aug_dialogue_len.mean():<10.1f} {aug_dialogue_len.std():<10.1f} {aug_dialogue_len.min():<10} {aug_dialogue_len.max():<10}")
    print(f"{'차이 (%)':>15} {(aug_dialogue_len.mean() - orig_dialogue_len.mean()) / orig_dialogue_len.mean() * 100:<10.1f}%")

    print(f"\n[Summary 길이]")
    print(f"{'':>15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("─" * 60)
    print(f"{'원본':>15} {orig_summary_len.mean():<10.1f} {orig_summary_len.std():<10.1f} {orig_summary_len.min():<10} {orig_summary_len.max():<10}")
    print(f"{'증강':>15} {aug_summary_len.mean():<10.1f} {aug_summary_len.std():<10.1f} {aug_summary_len.min():<10} {aug_summary_len.max():<10}")
    print(f"{'차이':>15} {aug_summary_len.mean() - orig_summary_len.mean():<10.1f}")


def analyze_topic_distribution(original: pd.DataFrame, augmented: pd.DataFrame):
    """토픽 분포 분석"""
    print("\n" + "=" * 80)
    print("🏷️  토픽 분포 분석")
    print("=" * 80)

    orig_topic_counts = original['topic'].value_counts().head(10)
    aug_topic_counts = augmented['topic'].value_counts().head(10)

    print(f"\n[원본 데이터 상위 10개 토픽]")
    for topic, count in orig_topic_counts.items():
        print(f"  {topic}: {count}개 ({count/len(original)*100:.1f}%)")

    print(f"\n[증강 데이터 상위 10개 토픽]")
    for topic, count in aug_topic_counts.items():
        print(f"  {topic}: {count}개 ({count/len(augmented)*100:.1f}%)")


def make_decision(rouge_stats: Dict) -> Tuple[str, str]:
    """품질 평가 및 진행 여부 결정"""
    print("\n" + "=" * 80)
    print("🎯 품질 평가 및 결론")
    print("=" * 80)

    rouge_l_mean = rouge_stats['rouge-l-f']['mean']

    print(f"\n📊 핵심 지표: ROUGE-L F1 평균 = {rouge_l_mean:.4f}")

    if rouge_l_mean >= 0.8:
        decision = "⚠️  재검토"
        reason = "증강 데이터가 원본과 너무 유사합니다 (>0.8). 다양성 부족 위험."
        recommendation = "증강 파라미터 조정 또는 다른 증강 방법 고려"
    elif rouge_l_mean >= 0.5:
        decision = "✅ 진행"
        reason = "증강 데이터 품질이 양호합니다 (0.5~0.8). 유사하면서도 다양성 확보."
        recommendation = "Experiment #1 (증강 데이터 학습) 진행"
    elif rouge_l_mean >= 0.3:
        decision = "⚠️  주의"
        reason = "증강 데이터가 원본과 다소 차이가 있습니다 (0.3~0.5)."
        recommendation = "샘플 품질 재확인 후 진행 여부 결정"
    else:
        decision = "❌ 중단"
        reason = "증강 데이터 품질이 의심됩니다 (<0.3). 원본과 너무 다름."
        recommendation = "증강 방법 재검토 필요"

    print(f"\n{decision}")
    print(f"이유: {reason}")
    print(f"권장사항: {recommendation}")

    return decision, recommendation


def main():
    """메인 함수"""
    data_path = "/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/data/train_with_augmentation.csv"

    # 1. 데이터 로드
    df, original, augmented = load_data(data_path)

    # 2. 샘플 비교
    compare_samples(original, augmented, n_samples=3)

    # 3. ROUGE 점수 계산
    rouge_stats = calculate_rouge_scores(original, augmented, n_samples=500)

    # 4. 길이 분포 분석
    analyze_length_distribution(original, augmented)

    # 5. 토픽 분포 분석
    analyze_topic_distribution(original, augmented)

    # 6. 최종 결론
    decision, recommendation = make_decision(rouge_stats)

    print("\n" + "=" * 80)
    print("✅ EDA 완료")
    print("=" * 80)

    return decision, recommendation


if __name__ == "__main__":
    main()