#!/usr/bin/env python3
"""
요약문 품질 분석 스크립트
- 제출 파일과 원본 대화를 비교하여 문제점 분석
"""

import pandas as pd
import re
from pathlib import Path

# 파일 경로
submission_file = "experiments/20251015/20251015_031503_inference_kobart_kfold_soft_voting_bs32_maxnew60_hf_solar/submission/20251015_031503_inference_kobart_kfold_soft_voting_bs32_maxnew60_hf_solar.csv"
test_file = "data/raw/test.csv"

# 데이터 로드
submission_df = pd.read_csv(submission_file)
test_df = pd.read_csv(test_file)

# 문제 카테고리
issues = {
    "prefix_remaining": [],  # 접두사 미제거
    "no_summarization": [],  # 요약 미실행 (원본 그대로)
    "wrong_speaker_names": [],  # 화자 명칭 오류
    "too_long": [],  # 요약문이 원본보다 김
    "generic_names": [],  # A/B, #Person1# 등 일반 명칭 사용
}

# 접두사 패턴
prefix_patterns = [
    r'^대화\s*(내용)?\s*요약\s*[:：]',
    r'^Summary\s*[:：]',
    r'^요약\s*[:：]',
    r'^대화\s*[:：]',
    r'^대화\s*상대',
    r'^대화에서는?',
    r'^대화\s*참여자',
]

# 일반 명칭 패턴
generic_name_patterns = [
    r'\b[A-D]\b',  # A, B, C, D
    r'#Person\d+#',  # #Person1#, #Person2#
    r'고객\s*[A-D]\b',  # 고객 A, 고객 B
    r'친구\s*[A-D]\b',  # 친구 A, 친구 B
]

print("=" * 80)
print("요약문 품질 분석 시작")
print("=" * 80)

for idx, row in submission_df.iterrows():
    fname = row['fname']
    summary = row['summary']

    # 원본 대화 찾기
    original = test_df[test_df['fname'] == fname]['dialogue'].values
    if len(original) == 0:
        continue
    original = original[0]

    # 1. 접두사 검사
    for pattern in prefix_patterns:
        if re.search(pattern, summary):
            issues["prefix_remaining"].append({
                "fname": fname,
                "summary": summary,
                "pattern": pattern
            })
            break

    # 2. 요약 미실행 검사 (원본과 90% 이상 유사)
    # 간단한 휴리스틱: 요약문이 원본의 50% 이상이고 원본 텍스트가 많이 포함됨
    if len(summary) > len(original) * 0.5:
        # 원본의 주요 구문이 그대로 포함되어 있는지 확인
        original_sentences = original.split('.')[:3]  # 앞 3문장 샘플링
        match_count = sum(1 for sent in original_sentences if sent.strip() in summary)
        if match_count >= 2:
            issues["no_summarization"].append({
                "fname": fname,
                "summary": summary,
                "original_len": len(original),
                "summary_len": len(summary)
            })

    # 3. 길이 검사
    if len(summary) > len(original):
        issues["too_long"].append({
            "fname": fname,
            "original_len": len(original),
            "summary_len": len(summary),
            "diff": len(summary) - len(original)
        })

    # 4. 일반 명칭 사용 검사
    for pattern in generic_name_patterns:
        if re.search(pattern, summary):
            issues["generic_names"].append({
                "fname": fname,
                "summary": summary,
                "pattern": pattern,
                "original": original[:200] + "..."
            })
            break

    # 5. 화자 명칭 오류 검사 (휴리스틱)
    # - 대화에 존재하는 이름을 추출
    names_in_dialogue = re.findall(r'\b[A-Z][a-z]+\b', original)
    # - 반말/존댓말 분석
    is_informal = bool(re.search(r'(야|너|니가|네가|해|해줘|그래|할래)', original))
    is_formal = bool(re.search(r'(입니다|습니다|세요|시오|하십시오|드립니다)', original))

    # 고객/상담사 같은 비즈니스 관계 명칭이 비격식 대화에 사용된 경우
    if is_informal and not is_formal:
        if re.search(r'(고객|상담사|직원|관리자|의사|환자)', summary):
            # 실제로 업무 관련 단어가 있는지 확인
            business_words = re.search(r'(회사|업무|직장|계약|거래|예약|체크아웃|상담|진료)', original)
            if not business_words:
                issues["wrong_speaker_names"].append({
                    "fname": fname,
                    "summary": summary,
                    "reason": "비격식 대화에 비즈니스 명칭 사용",
                    "original_sample": original[:150] + "..."
                })

# 결과 출력
print(f"\n[1] 접두사 미제거: {len(issues['prefix_remaining'])}건")
for item in issues['prefix_remaining'][:5]:
    print(f"  - {item['fname']}: {item['summary'][:80]}...")
    print(f"    패턴: {item['pattern']}")

print(f"\n[2] 요약 미실행 (원본 그대로): {len(issues['no_summarization'])}건")
for item in issues['no_summarization'][:5]:
    print(f"  - {item['fname']}: 원본={item['original_len']}자, 요약={item['summary_len']}자")
    print(f"    요약: {item['summary'][:100]}...")

print(f"\n[3] 요약문이 원본보다 긴 경우: {len(issues['too_long'])}건")
for item in issues['too_long'][:5]:
    print(f"  - {item['fname']}: 원본={item['original_len']}자, 요약={item['summary_len']}자 (차이: +{item['diff']}자)")

print(f"\n[4] 일반 명칭 사용 (A/B, #Person#): {len(issues['generic_names'])}건")
generic_name_stats = {}
for item in issues['generic_names']:
    pattern = item['pattern']
    generic_name_stats[pattern] = generic_name_stats.get(pattern, 0) + 1

for pattern, count in sorted(generic_name_stats.items(), key=lambda x: -x[1]):
    print(f"  - {pattern}: {count}건")

# 샘플 출력
print("\n  샘플:")
for item in issues['generic_names'][:5]:
    print(f"    {item['fname']}: {item['summary'][:80]}...")

print(f"\n[5] 화자 명칭 오류 (맥락 불일치): {len(issues['wrong_speaker_names'])}건")
for item in issues['wrong_speaker_names'][:5]:
    print(f"  - {item['fname']}: {item['reason']}")
    print(f"    원본 샘플: {item['original_sample']}")
    print(f"    요약: {item['summary'][:80]}...")

# 통계 요약
print("\n" + "=" * 80)
print("통계 요약")
print("=" * 80)
total_samples = len(submission_df)
print(f"전체 샘플 수: {total_samples}")
print(f"문제 발견 샘플 수:")
print(f"  - 접두사 미제거: {len(issues['prefix_remaining'])} ({len(issues['prefix_remaining'])/total_samples*100:.1f}%)")
print(f"  - 요약 미실행: {len(issues['no_summarization'])} ({len(issues['no_summarization'])/total_samples*100:.1f}%)")
print(f"  - 너무 긴 요약: {len(issues['too_long'])} ({len(issues['too_long'])/total_samples*100:.1f}%)")
print(f"  - 일반 명칭 사용: {len(issues['generic_names'])} ({len(issues['generic_names'])/total_samples*100:.1f}%)")
print(f"  - 화자 명칭 오류: {len(issues['wrong_speaker_names'])} ({len(issues['wrong_speaker_names'])/total_samples*100:.1f}%)")

# 상세 케이스 분석을 위한 특정 샘플 출력
print("\n" + "=" * 80)
print("사용자가 지적한 특정 케이스 분석")
print("=" * 80)

problem_cases = ["test_1", "test_8", "test_39", "test_312"]
for fname in problem_cases:
    if fname in submission_df['fname'].values:
        summary = submission_df[submission_df['fname'] == fname]['summary'].values[0]
        original = test_df[test_df['fname'] == fname]['dialogue'].values[0]

        print(f"\n[{fname}]")
        print(f"원본 대화 ({len(original)}자):")
        print(f"{original[:300]}...")
        print(f"\n요약문 ({len(summary)}자):")
        print(f"{summary}")
        print("-" * 80)

print("\n분석 완료!")
