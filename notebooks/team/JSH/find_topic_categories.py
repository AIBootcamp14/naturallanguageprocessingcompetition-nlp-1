# :라벨: SEMANTIC TOPIC CLUSTERING (Find Topic Categories)

from collections import defaultdict

import re

print("\n" + "="*100)
print(":라벨:  SEMANTIC TOPIC CLUSTERING")
print("="*100)

# Define topic domain patterns (Korean keywords)

domain_keywords = {
    '음식/식사': ['음식', '주문', '식당', '레스토랑', '메뉴', '요리', '먹', '식사', '점심', '저녁', '아침', '카페'],
    '쇼핑/구매': ['쇼핑', '구매', '가격', '할인', '상품', '물건', '사', '매장', '가게'],
    '여행/교통': ['여행', '택시', '비행기', '호텔', '공항', '버스', '지하철', '교통', '예약', '항공'],
    '업무/직장': ['회사', '업무', '회의', '직장', '사무실', '근무', '출근', '퇴근', '상사', '동료'],
    '건강/의료': ['병원', '의사', '건강', '검진', '치료', '약', '환자', '증상', '아프', '의료'],
    '학습/교육': ['학교', '공부', '수업', '학습', '교육', '학생', '선생', '과목', '시험'],
    '여가/취미': ['영화', '게임', '운동', '취미', '파티', '콘서트', '전시', '스포츠'],
    '가족/친구': ['가족', '친구', '부모', '자녀', '형제', '친척', '결혼', '생일'],
    '금융/은행': ['은행', '돈', '계좌', '카드', '대출', '저축', '입금', '출금', '금융'],
    '부동산': ['집', '아파트', '임대', '월세', '전세', '부동산', '주택'],
    '면접/취업': ['면접', '취업', '입사', '지원', '이력서', '채용'],
    '고객서비스': ['고객', '문의', '불만', '환불', '상담', '서비스', '직원'],
}

# Categorize topics

topic_categories = defaultdict(list)
uncategorized = []

for topic in train_df['topic'].unique():
    categorized = False

    for domain, keywords in domain_keywords.items():
        if any(keyword in topic for keyword in keywords):
            topic_categories[domain].append(topic)
            categorized = True
            break

    if not categorized:
        uncategorized.append(topic)

print(f"\n:막대_차트: Topic Categorization Results:")
print(f"  Total unique topics: {len(train_df['topic'].unique()):,}")
print(f"  Categorized: {sum(len(v) for v in topic_categories.values()):,}")
print(f"  Uncategorized: {len(uncategorized):,}")
print(f"\n:클립보드: Topics by Domain:")

for domain, topics in sorted(topic_categories.items(), key=lambda x: len(x[1]), reverse=True):
    # Count examples in this domain
    domain_examples = train_df[train_df['topic'].isin(topics)]
    print(f"\n  {domain}: {len(topics)} topics, {len(domain_examples)} examples ({len(domain_examples)/len(train_df)*100:.1f}%)")
    # Show top topics in this domain
    domain_topic_counts = train_df[train_df['topic'].isin(topics)]['topic'].value_counts().head(5)
    print(f"    Top topics:")

    for topic, count in domain_topic_counts.items():
        print(f"      - {topic}: {count} examples")

# Show some uncategorized topics

print(f"\n:질문: Sample Uncategorized Topics (need manual review):")

for topic in uncategorized[:20]:
    count = (train_df['topic'] == topic).sum()
    print(f"  - {topic}: {count} examples")