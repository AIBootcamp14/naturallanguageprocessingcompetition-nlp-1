# 토픽 클러스터링 파이프라인 요약

## 1. 개요
- 원본 데이터: `data/train.csv` (12,457행 / 고유 `topic` 9,235개)
- 목적: 롱테일 토픽 라벨을 상위 도메인으로 묶고, 희소 도메인을 증강해 학습 균형 확보
- 핵심 산출물
  - 클러스터링 결과: `analysis/topic_cluster_assignments.csv`, `analysis/topic_cluster_summary.csv`, `analysis/topic_cluster_metadata.json`
  - 인간관계 세부 클러스터: `analysis/human_interest_subclusters.csv`, `analysis/human_interest_subclusters_summary.csv`, `analysis/human_interest_subclusters_metadata.json`
  - 증강 데이터: `data/train_augmentation_origin.csv`
  - 최종 통합 데이터: `data/augmentation_final.csv`

## 2. 토픽 클러스터링 절차 (`scripts/topic_clustering.py`)
1. 토픽별 최대 3개 샘플을 모아 `aggregated_text` 구성
2. 정규식으로 특수문자 제거·공백 정리 (`token_string` 재사용)
3. 임베딩: `intfloat/multilingual-e5-base`, 입력 프리픽스 `passage: ...`
4. 차원 축소: UMAP(n_neighbors=15, n_components=15, min_dist=0.0, metric='cosine', random_state=42)
5. 군집화: HDBSCAN(min_cluster_size=25, min_samples=10, metric='euclidean', prediction_data=True)
6. 노이즈 재할당: 축소 공간에서 최근접 센트로이드로 할당 → 노이즈 0개
7. 키워드: TF–IDF(`(?u)[\w가-힣]+`, max_features=8000)
8. 제로샷 라벨링: `joeddav/xlm-roberta-large-xnli`, IPTC/IAB 상위 16개 도메인 후보
9. 결과 저장 및 메타데이터 기록

### 환경
- Python 3.10 / PyTorch 2.7.1+cu118
- `sentence-transformers 2.7.0`, `umap-learn 0.5.9.post2`, `hdbscan 0.8.39`
- GPU 사용 (`torch.cuda.is_available() == True`)
- 재실행 명령: `python scripts/topic_clustering.py`
- 수행 시간: 약 90~95초

## 3. 주요 지표
- 군집 수: 59 (노이즈 0)
- 군집 크기 중앙값: 104 (평균 ≈ 157)
- 제로샷 라벨 상위 분포 (샘플)

  | 도메인 | 토픽 수 |
  | --- | ---: |
  | Human interest & relationships | 5,493 |
  | Travel & transportation | 854 |
  | Lifestyle & leisure | 764 |
  | Education | 719 |
  | Health & medical | 470 |
  | Arts/culture/entertainment | 276 |
  | Science & technology | 162 |
  | Shopping & retail | 133 |
  | Labour & employment | 125 |
  | Environment | 59 |

## 4. 대규모 군집 수동 검토 (`analysis/topic_cluster_manual_review.csv`)
| cluster_id | 자동 라벨 | 제안 라벨 | 비고 |
| --- | --- | --- | --- |
| 11 | Human interest & relationships | 음식/식사 (Lifestyle & leisure) | 외식 위주 → 여가 도메인 권장 |
| 25 | Travel & transportation | 유지 | 교통 안내 중심 |
| 31 | Human interest & relationships | 여행/휴가 (Lifestyle & leisure) | 휴가·여행 계획 비중 ↑ |
| 40 | Human interest & relationships | 유지 | 감정/관계 대화 |
| 22 | Human interest & relationships | 금융/은행 (Business & finance) | ATM·은행 대화 다수 |

## 5. 인간관계/일상 세부 클러스터링 (`analysis/human_interest_subclusters*.csv`)
- 대상: `Human interest & relationships` 4,209건 (수동 보정 후)
- 파라미터: HDBSCAN(min_cluster_size=40, min_samples=10)
- 결과: 25개 세부 군집, 노이즈 0
- 한국어 라벨(`subcluster_label_ko`) 도입 → 학습/증강에서 바로 활용 가능

| 영문 라벨 | 한글 라벨 | 토픽 수 |
| --- | --- | ---: |
| Family relationships | 가족/친구 | 1,352 |
| Parenting and children | 육아/자녀 | 906 |
| Workplace relationship | 업무/직장 | 736 |
| Advice and guidance | 조언/안내 | 301 |
| Event or party planning | 행사/파티 계획 | 274 |
| Conflict resolution | 갈등 해결 | 239 |
| Daily chit-chat | 일상 대화 | 196 |
| Emotional support | 감정 지원 | 105 |
| Friendship support | 친구 상담 | 100 |

## 6. 증강 데이터 생성 (`scripts/generate_augmented_data.py`)
- 대상 도메인 및 목표 수량
  - 환경/스포츠/정치·정부/종교·신념: 200개 목표 → 각 134~162건 증강
  - 과학·기술, 쇼핑·소매: 400개 목표 → 각 213/202건 증강
- 생성 방식: 도메인별 시나리오 + 랜덤 조합으로 구어체 대화/문어체 요약 작성
- 출력 파일: `data/train_augmentation_origin.csv` (1,009행)
  - `fname`: `augment_{도메인}_{번호}`
  - `topic`: 한국어 기반 이름 (예: “환경 축제 준비 001”)
  - `cluster_id=-1`, `subcluster`/`subcluster_label` 비어 있음
  - 검증: 행 수, 도메인별 건수, 태그(`#Person#`) 포함 여부, 한글 포함 여부

### 증강 후 분포 (`data/augmentation_final.csv`)
- `data/train_with_domains.csv`(12,456행) + 증강 데이터 결합 → 13,465행
- 도메인별 최종 건수

  | 도메인 | 건수 |
  | --- | ---: |
  | 인간관계/일상 | 5,747 |
  | 여가/라이프스타일 | 2,545 |
  | 여행/교통 | 1,224 |
  | 교육/학습 | 849 |
  | 건강/의료 | 593 |
  | 금융/비즈니스 | 420 |
  | 과학/기술 | 400 |
  | 쇼핑/소매 | 400 |
  | 문화/엔터테인먼트 | 352 |
  | 환경/스포츠/정치/정부/종교/신념 | 각 200 |
  | 노동/고용 | 135 |

## 7. 후속 작업 안내
1. `data/augmentation_final.csv`(원본+증강)를 모델 학습에 사용하고, 희소 도메인에는 샘플 가중치/오버샘플링을 적용하여 균형 유지
2. `scripts/generate_augmented_data.py`는 seed=42로 재현 가능 → 필요 시 템플릿/seed 수정으로 다른 버전 생성 가능
3. 희소 세부 도메인(예: 감정 지원, 친구 상담)은 추가 검수 후 증강을 확대하거나, 도메인별 loss weight 조정
4. 추후 추가 증강(예: 도메인별 500건 목표)이 필요하면 현재 스크립트 기반으로 수량만 조정해 재생성 가능
