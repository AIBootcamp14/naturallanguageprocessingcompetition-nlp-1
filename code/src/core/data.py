#!/usr/bin/env python3
"""
데이터 로딩 및 가중치 샘플링 모듈
"""

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from typing import Dict, Tuple, Optional
from src.scripts.data_loader import Preprocess
from src.scripts.dataset import prepare_train_dataset, DatasetForTrain, DatasetForVal


class DataManager:
    """
    데이터 로딩 및 가중치 샘플링 관리 클래스
    """

    def __init__(self, config: Dict, tokenizer):
        """
        DataManager 초기화

        Args:
            config: 설정 딕셔너리
            tokenizer: 토크나이저
        """
        self.config = config
        self.tokenizer = tokenizer
        self.preprocessor = Preprocess(
            bos_token=config['tokenizer']['bos_token'],
            eos_token=config['tokenizer']['eos_token']
        )

    def prepare_data(self) -> Tuple[DatasetForTrain, DatasetForVal, Optional[WeightedRandomSampler]]:
        """
        학습/검증 데이터셋을 준비하고 가중치 샘플러를 생성합니다.

        Returns:
            (train_dataset, val_dataset, sampler)
            - train_dataset: 학습 데이터셋
            - val_dataset: 검증 데이터셋
            - sampler: WeightedRandomSampler (가중치 사용 시) 또는 None
        """
        data_path = self.config['general']['data_path']

        # 데이터셋 준비 (scripts/dataset.py 활용)
        train_dataset, val_dataset = prepare_train_dataset(
            self.config, self.preprocessor, data_path, self.tokenizer
        )

        print(f"\n✅ 데이터셋 준비 완료")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")

        # 가중치 샘플링 여부 확인
        use_weights = self.config.get('data', {}).get('use_weights', False)

        if use_weights:
            print("\n" + "=" * 80)
            print("⚖️  가중치 샘플링 시스템 구성 중...")
            print("=" * 80)
            sampler = self._create_weighted_sampler(data_path)
        else:
            print("\n⚖️  가중치 샘플링 비활성화 (자연 분포)")
            sampler = None

        return train_dataset, val_dataset, sampler

    def _create_weighted_sampler(self, data_path: str) -> WeightedRandomSampler:
        """
        WeightedRandomSampler를 생성합니다 (train_exp7f.py 로직 참조)

        Args:
            data_path: 데이터 디렉토리 경로

        Returns:
            WeightedRandomSampler
        """
        # CSV 로드 (메타데이터)
        train_file = self.config['data'].get('train_file', 'train.csv')
        df = pd.read_csv(os.path.join(data_path, train_file))
        print(f"✅ 메타데이터 로드: {len(df)} samples")

        # 가중치 설정 가져오기
        weight_config = self.config['data']['weight_config']
        domain_weights = weight_config['domain_weights']
        subcluster_weights = weight_config['subcluster_weights']
        domain_threshold = weight_config['domain_threshold']
        subcluster_threshold = weight_config['subcluster_threshold']

        # 가중치 정보 출력
        print(f"\n도메인 가중치 ({domain_threshold}개 기준):")
        for domain in sorted(domain_weights.keys(), key=lambda x: -domain_weights[x]):
            weight = domain_weights[domain]
            count = len(df[df['adjusted_label'] == domain])
            threshold_mark = "" if count < domain_threshold else f" (≥{domain_threshold})"
            print(f"   {domain:20s}: {weight:.2f}x ({count:4d}개{threshold_mark})")

        print(f"\n서브클러스터 가중치 ({subcluster_threshold}개 기준):")
        df_relationship = df[df['adjusted_label'] == '인간관계/일상']
        for subcluster in sorted(subcluster_weights.keys(), key=lambda x: -subcluster_weights[x]):
            weight = subcluster_weights[subcluster]
            count = len(df_relationship[df_relationship['subcluster_label'] == subcluster])
            threshold_mark = "" if count < subcluster_threshold else f" (≥{subcluster_threshold})"
            print(f"   {subcluster:20s}: {weight:.2f}x ({count:4d}개{threshold_mark})")

        # 각 샘플에 가중치 할당
        weights = self._calculate_weights(df, domain_weights, subcluster_weights)

        print(f"\n✅ 가중치 계산 완료: {len(weights)} samples")
        print(f"   Min weight: {min(weights):.2f}")
        print(f"   Max weight: {max(weights):.2f}")
        print(f"   Avg weight: {sum(weights)/len(weights):.2f}")

        # 가중치 분포 출력
        self._print_weight_distribution(weights)

        # WeightedRandomSampler 생성
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        print(f"\n✅ WeightedRandomSampler 생성 완료")
        print("=" * 80)

        return sampler

    def _calculate_weights(self, df: pd.DataFrame, domain_weights: Dict[str, float],
                          subcluster_weights: Dict[str, float]) -> list:
        """
        각 샘플의 가중치를 계산합니다 (train_exp7f.py 로직)

        Args:
            df: 학습 데이터프레임
            domain_weights: 도메인 가중치 딕셔너리
            subcluster_weights: 서브클러스터 가중치 딕셔너리

        Returns:
            가중치 리스트
        """
        weights = []
        for idx, row in df.iterrows():
            domain = row['adjusted_label']

            # 인간관계/일상: 서브클러스터 가중치 사용
            if domain == '인간관계/일상':
                subcluster = row['subcluster_label']
                weight = subcluster_weights.get(subcluster, 1.0)
            # 기타 도메인: 도메인 가중치 사용
            else:
                weight = domain_weights.get(domain, 1.0)

            weights.append(weight)

        return weights

    def _print_weight_distribution(self, weights: list):
        """
        가중치 분포를 출력합니다

        Args:
            weights: 가중치 리스트
        """
        from collections import Counter

        weight_dist = Counter([round(w, 2) for w in weights])
        print(f"\n가중치 분포 (Top 10):")
        for w in sorted(weight_dist.keys(), reverse=True)[:10]:
            count = weight_dist[w]
            pct = count / len(weights) * 100
            print(f"   {w:5.2f}배: {count:5d}개 ({pct:5.2f}%)")

        # 1.0x 비율 계산
        count_1x = sum(1 for w in weights if w == 1.0)
        pct_1x = count_1x / len(weights) * 100
        print(f"\n자연 분포 유지: {count_1x:5d}개 ({pct_1x:.1f}%)")
