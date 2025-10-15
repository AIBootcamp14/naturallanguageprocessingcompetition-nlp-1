#!/usr/bin/env python3
"""
추론 모듈
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

import torch
import pandas as pd
from typing import Dict
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from data_loader import Preprocess
from dataset import prepare_test_dataset
from inference_utils import generate_summaries, postprocess_summaries


class Inferencer:
    """
    추론 파이프라인 관리 클래스
    """

    def __init__(self, config: Dict, experiment_name: str, checkpoint_name: str):
        """
        Inferencer 초기화

        Args:
            config: 설정 딕셔너리
            experiment_name: 실험 이름
            checkpoint_name: 체크포인트 이름 (예: 'checkpoint-2068')
        """
        self.config = config
        self.experiment_name = experiment_name
        self.checkpoint_name = checkpoint_name
        self.device = self._get_device()

        # 체크포인트 경로 자동 생성
        self.checkpoint_path = os.path.join(
            self.config['general']['output_dir'],
            checkpoint_name
        )

    def _get_device(self) -> torch.device:
        """
        사용할 디바이스를 반환합니다

        Returns:
            torch.device
        """
        from utils import get_device
        return get_device()

    def run(self, model: BartForConditionalGeneration, tokenizer: PreTrainedTokenizerFast,
           output_path: str):
        """
        전체 추론 파이프라인을 실행합니다

        Args:
            model: 추론할 모델
            tokenizer: 토크나이저
            output_path: 결과 저장 경로 (CSV)
        """
        print("\n" + "=" * 80)
        print(f"🚀 {self.experiment_name} 추론 시작")
        print("=" * 80)
        print(f"체크포인트: {self.checkpoint_name}")
        print(f"출력 경로: {output_path}")
        print("=" * 80)

        # 1. Test 데이터셋 준비
        print("\n" + "=" * 80)
        print("1단계: Test 데이터셋 준비 중...")
        print("=" * 80)

        preprocessor = Preprocess(
            bos_token=self.config['tokenizer']['bos_token'],
            eos_token=self.config['tokenizer']['eos_token']
        )

        test_data_df, test_dataset = prepare_test_dataset(
            self.config, preprocessor, tokenizer
        )

        # DataLoader 생성
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config['inference']['batch_size'],
            shuffle=False
        )

        print(f"✅ Test 데이터셋 준비 완료: {len(test_dataset)} samples")

        # 2. 요약 생성
        print("\n" + "=" * 80)
        print("2단계: 요약 생성 중...")
        print("=" * 80)

        # tokenizer를 config에 임시 저장 (generate_summaries에서 사용)
        self.config['tokenizer'] = tokenizer

        fnames, raw_summaries = generate_summaries(
            model, test_dataloader, self.config, self.device
        )

        print(f"✅ {len(fnames)}개의 요약문 생성 완료")
        print(f"   - 첫 번째 파일: {fnames[0]}")
        print(f"   - 원본 요약 예시: {raw_summaries[0][:100]}...")

        # 3. 후처리 (특수 토큰 제거)
        print("\n" + "=" * 80)
        print("3단계: 후처리 중 (특수 토큰 제거)...")
        print("=" * 80)

        remove_tokens = self.config['inference']['remove_tokens']
        print(f"제거할 토큰: {remove_tokens}")

        cleaned_summaries = postprocess_summaries(raw_summaries, remove_tokens)

        print(f"✅ 후처리 완료")
        print(f"   - 후처리 요약 예시: {cleaned_summaries[0][:100]}...")

        # 4. 결과 저장 (Competition 형식: index 포함)
        print("\n" + "=" * 80)
        print("4단계: 결과 저장 중...")
        print("=" * 80)

        result_df = pd.DataFrame({
            'fname': fnames,
            'summary': cleaned_summaries
        })

        # index=True로 저장 (competition 제출 형식)
        result_df.to_csv(output_path, index=True)

        print(f"✅ 저장 완료: {output_path}")
        print(f"   형식: CSV with index column (competition format)")

        # 5. 결과 확인
        print("\n" + "=" * 80)
        print("🔍 결과 확인")
        print("=" * 80)
        print(f"총 {len(result_df)}개 요약 생성")
        print(f"\n샘플 (처음 3개):")
        print(result_df.head(3))

        print("\n" + "=" * 80)
        print(f"✅ {self.experiment_name} 추론 완료!")
        print("=" * 80)
        print(f"제출 파일: {output_path}")

        return result_df
