#!/usr/bin/env python3
"""
Exp #4 Test Set 추론 스크립트

길이 정규화 + Max Length 768로 학습된 모델을 사용하여
Test set에 대한 요약문을 생성하고 CSV 파일로 저장합니다.

변경사항:
1. encoder_max_len: 768 (긴 대화 처리)
2. length_penalty: 0.6 (GNMT 길이 정규화)

사용 방법:
    python infer_test_exp4.py

출력:
    - prediction/exp4_output.csv
"""

import sys
import os
import glob

# scripts 디렉토리를 Python path에 추가
sys.path.append('../scripts')

import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration

# 모듈 import
from utils import load_config, get_device, set_seed
from data_loader import Preprocess
from tokenizer_utils import load_tokenizer
from dataset import prepare_test_dataset
from inference_utils import run_inference

def main():
    print("=" * 80)
    print("🚀 Exp #4 Test Set 추론")
    print("   (길이 정규화 + Max Length 768)")
    print("=" * 80)

    # 1. Config 로드 - config_exp4.yaml 사용
    print("\n✅ Config 로드...")
    config = load_config('./config_exp4.yaml')

    # Device 설정
    device = get_device()
    print(f"   Device: {device}")

    # 시드 설정
    set_seed(config['training']['seed'])
    print(f"   Seed: {config['training']['seed']}")

    # Config 확인
    print(f"   Encoder Max Length: {config['tokenizer']['encoder_max_len']}")
    print(f"   Length Penalty: {config['inference'].get('length_penalty', 1.0)}")

    # 2. Checkpoint 자동 탐색 (최신 checkpoint 사용)
    output_dir = config['general']['output_dir']
    checkpoints = glob.glob(f"{output_dir}/checkpoint-*")

    if not checkpoints:
        print(f"\n❌ Checkpoint를 찾을 수 없습니다: {output_dir}")
        print(f"   먼저 python train_exp4.py를 실행하세요")
        return

    # checkpoint 번호로 정렬 (최신순)
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
    checkpoint_path = checkpoints[0]

    print(f"\n✅ Checkpoint: {checkpoint_path}")

    # 3. Tokenizer 로드
    print("\n✅ Tokenizer 로드...")
    model_name = config['general']['model_name']
    special_tokens = config['tokenizer']['special_tokens']
    tokenizer = load_tokenizer(model_name, special_tokens)
    print(f"   Vocab size: {len(tokenizer)}")

    # 4. 모델 로드 (checkpoint에서)
    print("\n✅ 모델 로드...")
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    model.to(device)
    print(f"   모델 로드 완료: {checkpoint_path}")

    # 5. Test 데이터셋 준비
    print("\n✅ Test 데이터셋 준비...")
    preprocessor = Preprocess(
        bos_token=config['tokenizer']['bos_token'],
        eos_token=config['tokenizer']['eos_token']
    )

    test_data, test_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    print(f"   Test samples: {len(test_dataset)}")

    # DataLoader 생성
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size']
    )

    # 6. 추론 실행
    print("\n" + "=" * 80)
    print("🔮 추론 시작...")
    print("=" * 80)

    # 저장 경로 설정
    save_path = './prediction/exp4_output.csv'

    result_df = run_inference(
        model=model,
        tokenizer=tokenizer,
        test_dataloader=test_dataloader,
        config=config,
        device=device,
        save_path=save_path
    )

    # 7. 결과 확인
    print("\n" + "=" * 80)
    print("📊 결과 확인")
    print("=" * 80)
    print(f"✅ 저장 완료: {save_path}")
    print(f"   샘플 수: {len(result_df)}")

    # 처음 3개 샘플 출력
    print("\n샘플 요약 (처음 3개):")
    print("-" * 80)
    for i in range(min(3, len(result_df))):
        print(f"\n[{i}] {result_df.iloc[i]['fname']}")
        summary = result_df.iloc[i]['summary']
        if len(summary) > 100:
            print(f"    {summary[:100]}...")
        else:
            print(f"    {summary}")
    print("-" * 80)

    # CSV 형식 검증
    print("\n✅ CSV 형식 검증...")
    from utils import validate_csv
    validation = validate_csv(save_path)

    if validation['valid']:
        print("   ✅ 검증 통과")
        print(f"   - 샘플 수: {validation['num_samples']}")
        print(f"   - 컬럼: {validation['columns']}")
    else:
        print("   ❌ 검증 실패")
        for error in validation['errors']:
            print(f"   - {error}")

    print("\n" + "=" * 80)
    print("✅ 완료!")
    print("=" * 80)
    print(f"\n다음 단계:")
    print(f"1. 파일 확인: {save_path}")
    print(f"2. 대회 플랫폼 제출")
    print(f"3. 점수 기록")
    print(f"   - Baseline: 46.95점")
    print(f"   - Exp #4: ???점")
    print(f"   - 델타: ???점")
    print("=" * 80)

if __name__ == '__main__':
    main()
