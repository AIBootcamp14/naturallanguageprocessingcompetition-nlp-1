#!/usr/bin/env python3
"""
Experiment #2: 후처리 개선 실행 스크립트

checkpoint-1750을 사용하여 추론을 실행하고,
postprocess_summaries_v2를 적용하여 output_modular_v2.csv를 생성합니다.

실행 방법:
    cd /Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code
    python run_exp2.py
"""

import sys
sys.path.append('../scripts')

import torch
from torch.utils.data import DataLoader

from utils import load_config, get_device, set_seed, validate_csv
from data_loader import Preprocess
from tokenizer_utils import load_tokenizer
from model_utils import load_model_for_inference
from dataset import prepare_test_dataset
from inference_utils import generate_summaries, postprocess_summaries_v2, save_predictions


def main():
    print("=" * 80)
    print("🚀 Experiment #2: 후처리 개선 (Post-processing v2)")
    print("=" * 80)
    print()
    print("변경사항:")
    print("  - postprocess_summaries_v2 사용")
    print("  - 공백 정규화 추가")
    print("  - 중복 문장 제거 추가")
    print()
    print("목표: +0.5~1.2점")
    print("예상 점수: 47.5~48.2")
    print("=" * 80)
    print()

    # 1. Config 로드
    print("Step 1: Config 로드")
    config = load_config('./config.yaml')
    device = get_device()
    set_seed(config['training']['seed'])
    print(f"  ✅ Device: {device}")
    print(f"  ✅ Seed: {config['training']['seed']}")
    print()

    # 2. Tokenizer 로드
    print("Step 2: Tokenizer 로드")
    tokenizer = load_tokenizer(
        config['general']['model_name'],
        config['tokenizer']['special_tokens']
    )
    print(f"  ✅ Model: {config['general']['model_name']}")
    print(f"  ✅ Vocab size: {len(tokenizer)}")
    print()

    # 3. 모델 로드 (checkpoint-1750)
    print("Step 3: 모델 로드 (checkpoint-1750)")
    checkpoint_path = '../submission/checkpoint-1750'
    print(f"  Checkpoint: {checkpoint_path}")
    model = load_model_for_inference(checkpoint_path, tokenizer, device)
    print(f"  ✅ 모델 로드 완료")
    print()

    # 4. Test 데이터 준비
    print("Step 4: Test 데이터 준비")
    preprocessor = Preprocess(
        bos_token=config['tokenizer']['bos_token'],
        eos_token=config['tokenizer']['eos_token']
    )
    test_data, test_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size']
    )
    print(f"  ✅ Test samples: {len(test_dataset)}")
    print()

    # 5. 추론 실행
    print("Step 5: 추론 실행")
    print("-" * 80)
    config['tokenizer'] = tokenizer
    fnames, raw_summaries = generate_summaries(model, test_dataloader, config, device)
    print()
    print(f"  ✅ {len(fnames)}개의 요약문 생성 완료")
    print(f"  원본 요약 예시: {raw_summaries[0][:80]}...")
    print()

    # 6. 후처리 v2 적용 (Exp #2의 핵심!)
    print("Step 6: 후처리 v2 적용 🔥")
    print("  처리 단계:")
    print("    1. 특수 토큰 제거")
    print("    2. 공백 정규화")
    print("    3. 중복 문장 제거")

    remove_tokens = config['inference']['remove_tokens']
    cleaned_summaries = postprocess_summaries_v2(raw_summaries, remove_tokens)

    print(f"  ✅ 후처리 v2 완료")
    print(f"  후처리 요약 예시: {cleaned_summaries[0][:80]}...")
    print()

    # 7. CSV 저장
    print("Step 7: CSV 저장")
    output_path = save_predictions(
        fnames, cleaned_summaries,
        output_dir='./prediction',
        filename='output_modular_v2.csv'
    )
    print(f"  ✅ 저장 완료: {output_path}")
    print()

    # 8. CSV 검증
    print("Step 8: CSV 검증")
    validation_result = validate_csv(output_path)
    print(f"  유효성: {'✅ 통과' if validation_result['valid'] else '❌ 실패'}")
    print(f"  샘플 수: {validation_result['num_samples']}")
    print(f"  컬럼: {validation_result['columns']}")

    if validation_result['errors']:
        print("\n  ⚠️ 오류:")
        for error in validation_result['errors']:
            print(f"    - {error}")
        return False
    print()

    # 9. 샘플 확인
    print("Step 9: 샘플 확인 (처음 5개)")
    print("-" * 80)
    import pandas as pd
    result_df = pd.read_csv(output_path)
    for i in range(min(5, len(result_df))):
        print(f"\n[{i}] {result_df.iloc[i]['fname']}")
        print(f"    {result_df.iloc[i]['summary'][:100]}...")
    print("-" * 80)
    print()

    # 10. Baseline과 비교
    print("Step 10: Baseline Modular과 비교")
    try:
        baseline_modular = pd.read_csv('./prediction/output_modular.csv')
        exp2_output = result_df

        identical_count = (baseline_modular['summary'] == exp2_output['summary']).sum()
        print(f"  Baseline Modular 샘플 수: {len(baseline_modular)}")
        print(f"  Exp #2 샘플 수: {len(exp2_output)}")
        print(f"  동일한 요약문 수: {identical_count} / {len(baseline_modular)}")
        print(f"  일치율: {identical_count / len(baseline_modular) * 100:.2f}%")

        # 차이나는 샘플 확인
        different_mask = baseline_modular['summary'] != exp2_output['summary']
        different_count = different_mask.sum()
        print(f"  변경된 샘플 수: {different_count}")

        if different_count > 0:
            print("\n  변경된 샘플 예시 (처음 3개):")
            different_samples = baseline_modular[different_mask].head(3)
            for idx in different_samples.index:
                print(f"\n    [{idx}] {baseline_modular.iloc[idx]['fname']}")
                print(f"      Before: {baseline_modular.iloc[idx]['summary'][:60]}...")
                print(f"      After:  {exp2_output.iloc[idx]['summary'][:60]}...")
    except FileNotFoundError:
        print("  ⚠️ Baseline Modular 결과 파일을 찾을 수 없습니다.")
    print()

    # 11. 완료
    print("=" * 80)
    print("✅ Experiment #2 추론 완료!")
    print("=" * 80)
    print()
    print("다음 단계:")
    print("  1. output_modular_v2.csv를 대회 플랫폼에 제출")
    print("  2. 점수 확인 (47.5~48.2 기대)")
    print("  3. 결과를 experiment_logs.md에 기록")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)