#!/usr/bin/env python3
"""
Dev Set 평가 스크립트 (간단 버전)

checkpoint로 Dev set ROUGE 점수를 빠르게 계산합니다.
"""

import sys
sys.path.append('../scripts')

import torch
from rouge import Rouge
import pandas as pd
from tqdm import tqdm

from utils import load_config, get_device, set_seed
from data_loader import Preprocess
from tokenizer_utils import load_tokenizer
from model_utils import load_model_for_inference
from inference_utils import postprocess_summaries


def evaluate_dev_set(checkpoint_path, config_path='./config.yaml'):
    # Config & Device
    config = load_config(config_path)
    device = get_device()
    set_seed(config['training']['seed'])

    # Tokenizer
    tokenizer = load_tokenizer(
        config['general']['model_name'],
        config['tokenizer']['special_tokens']
    )

    # Model
    print(f"🤖 모델 로드: {checkpoint_path}")
    model = load_model_for_inference(checkpoint_path, tokenizer, device)
    model.eval()

    # Dev data
    import os
    dev_path = os.path.join(config['general']['data_path'], 'dev.csv')
    print(f"📊 Dev set 로드: {dev_path}")
    dev_data = pd.read_csv(dev_path)

    # Preprocessor
    preprocessor = Preprocess(
        bos_token=config['tokenizer']['bos_token'],
        eos_token=config['tokenizer']['eos_token']
    )

    # Inference
    print(f"\n🔮 Dev set 추론 시작 ({len(dev_data)} samples)...")
    predictions = []
    references = []

    batch_size = config['inference']['batch_size']

    for i in tqdm(range(0, len(dev_data), batch_size), desc="Batches"):
        batch_data = dev_data.iloc[i:i+batch_size]

        # Preprocess
        encoder_inputs = []
        for _, row in batch_data.iterrows():
            text = preprocessor.make_input(row)
            encoder_inputs.append(text)

        # Tokenize
        inputs = tokenizer(
            encoder_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config['tokenizer']['encoder_max_len']
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
                early_stopping=config['inference']['early_stopping'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size']
            )

        # Decode predictions
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        predictions.extend(decoded)

        # References
        refs = batch_data['output'].tolist()
        references.extend(refs)

    # 후처리
    print(f"🧹 후처리 적용...")
    predictions = postprocess_summaries(
        predictions,
        config['inference']['remove_tokens']
    )

    # ROUGE 계산
    print(f"\n📈 ROUGE 점수 계산...")
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)

    # 결과 출력
    print("\n" + "="*60)
    print("📊 Baseline Dev Set ROUGE 점수")
    print("="*60)
    print(f"ROUGE-1 F1:  {scores['rouge-1']['f']*100:.2f}%")
    print(f"ROUGE-2 F1:  {scores['rouge-2']['f']*100:.2f}%")
    print(f"ROUGE-L F1:  {scores['rouge-l']['f']*100:.2f}%")
    print("-"*60)
    avg_score = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3 * 100
    print(f"평균 점수:    {avg_score:.4f}")
    print("="*60)

    return {
        'rouge-1': scores['rouge-1']['f'] * 100,
        'rouge-2': scores['rouge-2']['f'] * 100,
        'rouge-l': scores['rouge-l']['f'] * 100,
        'avg': avg_score
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='./config.yaml')
    args = parser.parse_args()

    scores = evaluate_dev_set(args.checkpoint, args.config)
