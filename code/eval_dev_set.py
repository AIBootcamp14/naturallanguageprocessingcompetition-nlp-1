#!/usr/bin/env python3
"""
Dev Set 평가 스크립트

checkpoint로 Dev set ROUGE 점수를 계산합니다.
Baseline과 Exp 비교를 위한 기준선 확보용.
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
from dataset import DatasetForVal
from inference_utils import postprocess_summaries


def evaluate_dev_set(checkpoint_path, config_path='./config.yaml'):
    """
    Dev set으로 모델 평가

    Args:
        checkpoint_path: 모델 checkpoint 경로
        config_path: config.yaml 경로

    Returns:
        dict: ROUGE 점수
    """
    # 1. Config 로드
    print(f"📁 Config 로드: {config_path}")
    config = load_config(config_path)
    device = get_device()
    set_seed(config['training']['seed'])

    # 2. Tokenizer 로드
    print(f"🔤 Tokenizer 로드: {config['general']['model_name']}")
    tokenizer = load_tokenizer(
        config['general']['model_name'],
        config['tokenizer']['special_tokens']
    )

    # 3. 모델 로드
    print(f"🤖 모델 로드: {checkpoint_path}")
    model = load_model_for_inference(checkpoint_path, tokenizer, device)
    model.eval()

    # 4. Dev 데이터 로드
    import os
    dev_path = os.path.join(config['general']['data_path'], 'dev.csv')
    print(f"📊 Dev set 로드: {dev_path}")
    preprocessor = Preprocess(
        bos_token=config['tokenizer']['bos_token'],
        eos_token=config['tokenizer']['eos_token']
    )

    dev_data = pd.read_csv(dev_path)
    dev_dataset = DatasetForVal(dev_data, tokenizer, preprocessor)

    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False
    )

    # 5. 추론 실행
    print(f"\n🔮 Dev set 추론 시작 ({len(dev_data)} samples)...")
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
                early_stopping=config['inference']['early_stopping'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size']
            )

            # Decode
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            predictions.extend(decoded)

            # Ground truth
            labels = batch['labels']
            labels = labels.masked_fill(labels == -100, tokenizer.pad_token_id)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=False)
            references.extend(refs)

    # 6. 후처리
    print(f"🧹 후처리 적용...")
    predictions = postprocess_summaries(
        predictions,
        config['inference']['remove_tokens']
    )
    references = postprocess_summaries(
        references,
        config['inference']['remove_tokens']
    )

    # 7. ROUGE 계산
    print(f"\n📈 ROUGE 점수 계산...")
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)

    # 8. 결과 출력
    print("\n" + "="*60)
    print("📊 Dev Set ROUGE 점수")
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

    parser = argparse.ArgumentParser(description='Dev set 평가')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='모델 checkpoint 경로')
    parser.add_argument('--config', type=str, default='./config.yaml',
                       help='config.yaml 경로')

    args = parser.parse_args()

    scores = evaluate_dev_set(args.checkpoint, args.config)