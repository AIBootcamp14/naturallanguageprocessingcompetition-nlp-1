#!/usr/bin/env python3
"""
Dev Set 평가 스크립트 (Baseline 패턴 준수)

checkpoint로 Dev set ROUGE 점수를 계산합니다.
baseline_modular.ipynb의 정확한 패턴을 따릅니다.
"""

import sys
sys.path.append('../scripts')

import os
import torch
from torch.utils.data import DataLoader
from rouge import Rouge
from tqdm import tqdm

from utils import load_config, get_device, set_seed
from data_loader import Preprocess
from tokenizer_utils import load_tokenizer
from model_utils import load_model_for_inference
from dataset import DatasetForVal
from inference_utils import postprocess_summaries


def evaluate_dev_set(checkpoint_path, config_path='./config.yaml'):
    """
    Dev set으로 모델 평가 (baseline_modular 패턴 준수)
    """
    print("="*80)
    print("📊 Baseline Dev Set 평가")
    print("="*80)

    # 1. Config & Device
    config = load_config(config_path)
    device = get_device()
    set_seed(config['training']['seed'])
    print(f"✅ Config 로드 완료")
    print(f"   Device: {device}")

    # 2. Tokenizer
    tokenizer = load_tokenizer(
        config['general']['model_name'],
        config['tokenizer']['special_tokens']
    )
    print(f"✅ Tokenizer 로드 완료")
    print(f"   Vocab size: {len(tokenizer)}")

    # 3. Preprocessor
    preprocessor = Preprocess(
        bos_token=config['tokenizer']['bos_token'],
        eos_token=config['tokenizer']['eos_token']
    )

    # 4. Dev 데이터 로드 및 전처리 (prepare_train_dataset 패턴)
    dev_file_path = os.path.join(config['general']['data_path'], 'dev.csv')
    print(f"\n📂 Dev set 로드: {dev_file_path}")

    # CSV 로드 및 전처리
    dev_data = preprocessor.make_set_as_df(dev_file_path, is_train=True)
    print(f"✅ Dev 데이터: {len(dev_data)} samples")

    # BART 입력 형태로 변환
    encoder_input_dev, decoder_input_dev, decoder_output_dev = \
        preprocessor.make_input(dev_data, is_test=False)

    # 5. Tokenization (prepare_train_dataset 패턴)
    print(f"\n🔤 Tokenization 시작...")

    # Encoder input
    tokenized_encoder_inputs = tokenizer(
        encoder_input_dev,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    # Decoder input
    tokenized_decoder_inputs = tokenizer(
        decoder_input_dev,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    # Decoder output (labels)
    tokenized_decoder_outputs = tokenizer(
        decoder_output_dev,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    print(f"✅ Tokenization 완료")

    # 6. Dataset 생성 (prepare_train_dataset 패턴)
    dev_dataset = DatasetForVal(
        tokenized_encoder_inputs,
        tokenized_decoder_inputs,
        tokenized_decoder_outputs,
        len(encoder_input_dev)
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False
    )

    print(f"✅ Dataset 생성 완료 ({len(dev_dataset)} samples)")

    # 7. 모델 로드
    print(f"\n🤖 모델 로드: {checkpoint_path}")
    model = load_model_for_inference(checkpoint_path, tokenizer, device)
    model.eval()
    print(f"✅ 모델 로드 완료")

    # 8. 추론 실행
    print(f"\n🔮 Dev set 추론 시작...")
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Inference"):
            # Encoder input
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

            # Decode predictions
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            predictions.extend(decoded)

            # Decode references (labels)
            labels = batch['labels']
            # -100을 pad_token_id로 변경
            labels = labels.masked_fill(labels == -100, tokenizer.pad_token_id)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=False)
            references.extend(refs)

    print(f"✅ 추론 완료 ({len(predictions)} samples)")

    # 9. 후처리 (baseline 패턴)
    print(f"\n🧹 후처리 적용...")
    predictions = postprocess_summaries(
        predictions,
        config['inference']['remove_tokens']
    )
    references = postprocess_summaries(
        references,
        config['inference']['remove_tokens']
    )

    # 10. ROUGE 계산
    print(f"\n📈 ROUGE 점수 계산...")
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)

    # 11. 결과 출력
    print("\n" + "="*80)
    print("📊 Baseline Dev Set ROUGE 점수")
    print("="*80)
    print(f"ROUGE-1 F1:  {scores['rouge-1']['f']*100:.2f}%")
    print(f"ROUGE-2 F1:  {scores['rouge-2']['f']*100:.2f}%")
    print(f"ROUGE-L F1:  {scores['rouge-l']['f']*100:.2f}%")
    print("-"*80)
    avg_score = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3 * 100
    print(f"평균 점수:    {avg_score:.4f}")
    print("="*80)

    return {
        'rouge-1': scores['rouge-1']['f'] * 100,
        'rouge-2': scores['rouge-2']['f'] * 100,
        'rouge-l': scores['rouge-l']['f'] * 100,
        'avg': avg_score
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='모델 checkpoint 경로')
    parser.add_argument('--config', default='./config.yaml', help='config.yaml 경로')
    args = parser.parse_args()

    scores = evaluate_dev_set(args.checkpoint, args.config)
