"""
Dev Set 평가 및 에러 분석 스크립트
"""
import pandas as pd
import numpy as np
from rouge import Rouge
import yaml
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import json
import os
import re

def find_best_checkpoint(base_path):
    """
    Best checkpoint 찾기
    우선순위:
    1. checkpoint-best 또는 'best'가 포함된 폴더
    2. checkpoint-{숫자} 중 가장 큰 숫자
    3. None (baseline 사용)
    """
    checkpoint_dir = os.path.join(base_path, 'checkpoint')

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = os.listdir(checkpoint_dir)

    # 1. Best checkpoint 찾기
    for ckpt in checkpoints:
        if 'best' in ckpt.lower():
            full_path = os.path.join(checkpoint_dir, ckpt)
            if os.path.isdir(full_path):
                return full_path

    # 2. 가장 큰 숫자 checkpoint 찾기
    numbered = []
    for ckpt in checkpoints:
        match = re.search(r'checkpoint-(\d+)', ckpt)
        if match:
            num = int(match.group(1))
            full_path = os.path.join(checkpoint_dir, ckpt)
            if os.path.isdir(full_path):
                numbered.append((num, full_path))

    if numbered:
        numbered.sort(reverse=True)  # 내림차순 정렬
        return numbered[0][1]  # 가장 큰 숫자의 경로 반환

    return None

# Config 로드
with open('../code/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 데이터 로드
dev_df = pd.read_csv(f"{config['general']['data_path']}/dev.csv")
print(f"Dev set: {len(dev_df)} samples")

# 모델 및 토크나이저 로드
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = config['general']['model_name']

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Special tokens 추가
special_tokens = config['tokenizer']['special_tokens']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# Best checkpoint 찾기 및 모델 로드
print("\nSearching for best checkpoint...")
checkpoint_path = find_best_checkpoint('../code')

if checkpoint_path:
    print(f"Found checkpoint: {checkpoint_path}")
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    print(f"✅ Loaded trained checkpoint: {os.path.basename(checkpoint_path)}")
else:
    print("No checkpoint found, using baseline model")
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Loaded baseline model (not trained)")

model.to(device)
model.eval()

# 예측 수행
print("\nGenerating predictions...")
predictions = []
references = []

for idx, row in tqdm(dev_df.iterrows(), total=len(dev_df)):
    dialogue = row['dialogue']
    summary = row['summary']

    # Tokenize
    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        max_length=config['tokenizer']['encoder_max_len'],
        truncation=True,
        padding=True
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            num_beams=config['inference']['num_beams'],
            max_length=config['inference']['generate_max_length'],
            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
            early_stopping=config['inference']['early_stopping']
        )

    # Decode
    pred = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Remove special tokens
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        pred = pred.replace(token, " ")

    pred = pred.strip()

    predictions.append(pred)
    references.append(summary)

# ROUGE 평가
print("\nCalculating ROUGE scores...")
rouge = Rouge()

# 전체 평가
overall_scores = rouge.get_scores(predictions, references, avg=True)

print("\n" + "="*50)
print("Overall ROUGE Scores (Dev Set)")
print("="*50)
for metric, values in overall_scores.items():
    print(f"{metric}:")
    print(f"  Precision: {values['p']:.4f}")
    print(f"  Recall:    {values['r']:.4f}")
    print(f"  F1:        {values['f']:.4f}")

final_score = (overall_scores['rouge-1']['f'] +
               overall_scores['rouge-2']['f'] +
               overall_scores['rouge-l']['f']) / 3 * 100

print(f"\nFinal Score: {final_score:.4f}")

# 샘플별 점수
print("\nCalculating per-sample scores...")
sample_scores = []
for i, (pred, ref) in enumerate(zip(predictions, references)):
    try:
        scores = rouge.get_scores(pred, ref)[0]
        sample_scores.append({
            'idx': i,
            'fname': dev_df.iloc[i]['fname'],
            'rouge-1-f': scores['rouge-1']['f'],
            'rouge-2-f': scores['rouge-2']['f'],
            'rouge-l-f': scores['rouge-l']['f'],
            'avg_f': (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3,
            'prediction': pred,
            'reference': ref,
            'dialogue': dev_df.iloc[i]['dialogue'],
            'dialogue_len': len(dev_df.iloc[i]['dialogue']),
            'pred_len': len(pred),
            'ref_len': len(ref)
        })
    except:
        sample_scores.append({
            'idx': i,
            'fname': dev_df.iloc[i]['fname'],
            'rouge-1-f': 0,
            'rouge-2-f': 0,
            'rouge-l-f': 0,
            'avg_f': 0,
            'prediction': pred,
            'reference': ref,
            'dialogue': dev_df.iloc[i]['dialogue'],
            'dialogue_len': len(dev_df.iloc[i]['dialogue']),
            'pred_len': len(pred),
            'ref_len': len(ref)
        })

# DataFrame으로 변환
scores_df = pd.DataFrame(sample_scores)

# 에러 분석
print("\n" + "="*50)
print("Error Analysis")
print("="*50)

# 최악 샘플
worst_samples = scores_df.nsmallest(10, 'avg_f')
print("\n[Top 10 Worst Samples]")
for _, row in worst_samples.iterrows():
    print(f"\nfname: {row['fname']}")
    print(f"ROUGE-1/2/L: {row['rouge-1-f']:.3f} / {row['rouge-2-f']:.3f} / {row['rouge-l-f']:.3f}")
    print(f"Dialogue length: {row['dialogue_len']}")
    print(f"Prediction: {row['prediction'][:100]}...")
    print(f"Reference:  {row['reference'][:100]}...")

# 통계
print("\n" + "="*50)
print("Statistics")
print("="*50)
print(f"Mean ROUGE-1: {scores_df['rouge-1-f'].mean():.4f}")
print(f"Mean ROUGE-2: {scores_df['rouge-2-f'].mean():.4f}")
print(f"Mean ROUGE-L: {scores_df['rouge-l-f'].mean():.4f}")
print(f"Mean Avg:     {scores_df['avg_f'].mean():.4f}")

print(f"\nStd ROUGE-1: {scores_df['rouge-1-f'].std():.4f}")
print(f"Std ROUGE-2: {scores_df['rouge-2-f'].std():.4f}")
print(f"Std ROUGE-L: {scores_df['rouge-l-f'].std():.4f}")

# 길이별 분석
print("\n" + "="*50)
print("Performance by Dialogue Length")
print("="*50)

scores_df['length_bin'] = pd.cut(scores_df['dialogue_len'],
                                   bins=[0, 300, 500, 700, float('inf')],
                                   labels=['short (<300)', 'medium (300-500)', 'long (500-700)', 'very_long (700+)'])

for length_bin in scores_df['length_bin'].cat.categories:
    bin_data = scores_df[scores_df['length_bin'] == length_bin]
    if len(bin_data) > 0:
        print(f"\n{length_bin} ({len(bin_data)} samples):")
        print(f"  ROUGE-1: {bin_data['rouge-1-f'].mean():.4f}")
        print(f"  ROUGE-2: {bin_data['rouge-2-f'].mean():.4f}")
        print(f"  ROUGE-L: {bin_data['rouge-l-f'].mean():.4f}")

# 결과 저장
output_dir = '../analysis'
import os
os.makedirs(output_dir, exist_ok=True)

scores_df.to_csv(f'{output_dir}/dev_evaluation_results.csv', index=False)
print(f"\n✅ Results saved to {output_dir}/dev_evaluation_results.csv")

# JSON으로도 저장 (상세 분석용)
analysis_results = {
    'overall_scores': overall_scores,
    'final_score': final_score,
    'statistics': {
        'mean_rouge1': float(scores_df['rouge-1-f'].mean()),
        'mean_rouge2': float(scores_df['rouge-2-f'].mean()),
        'mean_rougeL': float(scores_df['rouge-l-f'].mean()),
        'std_rouge1': float(scores_df['rouge-1-f'].std()),
        'std_rouge2': float(scores_df['rouge-2-f'].std()),
        'std_rougeL': float(scores_df['rouge-l-f'].std()),
    },
    'worst_samples': worst_samples[['fname', 'avg_f', 'rouge-1-f', 'rouge-2-f', 'rouge-l-f']].to_dict('records')
}

with open(f'{output_dir}/dev_evaluation_summary.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2, ensure_ascii=False)

print(f"✅ Summary saved to {output_dir}/dev_evaluation_summary.json")
print("\n" + "="*50)
print("Evaluation Complete!")
print("="*50)