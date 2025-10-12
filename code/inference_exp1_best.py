#!/usr/bin/env python3
"""
Experiment #1 재추론 (Best 체크포인트: checkpoint-1996)
"""

import pandas as pd
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import yaml

# Config 로드
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Best 체크포인트 경로 설정
BEST_CHECKPOINT = '/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission/checkpoint-1996'
config['inference']['ckt_path'] = BEST_CHECKPOINT

print("=" * 80)
print("🔄 Experiment #1 재추론 (Best 체크포인트)")
print("=" * 80)
print(f"체크포인트: {BEST_CHECKPOINT}")
print(f"Epoch: 4 (ROUGE-L: 0.2500)")
print("=" * 80)

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\n디바이스: {device}")

# Tokenizer & Model 로드
print("\n모델 로딩 중...")
model_name = config['general']['model_name']
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# Special tokens 추가
special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
tokenizer.add_special_tokens(special_tokens_dict)

# 체크포인트에서 모델 로드
model = BartForConditionalGeneration.from_pretrained(BEST_CHECKPOINT, local_files_only=True)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
print(f"✅ 모델 로드 완료 (vocab size: {len(tokenizer)})")

# Test 데이터 로드
print("\n테스트 데이터 로딩 중...")
test_df = pd.read_csv(os.path.join(config['general']['data_path'], 'test.csv'))
print(f"✅ Test 데이터: {len(test_df)}개")

# Tokenization
encoder_inputs = test_df['dialogue'].tolist()
tokenized = tokenizer(
    encoder_inputs,
    return_tensors="pt",
    padding=True,
    add_special_tokens=True,
    truncation=True,
    max_length=config['tokenizer']['encoder_max_len'],
    return_token_type_ids=False
)

# Dataset class
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, fnames):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.fnames = fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'fname': self.fnames[idx]
        }

dataset = TestDataset(
    tokenized['input_ids'],
    tokenized['attention_mask'],
    test_df['fname'].tolist()
)

dataloader = DataLoader(dataset, batch_size=config['inference']['batch_size'])

# 추론
print("\n추론 시작...")
summaries = []
fnames = []

model.eval()
with torch.no_grad():
    for batch in tqdm(dataloader, desc="추론 진행"):
        fnames.extend(batch['fname'])

        generated_ids = model.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
            early_stopping=config['inference']['early_stopping'],
            max_length=config['inference']['generate_max_length'],
            num_beams=config['inference']['num_beams'],
        )

        for ids in generated_ids:
            summary = tokenizer.decode(ids, skip_special_tokens=False)
            summaries.append(summary)

# 후처리 (특수 토큰 제거)
print("\n후처리 중...")
remove_tokens = config['inference']['remove_tokens']
cleaned_summaries = summaries.copy()

for token in remove_tokens:
    cleaned_summaries = [s.replace(token, " ") for s in cleaned_summaries]

# 결과 저장
output_df = pd.DataFrame({
    'fname': fnames,
    'summary': cleaned_summaries
})

result_path = './prediction/'
os.makedirs(result_path, exist_ok=True)
output_path = os.path.join(result_path, 'output_exp1_best.csv')

output_df.to_csv(output_path, index=False)

print("=" * 80)
print("✅ 추론 완료!")
print("=" * 80)
print(f"출력 파일: {output_path}")
print(f"샘플 수: {len(output_df)}")
print("\n샘플 (처음 3개):")
for i in range(min(3, len(output_df))):
    print(f"  [{i}] {output_df.iloc[i]['fname']}: {output_df.iloc[i]['summary'][:80]}...")
