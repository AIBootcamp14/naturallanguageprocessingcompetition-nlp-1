#!/usr/bin/env python3
"""
Experiment #4: 길이 정규화 + Max Length 768

변경사항:
1. encoder_max_len: 512 → 768 (긴 대화 처리)
2. length_penalty: 0.6 추가 (GNMT 길이 정규화)

예상 효과: +1~2점 (Baseline 46.95 → 48~49점)
소요 시간: ~30분
"""

import sys
sys.path.append('../scripts')

import torch

from utils import load_config, get_device, set_seed
from data_loader import Preprocess
from tokenizer_utils import load_tokenizer
from model_utils import load_model_for_train
from dataset import prepare_train_dataset
from trainer_utils import get_trainer

print("="*80)
print("🚀 Experiment #4: 길이 정규화 + Max Length 768")
print("="*80)
print()
print("변경사항:")
print("  1. encoder_max_len: 512 → 768")
print("  2. length_penalty: 0.6 (GNMT)")
print()
print("예상 효과: +1~2점 (46.95 → 48~49점)")
print("="*80)

# Config - config_exp4.yaml 사용
config = load_config('./config_exp4.yaml')
device = get_device()
set_seed(config['training']['seed'])

print(f"\n✅ Config 로드 완료")
print(f"   Device: {device}")
print(f"   Encoder Max Length: {config['tokenizer']['encoder_max_len']}")
print(f"   Length Penalty: {config['inference'].get('length_penalty', 1.0)}")
print(f"   Learning Rate: {config['training']['learning_rate']}")
print(f"   Epochs: {config['training']['num_train_epochs']}")

# Wandb 비활성화 확인
print(f"   Wandb: {config['training'].get('report_to', 'none')}")

# Tokenizer
tokenizer = load_tokenizer(
    config['general']['model_name'],
    config['tokenizer']['special_tokens']
)
print(f"\n✅ Tokenizer 로드 완료 (vocab size: {len(tokenizer)})")

# Dataset
preprocessor = Preprocess(
    bos_token=config['tokenizer']['bos_token'],
    eos_token=config['tokenizer']['eos_token']
)

data_path = config['general']['data_path']
train_dataset, val_dataset = prepare_train_dataset(
    config, preprocessor, data_path, tokenizer
)

print(f"\n✅ 데이터셋 준비 완료")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Val: {len(val_dataset)} samples")

# Model
model = load_model_for_train(config, tokenizer, device)
print(f"\n✅ 모델 로드 완료")

# Trainer
trainer = get_trainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

print(f"\n✅ Trainer 설정 완료")

# Train
print("\n" + "="*80)
print("🚀 학습 시작... (약 30분 소요)")
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print("✅ 학습 완료!")
print("="*80)

# Save final model info
print(f"\n📁 모델 저장 위치: {config['general']['output_dir']}")
print(f"📊 Best checkpoint를 사용하여 추론을 진행하세요")
print(f"\n다음 단계:")
print(f"1. python infer_test_exp4.py 실행")
print(f"2. prediction/exp4_output.csv 제출")
print(f"3. 점수 비교 (Baseline 46.95 vs Exp #4)")
print("="*80)
