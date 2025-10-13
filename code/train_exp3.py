#!/usr/bin/env python3
"""
Experiment #3: Learning Rate 2e-5

Baseline에서 learning rate만 1e-5 → 2e-5로 변경
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
print("🚀 Experiment #3: Learning Rate 2e-5")
print("="*80)

# Config
config = load_config('./config.yaml')
device = get_device()
set_seed(config['training']['seed'])

print(f"✅ Config 로드 완료")
print(f"   Device: {device}")
print(f"   Learning Rate: {config['training']['learning_rate']}")
print(f"   Epochs: {config['training']['num_train_epochs']}")

# Wandb 비활성화
config['training']['report_to'] = 'none'
print(f"   Wandb: 비활성화")

# Tokenizer
tokenizer = load_tokenizer(
    config['general']['model_name'],
    config['tokenizer']['special_tokens']
)
print(f"✅ Tokenizer 로드 완료 (vocab size: {len(tokenizer)})")

# Dataset
preprocessor = Preprocess(
    bos_token=config['tokenizer']['bos_token'],
    eos_token=config['tokenizer']['eos_token']
)

data_path = config['general']['data_path']
train_dataset, val_dataset = prepare_train_dataset(
    config, preprocessor, data_path, tokenizer
)

print(f"✅ 데이터셋 준비 완료")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Val: {len(val_dataset)} samples")

# Model
model = load_model_for_train(config, tokenizer, device)
print(f"✅ 모델 로드 완료")

# Trainer
trainer = get_trainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

print(f"✅ Trainer 설정 완료")

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
print(f"1. Dev set 평가")
print(f"2. Baseline Dev (26.79%)와 비교")
print(f"3. Test 제출 여부 결정")
