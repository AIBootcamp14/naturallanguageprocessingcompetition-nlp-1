import os
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.inference.predictor import create_predictor

# 모델 경로 (로그에서 확인된 경로)
model_path = 'experiments/20251013/20251013_205042_strategy6_kobart_solar_api/model_0_kobart/default/final_model'
if not os.path.exists(model_path):
    # fallback: try to find any final_model under experiments
    for root, dirs, files in os.walk('experiments'):
        if 'final_model' in dirs:
            model_path = os.path.join(root, 'final_model')
            break

print('Using model path:', model_path)

# load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# create predictor with config that forces longer generation
from omegaconf import DictConfig
cfg = DictConfig({'tokenizer': {'encoder_max_len': 512}, 'inference': {'generate_max_length': 512, 'generate_max_new_tokens': 200, 'num_beams': 5, 'no_repeat_ngram_size': 3}})
predictor = create_predictor(model, tokenizer, config=cfg)

# load test dialogues
test_df = pd.read_csv('data/raw/test.csv')
# read existing submission for comparison
sub_df = pd.read_csv('submissions/20251013/20251013_205042_strategy6_kobart_solar_api.csv')

n = min(10, len(test_df))
print(f'Checking first {n} samples')
for i in range(n):
    did = test_df.loc[i, 'id'] if 'id' in test_df.columns else f'test_{i}'
    dialogue = test_df.loc[i, 'dialogue']
    old_summary = sub_df.loc[sub_df['id']==did, 'summary'].values[0] if (did in sub_df['id'].values) else ''
    new_summary = predictor.predict_single(dialogue, max_new_tokens=200, num_beams=5)
    print('---')
    print('id:', did)
    print('old:', old_summary)
    print('new:', new_summary)

print('Done')
