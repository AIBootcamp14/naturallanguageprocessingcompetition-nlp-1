# src/inference/debug_regenerate_predictions.py
"""
학습된 모델의 예측 결과를 재생성하고 검증하는 디버깅 스크립트

주요 기능:
- 저장된 모델을 로드하여 예측 재생성
- 기존 제출 파일과 새로운 예측 결과 비교
- 생성 파라미터 조정 효과 검증
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import os

# ------------------------- 서드파티 라이브러리 ------------------------- #
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from omegaconf import DictConfig

# ------------------------- 프로젝트 모듈 ------------------------- #
from src.inference.predictor import create_predictor


# ==================== 메인 실행부 ==================== #
# ---------------------- 모델 경로 탐색 ---------------------- #
model_path = 'experiments/20251013/20251013_205042_strategy6_kobart_solar_api/model_0_kobart/default/final_model'

# -------------- 모델 경로 존재 여부 확인 -------------- #
if not os.path.exists(model_path):
    # fallback 경로 탐색 (experiments 폴더에서 final_model 검색)
    for root, dirs, files in os.walk('experiments'):
        if 'final_model' in dirs:
            model_path = os.path.join(root, 'final_model')
            break

print('Using model path:', model_path)


# ---------------------- 모델 및 토크나이저 로드 ---------------------- #
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# ---------------------- Predictor 생성 및 설정 ---------------------- #
# 긴 요약 생성을 위한 설정 (max_new_tokens=200)
cfg = DictConfig({
    'tokenizer': {
        'encoder_max_len': 512
    },
    'inference': {
        'generate_max_length': 512,
        'generate_max_new_tokens': 200,
        'num_beams': 5,
        'no_repeat_ngram_size': 3
    }
})

predictor = create_predictor(model, tokenizer, config=cfg)


# ---------------------- 테스트 데이터 로드 ---------------------- #
test_df = pd.read_csv('data/raw/test.csv')
sub_df = pd.read_csv('submissions/20251013/20251013_205042_strategy6_kobart_solar_api.csv')


# ---------------------- 예측 결과 비교 및 검증 ---------------------- #
n = min(10, len(test_df))
print(f'Checking first {n} samples')

# -------------- 샘플별 예측 수행 -------------- #
for i in range(n):
    # fname 추출
    fname = test_df.loc[i, 'fname']
    dialogue = test_df.loc[i, 'dialogue']

    # 기존 예측 결과 가져오기 (제출 파일은 fname 컬럼 사용)
    old_summary = sub_df.loc[sub_df['fname']==fname, 'summary'].values[0] if (fname in sub_df['fname'].values) else ''

    # 새로운 예측 생성
    new_summary = predictor.predict_single(dialogue, max_new_tokens=200, num_beams=5)

    # 결과 출력
    print('---')
    print('fname:', fname)
    print('old:', old_summary)
    print('new:', new_summary)

print('Done')
