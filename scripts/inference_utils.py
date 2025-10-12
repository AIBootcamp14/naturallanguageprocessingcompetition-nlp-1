#!/usr/bin/env python3
"""
추론 관련 유틸리티 모듈

이 모듈은 학습된 BART 모델을 사용하여 대화 요약문을 생성하고,
결과를 후처리하여 CSV 파일로 저장하는 기능을 제공합니다.

주요 함수:
- generate_summaries: 모델을 사용하여 요약문 생성
- postprocess_summaries: 생성된 요약문에서 특수 토큰 제거
- save_predictions: 예측 결과를 CSV 파일로 저장
- run_inference: 전체 추론 파이프라인 실행
"""

import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from typing import List, Tuple


def generate_summaries(model: BartForConditionalGeneration,
                      dataloader: DataLoader,
                      config: dict,
                      device: torch.device) -> Tuple[List[str], List[str]]:
    """
    모델을 사용하여 요약문을 생성합니다.

    baseline.ipynb Cell 42의 inference 함수를 기반으로 작성되었습니다.
    모델의 generate 메서드를 사용하여 beam search 기반 요약문을 생성하며,
    특수 토큰을 포함한 원본 텍스트를 반환합니다 (후처리 단계에서 제거).

    Args:
        model: 학습된 BART 모델 (BartForConditionalGeneration)
        dataloader: Test 데이터 로더 (DatasetForInference 포함)
        config: 설정 딕셔너리 (inference 섹션 필요)
        device: 사용할 디바이스 (cuda:0 또는 cpu)

    Returns:
        Tuple[List[str], List[str]]: (fname_list, summary_list)
            - fname_list: 파일명 리스트 (예: ['test_0', 'test_1', ...])
            - summary_list: 생성된 요약문 리스트 (특수 토큰 포함)

    Example:
        >>> device = torch.device('cuda:0')
        >>> fnames, summaries = generate_summaries(model, test_loader, config, device)
        >>> print(fnames[0])  # 'test_0'
        >>> print(summaries[0])  # '<s>요약문...</s>'
    """
    # 모델을 평가 모드로 전환 (dropout 등 비활성화)
    model.eval()

    summaries = []
    fnames = []

    # gradient 계산 비활성화로 메모리 절약 및 속도 향상
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="추론 진행"):
            # 파일명 수집
            fnames.extend(batch['ID'])

            # Generate - baseline.ipynb Cell 42와 동일한 파라미터 사용
            generated_ids = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],  # 2
                early_stopping=config['inference']['early_stopping'],  # True
                max_length=config['inference']['generate_max_length'],  # 100
                num_beams=config['inference']['num_beams']  # 4
            )

            # Decode - skip_special_tokens=False로 특수 토큰 포함
            # (후처리 단계에서 제거하여 정확한 평가 수행)
            for ids in generated_ids:
                summary = config['tokenizer'].decode(ids, skip_special_tokens=False)
                summaries.append(summary)

    return fnames, summaries


def postprocess_summaries(summaries: List[str], remove_tokens: List[str]) -> List[str]:
    """
    생성된 요약문에서 특수 토큰을 제거합니다.

    baseline.ipynb Cell 42의 후처리 로직을 기반으로 작성되었습니다.
    '<usr>', '<s>', '</s>', '<pad>' 등 노이즈에 해당하는 특수 토큰을
    공백으로 치환하여 정확한 평가를 가능하게 합니다.

    Args:
        summaries: 원본 요약문 리스트 (특수 토큰 포함)
        remove_tokens: 제거할 토큰 리스트
                      (예: ['<usr>', '<s>', '</s>', '<pad>'])

    Returns:
        List[str]: 후처리된 요약문 리스트 (특수 토큰이 공백으로 치환됨)

    Example:
        >>> summaries = ['<s>요약문입니다.</s><pad>']
        >>> remove_tokens = ['<s>', '</s>', '<pad>']
        >>> cleaned = postprocess_summaries(summaries, remove_tokens)
        >>> print(cleaned[0])  # ' 요약문입니다.  '
    """
    # baseline.ipynb Cell 42의 로직과 동일
    cleaned = summaries.copy()
    for token in remove_tokens:
        cleaned = [s.replace(token, " ") for s in cleaned]
    return cleaned


def save_predictions(fnames: List[str], summaries: List[str],
                    output_dir: str, filename: str = "output.csv") -> str:
    """
    예측 결과를 CSV 파일로 저장합니다.

    baseline.ipynb Cell 42의 CSV 저장 로직을 기반으로 작성되었습니다.
    대회 제출 형식에 맞춰 'fname'과 'summary' 컬럼을 포함하며,
    index=False로 저장하여 불필요한 인덱스 컬럼을 제거합니다.

    Args:
        fnames: 파일명 리스트 (예: ['test_0', 'test_1', ...])
        summaries: 요약문 리스트 (후처리 완료된 상태)
        output_dir: 저장 디렉토리 경로
        filename: 파일명 (기본값: "output.csv")

    Returns:
        str: 저장된 파일의 전체 경로

    Example:
        >>> path = save_predictions(
        ...     fnames=['test_0', 'test_1'],
        ...     summaries=['요약1', '요약2'],
        ...     output_dir='./prediction',
        ...     filename='output.csv'
        ... )
        >>> print(path)  # './prediction/output.csv'
    """
    # DataFrame 생성 - baseline.ipynb Cell 42와 동일한 구조
    output_df = pd.DataFrame({
        'fname': fnames,
        'summary': summaries
    })

    # 디렉토리 생성 (없으면 생성)
    os.makedirs(output_dir, exist_ok=True)

    # CSV 파일로 저장 (index=False로 인덱스 컬럼 제거)
    output_path = os.path.join(output_dir, filename)
    output_df.to_csv(output_path, index=False)

    return output_path


def run_inference(model: BartForConditionalGeneration,
                 tokenizer: PreTrainedTokenizerFast,
                 test_dataloader: DataLoader,
                 config: dict,
                 device: torch.device,
                 save_path: str = None) -> pd.DataFrame:
    """
    전체 추론 파이프라인을 실행합니다.

    이 함수는 다음 단계를 순차적으로 수행합니다:
    1. 모델을 사용하여 요약문 생성
    2. 생성된 요약문에서 특수 토큰 제거 (후처리)
    3. 결과를 DataFrame으로 변환
    4. (선택적) CSV 파일로 저장

    Args:
        model: 학습된 BART 모델
        tokenizer: 사용할 tokenizer (디코딩에 필요)
        test_dataloader: Test 데이터 로더
        config: 설정 딕셔너리 (inference 섹션 필요)
        device: 사용할 디바이스
        save_path: 저장 경로 (None이면 저장하지 않음)
                  예: './prediction/output.csv'

    Returns:
        pd.DataFrame: 예측 결과 DataFrame (컬럼: fname, summary)

    Example:
        >>> result_df = run_inference(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     test_dataloader=test_loader,
        ...     config=loaded_config,
        ...     device=torch.device('cuda:0'),
        ...     save_path='./prediction/output.csv'
        ... )
        >>> print(result_df.head())
        #     fname          summary
        # 0  test_0  요약문입니다...
        # 1  test_1  요약문입니다...
    """
    # 1. 요약 생성
    print("=" * 80)
    print("1단계: 요약 생성 중...")
    print("=" * 80)

    # tokenizer를 config에 임시 저장 (generate_summaries에서 사용)
    config['tokenizer'] = tokenizer
    fnames, raw_summaries = generate_summaries(model, test_dataloader, config, device)

    print(f"✅ {len(fnames)}개의 요약문 생성 완료")
    print(f"   - 첫 번째 파일: {fnames[0]}")
    print(f"   - 원본 요약 예시: {raw_summaries[0][:100]}...")

    # 2. 후처리
    print("\n" + "=" * 80)
    print("2단계: 후처리 중 (특수 토큰 제거)...")
    print("=" * 80)

    remove_tokens = config['inference']['remove_tokens']
    print(f"제거할 토큰: {remove_tokens}")
    cleaned_summaries = postprocess_summaries(raw_summaries, remove_tokens)

    print(f"✅ 후처리 완료")
    print(f"   - 후처리 요약 예시: {cleaned_summaries[0][:100]}...")

    # 3. DataFrame 생성
    result_df = pd.DataFrame({
        'fname': fnames,
        'summary': cleaned_summaries
    })

    print(f"\n✅ DataFrame 생성 완료: {len(result_df)}개 행")

    # 4. 저장 (옵션)
    if save_path:
        print("\n" + "=" * 80)
        print("3단계: 결과 저장 중...")
        print("=" * 80)

        output_dir = os.path.dirname(save_path) or config['inference']['result_path']
        filename = os.path.basename(save_path) or "output.csv"
        saved_path = save_predictions(fnames, cleaned_summaries, output_dir, filename)
        print(f"✅ 저장 완료: {saved_path}")
    else:
        print("\n⚠️  저장 경로가 지정되지 않아 파일로 저장하지 않았습니다.")

    return result_df