#!/usr/bin/env python3
"""
데이터 로딩 및 전처리 모듈
"""

import pandas as pd
import os
from typing import Tuple, List


class Preprocess:
    """
    데이터 전처리 클래스 (baseline.ipynb Cell 25 참조)
    """

    def __init__(self, bos_token: str, eos_token: str) -> None:
        """
        전처리 클래스 초기화

        Args:
            bos_token: Beginning of sequence 토큰
            eos_token: End of sequence 토큰
        """
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path: str, is_train: bool = True) -> pd.DataFrame:
        """
        CSV 파일을 DataFrame으로 로드합니다.

        Args:
            file_path: CSV 파일 경로
            is_train: Train/Val 데이터 여부 (False면 Test)

        Returns:
            필요한 컬럼만 포함된 DataFrame
        """
        # baseline.ipynb Cell 25 참조
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname', 'dialogue', 'summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname', 'dialogue']]
            return test_df

    def make_input(self, dataset: pd.DataFrame, is_test: bool = False) -> Tuple:
        """
        BART 입력 형태로 데이터를 변환합니다.

        Args:
            dataset: 입력 DataFrame
            is_test: Test 데이터 여부

        Returns:
            Train/Val: (encoder_input_list, decoder_input_list, decoder_output_list)
            Test: (encoder_input_list, decoder_input_list)
        """
        # baseline.ipynb Cell 25 참조
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            # Ground truth를 디코더의 input으로 사용하여 학습합니다.
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()


def load_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train, Dev, Test 데이터를 로드합니다.

    Args:
        config: 설정 딕셔너리

    Returns:
        (train_df, dev_df, test_df)
    """
    # baseline.ipynb Cell 21-23 참조
    data_path = config['general']['data_path']

    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(data_path, 'dev.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

    return train_df, dev_df, test_df