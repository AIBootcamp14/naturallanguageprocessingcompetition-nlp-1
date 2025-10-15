#!/usr/bin/env python3
"""
PyTorch Dataset 클래스 모음
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import pandas as pd
import os


class DatasetForTrain(Dataset):
    """
    학습용 Dataset 클래스 (baseline.ipynb Cell 26 참조)
    """

    def __init__(self, encoder_input, decoder_input, labels, length):
        """
        Args:
            encoder_input: Tokenized encoder input (dict with 'input_ids', 'attention_mask')
            decoder_input: Tokenized decoder input (dict with 'input_ids', 'attention_mask')
            labels: Tokenized decoder output (labels)
            length: 데이터셋 길이
        """
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = length

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터 반환

        Returns:
            {
                'input_ids': encoder input_ids,
                'attention_mask': encoder attention_mask,
                'decoder_input_ids': decoder input_ids,
                'decoder_attention_mask': decoder attention_mask,
                'labels': decoder output (ground truth)
            }
        """
        # Encoder input
        item = {key: val[idx].clone().detach()
                for key, val in self.encoder_input.items()}

        # Decoder input
        item2 = {key: val[idx].clone().detach()
                 for key, val in self.decoder_input.items()}

        # 키 이름 변경 (Seq2Seq 형식에 맞게)
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')

        # 합치기
        item.update(item2)

        # Labels 추가
        item['labels'] = self.labels['input_ids'][idx]

        return item

    def __len__(self):
        return self.len


class DatasetForVal(Dataset):
    """
    검증용 Dataset 클래스 (baseline.ipynb Cell 26 참조)
    DatasetForTrain과 동일한 구조
    """

    def __init__(self, encoder_input, decoder_input, labels, length):
        """
        Args:
            encoder_input: Tokenized encoder input
            decoder_input: Tokenized decoder input
            labels: Tokenized decoder output
            length: 데이터셋 길이
        """
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = length

    def __getitem__(self, idx):
        """인덱스에 해당하는 데이터 반환"""
        # Encoder input
        item = {key: val[idx].clone().detach()
                for key, val in self.encoder_input.items()}

        # Decoder input
        item2 = {key: val[idx].clone().detach()
                 for key, val in self.decoder_input.items()}

        # 키 이름 변경
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')

        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]

        return item

    def __len__(self):
        return self.len


class DatasetForInference(Dataset):
    """
    추론용 Dataset 클래스 (baseline.ipynb Cell 26 참조)
    """

    def __init__(self, encoder_input, test_id, length):
        """
        Args:
            encoder_input: Tokenized encoder input
            test_id: 파일명 리스트 (fname)
            length: 데이터셋 길이
        """
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = length

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터 반환

        Returns:
            {
                'input_ids': encoder input_ids,
                'attention_mask': encoder attention_mask,
                'ID': fname (파일명)
            }
        """
        item = {key: val[idx].clone().detach()
                for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len


def prepare_train_dataset(config: dict, preprocessor, data_path: str, tokenizer):
    """
    학습/검증 데이터셋을 준비합니다 (baseline.ipynb Cell 27 참조)

    Args:
        config: 설정 딕셔너리
        preprocessor: Preprocess 클래스 인스턴스
        data_path: 데이터 경로
        tokenizer: 설정된 tokenizer

    Returns:
        (train_dataset, val_dataset)
    """
    # 1. CSV 로드
    train_file_path = os.path.join(data_path, 'train.csv')
    val_file_path = os.path.join(data_path, 'dev.csv')

    train_data = preprocessor.make_set_as_df(train_file_path, is_train=True)
    val_data = preprocessor.make_set_as_df(val_file_path, is_train=True)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"][0]}')
    print(f'val_label:\n {val_data["summary"][0]}')

    # 2. BART 입력 형태로 변환
    encoder_input_train, decoder_input_train, decoder_output_train = \
        preprocessor.make_input(train_data, is_test=False)
    encoder_input_val, decoder_input_val, decoder_output_val = \
        preprocessor.make_input(val_data, is_test=False)

    print('-'*10, 'Load data complete', '-'*10)

    # 3. Tokenization
    # Train encoder
    tokenized_encoder_inputs_train = tokenizer(
        encoder_input_train,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    # Train decoder input
    tokenized_decoder_inputs_train = tokenizer(
        decoder_input_train,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    # Train decoder output (labels)
    tokenized_decoder_outputs_train = tokenizer(
        decoder_output_train,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    # Val encoder
    tokenized_encoder_inputs_val = tokenizer(
        encoder_input_val,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    # Val decoder input
    tokenized_decoder_inputs_val = tokenizer(
        decoder_input_val,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    # Val decoder output (labels)
    tokenized_decoder_outputs_val = tokenizer(
        decoder_output_val,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    # 4. Dataset 생성
    train_inputs_dataset = DatasetForTrain(
        tokenized_encoder_inputs_train,
        tokenized_decoder_inputs_train,
        tokenized_decoder_outputs_train,
        len(encoder_input_train)
    )

    val_inputs_dataset = DatasetForVal(
        tokenized_encoder_inputs_val,
        tokenized_decoder_inputs_val,
        tokenized_decoder_outputs_val,
        len(encoder_input_val)
    )

    print('-'*10, 'Make dataset complete', '-'*10)
    return train_inputs_dataset, val_inputs_dataset


def prepare_test_dataset(config: dict, preprocessor, tokenizer):
    """
    테스트 데이터셋을 준비합니다 (baseline.ipynb Cell 39 참조)

    Args:
        config: 설정 딕셔너리
        preprocessor: Preprocess 클래스 인스턴스
        tokenizer: 설정된 tokenizer

    Returns:
        (test_data_df, test_dataset)
    """
    # 1. CSV 로드
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    # 2. BART 입력 형태로 변환
    encoder_input_test, decoder_input_test = preprocessor.make_input(
        test_data, is_test=True
    )

    print('-'*10, 'Load data complete', '-'*10)

    # 3. Tokenization (Encoder만)
    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    # 4. Dataset 생성
    test_encoder_inputs_dataset = DatasetForInference(
        test_tokenized_encoder_inputs,
        test_id,
        len(encoder_input_test)
    )

    print('-'*10, 'Make dataset complete', '-'*10)
    return test_data, test_encoder_inputs_dataset