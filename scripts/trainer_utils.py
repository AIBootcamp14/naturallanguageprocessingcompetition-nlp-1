#!/usr/bin/env python3
"""
학습 관련 유틸리티
"""

import torch
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    BartForConditionalGeneration
)
from torch.utils.data import Dataset
from rouge import Rouge


def compute_metrics(config: dict, tokenizer: PreTrainedTokenizerFast, pred):
    """
    ROUGE 메트릭을 계산합니다.

    Args:
        config: 설정 딕셔너리
        tokenizer: 사용할 tokenizer
        pred: Trainer의 예측 결과 (EvalPrediction)

    Returns:
        ROUGE F1 scores 딕셔너리
    """
    # baseline.ipynb Cell 29 참조
    rouge = Rouge()

    predictions = pred.predictions
    labels = pred.label_ids

    # -100을 pad_token_id로 변환
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # Decode
    decoded_preds = tokenizer.batch_decode(
        predictions,
        clean_up_tokenization_spaces=True
    )
    decoded_labels = tokenizer.batch_decode(
        labels,
        clean_up_tokenization_spaces=True
    )

    # 특수 토큰 제거
    remove_tokens = config['inference']['remove_tokens']
    replaced_predictions = decoded_preds.copy()
    replaced_labels = decoded_labels.copy()

    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]

    # 첫 3개 샘플 출력
    print('-' * 150)
    print(f"PRED: {replaced_predictions[0]}")
    print(f"GOLD: {replaced_labels[0]}")
    print('-' * 150)
    print(f"PRED: {replaced_predictions[1]}")
    print(f"GOLD: {replaced_labels[1]}")
    print('-' * 150)
    print(f"PRED: {replaced_predictions[2]}")
    print(f"GOLD: {replaced_labels[2]}")

    # ROUGE 계산
    try:
        results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)
        # F1 score만 반환
        result = {key: value["f"] for key, value in results.items()}
    except Exception as e:
        print(f"ROUGE 계산 오류: {e}")
        result = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    return result


def get_training_arguments(config: dict) -> Seq2SeqTrainingArguments:
    """
    학습 인자를 생성합니다.

    Args:
        config: 설정 딕셔너리

    Returns:
        Seq2SeqTrainingArguments
    """
    # baseline.ipynb Cell 30 참조
    training_config = config['training']

    return Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=training_config['overwrite_output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        learning_rate=training_config['learning_rate'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        warmup_ratio=training_config['warmup_ratio'],
        weight_decay=training_config['weight_decay'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        optim=training_config['optim'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        eval_strategy=training_config['evaluation_strategy'],  # evaluation_strategy → eval_strategy (transformers 최신 버전)
        save_strategy=training_config['save_strategy'],
        save_total_limit=training_config['save_total_limit'],
        fp16=training_config['fp16'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        seed=training_config['seed'],
        logging_dir=training_config['logging_dir'],
        logging_strategy=training_config['logging_strategy'],
        predict_with_generate=training_config['predict_with_generate'],
        generation_max_length=training_config['generation_max_length'],
        do_train=training_config['do_train'],
        do_eval=training_config['do_eval'],
        report_to=training_config.get('report_to', 'none')  # wandb 또는 none
    )


def get_trainer(config: dict, model: BartForConditionalGeneration,
                tokenizer: PreTrainedTokenizerFast,
                train_dataset: Dataset, val_dataset: Dataset) -> Seq2SeqTrainer:
    """
    Seq2SeqTrainer를 생성합니다.

    Args:
        config: 설정 딕셔너리
        model: 학습할 모델
        tokenizer: 사용할 tokenizer
        train_dataset: 학습 데이터셋
        val_dataset: 검증 데이터셋

    Returns:
        설정된 Seq2SeqTrainer
    """
    # baseline.ipynb Cell 30 참조
    print('-' * 10, 'Make training arguments', '-' * 10)
    training_args = get_training_arguments(config)
    print('-' * 10, 'Make training arguments complete', '-' * 10)

    # Early stopping callback
    print('-' * 10, 'Make trainer', '-' * 10)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    # Trainer 생성
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[early_stopping]
    )

    print('-' * 10, 'Make trainer complete', '-' * 10)
    return trainer