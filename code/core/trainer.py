#!/usr/bin/env python3
"""
학습 모듈
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Optional
from transformers import Seq2SeqTrainer, BartForConditionalGeneration, PreTrainedTokenizerFast
from trainer_utils import get_trainer
from dataset import DatasetForTrain, DatasetForVal


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    WeightedRandomSampler를 지원하는 Custom Trainer (train_exp7f.py 참조)
    """

    def __init__(self, *args, train_sampler: Optional[WeightedRandomSampler] = None, **kwargs):
        """
        WeightedSeq2SeqTrainer 초기화

        Args:
            train_sampler: WeightedRandomSampler (None이면 기본 샘플러 사용)
        """
        super().__init__(*args, **kwargs)
        self.train_sampler = train_sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        학습 데이터로더를 반환합니다 (WeightedRandomSampler 적용)

        Returns:
            DataLoader
        """
        if self.train_sampler is None:
            return super().get_train_dataloader()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class Trainer:
    """
    학습 파이프라인 관리 클래스
    """

    def __init__(self, config: Dict, experiment_name: str):
        """
        Trainer 초기화

        Args:
            config: 설정 딕셔너리
            experiment_name: 실험 이름
        """
        self.config = config
        self.experiment_name = experiment_name
        self.device = self._get_device()

        # scripts 경로 추가
        scripts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        if scripts_path not in sys.path:
            sys.path.append(scripts_path)

    def _get_device(self) -> torch.device:
        """
        사용할 디바이스를 반환합니다

        Returns:
            torch.device
        """
        from utils import get_device
        return get_device()

    def train(self, model: BartForConditionalGeneration, tokenizer: PreTrainedTokenizerFast,
             train_dataset: DatasetForTrain, val_dataset: DatasetForVal,
             sampler: Optional[WeightedRandomSampler] = None):
        """
        전체 학습 파이프라인을 실행합니다

        Args:
            model: 학습할 모델
            tokenizer: 토크나이저
            train_dataset: 학습 데이터셋
            val_dataset: 검증 데이터셋
            sampler: WeightedRandomSampler (선택적)
        """
        print("\n" + "=" * 80)
        print("🚀 Trainer 설정 중...")
        print("=" * 80)

        # scripts/trainer_utils.py의 get_trainer 활용
        base_trainer = get_trainer(
            config=self.config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )

        # WeightedSeq2SeqTrainer로 변환 (가중치 사용 시)
        if sampler is not None:
            print("\n✅ WeightedSeq2SeqTrainer 적용 (가중치 샘플링)")
            trainer = WeightedSeq2SeqTrainer(
                model=base_trainer.model,
                args=base_trainer.args,
                data_collator=base_trainer.data_collator,
                train_dataset=base_trainer.train_dataset,
                eval_dataset=base_trainer.eval_dataset,
                tokenizer=base_trainer.tokenizer,
                compute_metrics=base_trainer.compute_metrics if hasattr(base_trainer, 'compute_metrics') else None,
                callbacks=base_trainer.callback_handler.callbacks if hasattr(base_trainer, 'callback_handler') else None,
                train_sampler=sampler
            )
        else:
            print("\n✅ Seq2SeqTrainer 적용 (자연 분포)")
            trainer = base_trainer

        print("=" * 80)

        # 학습 시작
        print("\n" + "=" * 80)
        print("🚀 학습 시작...")
        print("=" * 80)

        try:
            trainer.train()

            print("\n" + "=" * 80)
            print(f"✅ {self.experiment_name} 학습 완료!")
            print("=" * 80)
            print(f"\n📁 모델 저장 위치: {self.config['general']['output_dir']}")
            print(f"📊 Best checkpoint를 사용하여 추론을 진행하세요")
            print("=" * 80)

            return trainer

        except Exception as e:
            print(f"\n❌ 학습 중 오류 발생: {e}")
            raise
