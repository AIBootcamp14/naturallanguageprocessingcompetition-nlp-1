#!/usr/bin/env python3
"""Fine-tune KoBART with Solar pseudo-label augmentation."""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from rouge import Rouge
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from generation import build_generation_kwargs
from preprocess import (
    _clean_text_block,  # type: ignore[attr-defined]
    _normalize_speaker_mentions,  # type: ignore[attr-defined]
    load_and_preprocess,
)


@dataclass
class ConfigBundle:
    config_path: Path
    data_dir: Path
    pseudo_path: Path
    output_dir: Path
    pseudo_ratio: float
    seed: int
    resume_from_checkpoint: Optional[str]


class DialogueSummaryDataset(Dataset):
    """PyTorch dataset for dialogue summarization."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        sample = self.dataframe.iloc[idx]
        dialogue = sample["dialogue"]
        summary = sample["summary"]

        model_inputs = self.tokenizer(
            dialogue,
            max_length=self.max_source_length,
            truncation=True,
            return_token_type_ids=False,
        )
        labels = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            truncation=True,
        )["input_ids"]

        # BART does not use token_type_ids; drop if tokenizer still returned it.
        model_inputs.pop("token_type_ids", None)
        model_inputs["labels"] = labels
        return model_inputs


def parse_args() -> ConfigBundle:
    parser = argparse.ArgumentParser(description="Train with Solar pseudo-labels.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--pseudo-path",
        type=Path,
        default=Path("data/dev_solar_pseudo_labels.csv"),
        help="CSV with columns [fname, dialogue, summary_solar].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./pseudo_augmented"),
        help="Where to store checkpoints and tokenizer.",
    )
    parser.add_argument(
        "--pseudo-ratio",
        type=float,
        default=0.5,
        help="Desired pseudo-label sample count relative to gold train size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint directory to resume from.",
    )
    args = parser.parse_args()
    return ConfigBundle(
        config_path=args.config,
        data_dir=args.data_dir,
        pseudo_path=args.pseudo_path,
        output_dir=args.output_dir,
        pseudo_ratio=args.pseudo_ratio,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict[str, Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix == ".json":
            return json.load(fh)
        return yaml.safe_load(fh)


def prepare_pseudo_dataframe(pseudo_path: Path) -> pd.DataFrame:
    if not pseudo_path.exists():
        raise FileNotFoundError(f"Pseudo-label file not found: {pseudo_path}")
    pseudo_df = pd.read_csv(pseudo_path)
    if "summary_solar" not in pseudo_df.columns:
        raise ValueError("Pseudo-label CSV must include a 'summary_solar' column.")

    pseudo_df = pseudo_df[["fname", "dialogue", "summary_solar"]].copy()
    pseudo_df.rename(columns={"summary_solar": "summary"}, inplace=True)

    pseudo_df["dialogue"] = pseudo_df["dialogue"].map(_clean_text_block).map(
        _normalize_speaker_mentions
    )
    pseudo_df["summary"] = pseudo_df["summary"].map(_clean_text_block).map(
        _normalize_speaker_mentions
    )
    pseudo_df["source"] = "pseudo"
    return pseudo_df


def balance_pseudo_samples(
    gold_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    ratio: float,
    seed: int,
) -> pd.DataFrame:
    if ratio <= 0 or pseudo_df.empty:
        return pseudo_df.iloc[0:0].copy()

    desired = int(len(gold_df) * ratio)
    if desired <= 0:
        return pseudo_df.iloc[0:0].copy()

    if len(pseudo_df) >= desired:
        return pseudo_df.sample(n=desired, random_state=seed)

    frames = []
    reps, remainder = divmod(desired, len(pseudo_df))
    if reps > 0:
        frames.extend([pseudo_df] * reps)
    if remainder > 0:
        frames.append(pseudo_df.sample(n=remainder, random_state=seed, replace=True))
    return pd.concat(frames, ignore_index=True)


def build_datasets(
    config: Dict[str, Dict[str, object]],
    data_dir: Path,
    pseudo_path: Path,
    pseudo_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = load_and_preprocess(data_dir)
    train_df = datasets["train"].copy()
    train_df["summary"] = train_df["summary"].astype(str)
    train_df["source"] = "gold"
    dev_df = datasets["dev"]

    pseudo_df = prepare_pseudo_dataframe(pseudo_path)
    pseudo_balanced = balance_pseudo_samples(train_df, pseudo_df, pseudo_ratio, seed)

    combined_train = pd.concat([train_df, pseudo_balanced], ignore_index=True)
    combined_train = combined_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined_train, dev_df, pseudo_balanced


def postprocess_text(preds: List[str], labels: List[str]) -> tuple[List[str], List[str]]:
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def main() -> None:
    bundle = parse_args()
    config = load_config(bundle.config_path)
    training_cfg = config.get("training", {})
    tokenizer_cfg = config.get("tokenizer", {})
    general_cfg = config.get("general", {})

    set_seed(bundle.seed)

    model_name = general_cfg.get("model_name", "digit82/kobart-summarization")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = tokenizer_cfg.get("special_tokens")
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens)})

    encoder_max_len = int(tokenizer_cfg.get("encoder_max_len", 512))
    decoder_max_len = int(tokenizer_cfg.get("decoder_max_len", 100))

    combined_train_df, dev_df, _ = build_datasets(
        config, bundle.data_dir, bundle.pseudo_path, bundle.pseudo_ratio, bundle.seed
    )

    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = DialogueSummaryDataset(
        combined_train_df, tokenizer, encoder_max_len, decoder_max_len
    )
    dev_dataset = DialogueSummaryDataset(dev_df, tokenizer, encoder_max_len, decoder_max_len)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    rouge = Rouge()
    generation_kwargs = build_generation_kwargs(tokenizer)

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"]
            if isinstance(scores, dict)
            else scores[0]["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"]
            if isinstance(scores, dict)
            else scores[0]["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
            if isinstance(scores, dict)
            else scores[0]["rouge-l"]["f"],
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(bundle.output_dir),
        overwrite_output_dir=bool(training_cfg.get("overwrite_output_dir", True)),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 5)),
        learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 8)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        optim=training_cfg.get("optim", "adamw_torch"),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        save_total_limit=int(training_cfg.get("save_total_limit", 3)),
        fp16=bool(training_cfg.get("fp16", False)),
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", True)),
        seed=bundle.seed,
        logging_dir=str(Path(training_cfg.get("logging_dir", "./logs"))),
        logging_strategy=training_cfg.get("logging_strategy", "steps"),
        predict_with_generate=bool(training_cfg.get("predict_with_generate", True)),
        generation_max_length=int(training_cfg.get("generation_max_length", decoder_max_len)),
        report_to=training_cfg.get("report_to", None),
    )

    callbacks: List[EarlyStoppingCallback] = []
    if training_cfg.get("early_stopping_patience") is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(training_cfg["early_stopping_patience"]),
                early_stopping_threshold=float(
                    training_cfg.get("early_stopping_threshold", 0.0)
                ),
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=bundle.resume_from_checkpoint)

    bundle.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(bundle.output_dir))
    tokenizer.save_pretrained(str(bundle.output_dir))

    metrics = trainer.evaluate(**generation_kwargs)
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(dev_dataset)
    with open(bundle.output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    pseudo_used_count = int(combined_train_df["source"].eq("pseudo").sum())
    with open(bundle.output_dir / "pseudo_stats.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "pseudo_ratio": bundle.pseudo_ratio,
                "pseudo_used": pseudo_used_count,
                "train_total": len(combined_train_df),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
