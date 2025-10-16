"""Utilities for loading and preprocessing dialogue summarization datasets."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List

import pandas as pd

# Matches speaker prefixes such as "#Person1#:" or "Person2:".
_TURN_PATTERN = re.compile(r"^(#?Person\\d+#?):?\\s*(.*)$")
_SPEAKER_INLINE_PATTERN = re.compile(r"#Person(\\d+)#")


def _clean_text_block(text: str) -> str:
    """Normalize whitespace within a dialogue or summary string."""
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\\s+", " ", line.strip()) for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line)


def _normalize_speaker_mentions(text: str) -> str:
    """Convert speaker markers like ``#Person1#`` to ``Person1`` and add spacing."""
    if not text:
        return ""

    normalized_lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        tagged = re.sub(r"#Person(\\d+)#:", r"Person\\1:", stripped)
        tagged = _SPEAKER_INLINE_PATTERN.sub(lambda m: f"Person{m.group(1)}", tagged)
        tagged = re.sub(r"(Person\\d)(?=[^\s:])", r"\\1 ", tagged)
        tagged = re.sub(r"\\s+", " ", tagged).strip()
        tagged = tagged.replace(" :", ":")
        normalized_lines.append(tagged)

    return "\n".join(normalized_lines)


def _dialogue_to_turns(dialogue: str) -> List[Dict[str, Any]]:
    """Split a dialogue string into structured speaker turns."""
    turns: List[Dict[str, Any]] = []

    for raw_line in dialogue.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        match = _TURN_PATTERN.match(line)
        if match:
            speaker_tag, utterance = match.groups()
            speaker = speaker_tag.replace("#", "")  # e.g. "Person1"
            turns.append({"speaker": speaker, "utterance": utterance.strip()})
        elif turns:
            # Attach unlabelled continuation to the previous speaker.
            turns[-1]["utterance"] = f"{turns[-1]['utterance']} {line}".strip()
        else:
            turns.append({"speaker": None, "utterance": line})

    return turns


def load_and_preprocess(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load train/dev/test CSVs and apply text normalization suitable for modeling."""
    data_path = Path(data_dir)
    datasets: Dict[str, pd.DataFrame] = {}

    for split, has_labels in (("train", True), ("dev", True), ("test", False)):
        csv_path = data_path / f"{split}.csv"
        df = pd.read_csv(csv_path, keep_default_na=False)

        df["dialogue_raw"] = df["dialogue"]
        cleaned_dialogue = df["dialogue"].map(_clean_text_block)
        df["dialogue"] = cleaned_dialogue.map(_normalize_speaker_mentions)
        df["dialogue_turns"] = df["dialogue"].map(_dialogue_to_turns)

        if has_labels and "summary" in df.columns:
            df["summary_raw"] = df["summary"]
            cleaned_summary = df["summary"].map(_clean_text_block)
            df["summary"] = cleaned_summary.map(_normalize_speaker_mentions)

        if has_labels and "topic" in df.columns:
            df["topic"] = df["topic"].replace("", pd.NA).astype("category")

        datasets[split] = df

    return datasets


__all__ = [
    "load_and_preprocess",
]
