#!/usr/bin/env python3
"""Generate Solar pseudo-label summaries for a dataset split."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from preprocess import load_and_preprocess

SOLAR_BASE_URL = "https://api.upstage.ai/v1/solar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pseudo-label summaries with Solar Chat API."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train/dev/test CSV files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=("train", "dev", "test"),
        help="Dataset split to summarize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dev_solar_pseudo_labels.csv"),
        help="Where to write the pseudo-label CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="solar-1-mini-chat",
        help="Solar model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature passed to the chat completion API.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.3,
        help="Nucleus sampling parameter for the API.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of rows (0 means use the full split).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, skip rows already summarized.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=100,
        help="Requests per window before pausing (0 disables throttling).",
    )
    parser.add_argument(
        "--rate-window",
        type=int,
        default=60,
        help="Window size in seconds that --rate-limit applies to.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Extra seconds to wait after hitting the rate limit.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per request when the API fails.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=3.0,
        help="Base seconds to wait before retrying (exponential backoff).",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=(
            "You are an expert dialogue summarizer. Produce a concise, factual "
            "summary in Korean, preserving speaker tags like Person1, Person2."
        ),
        help="System prompt instruction for the model.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Solar API key. If omitted, reads UPSTAGE_API_KEY from the environment.",
    )
    return parser.parse_args()


def create_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=SOLAR_BASE_URL)


def build_messages(dialogue: str, instruction: str) -> List[Dict[str, str]]:
    user_prompt = (
        "다음 대화를 요약하세요.\n\n"
        f"Dialogue:\n{dialogue.strip()}\n\n"
        "Summary:"
    )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt},
    ]


def maybe_wait_for_rate_limit(
    request_count: int,
    window_start: float,
    limit: int,
    window_seconds: int,
    cooldown: float,
) -> float:
    if limit <= 0:
        return window_start
    if request_count % limit != 0:
        return window_start
    elapsed = time.monotonic() - window_start
    if elapsed < window_seconds:
        sleep_for = window_seconds - elapsed + cooldown
        time.sleep(max(sleep_for, 0.0))
    return time.monotonic()


def generate_summary(
    client: OpenAI,
    dialogue: str,
    instruction: str,
    model: str,
    temperature: float,
    top_p: float,
    max_retries: int,
    retry_wait: float,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(dialogue, instruction),
                temperature=temperature,
                top_p=top_p,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:  # pylint: disable=broad-except
            if attempt + 1 == max_retries:
                raise RuntimeError(f"Failed to summarize dialogue: {exc}") from exc
            wait_time = retry_wait * (2 ** attempt)
            time.sleep(wait_time)
    return ""


def load_existing(output_path: Path) -> Dict[str, str]:
    if not output_path.exists():
        return {}
    existing = pd.read_csv(output_path)
    return dict(zip(existing["fname"], existing["summary_solar"]))


def main() -> None:
    args = parse_args()
    api_key = "up_yAA2b1eZJqfYC4Bqqle0zg4Ce3Ske"
    
    
    datasets = load_and_preprocess(args.data_dir)
    if args.split not in datasets:
        sys.exit(f"Split '{args.split}' not found under {args.data_dir}.")
    df = datasets[args.split]
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    existing = load_existing(args.output) if args.resume else {}
    client = create_client(api_key)
    rows: List[Dict[str, Optional[str]]] = []
    processed = 0
    request_count = 0
    window_start = time.monotonic()
    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Summarizing")
    for row in iterator:
        fname = getattr(row, "fname", f"{args.split}_{processed}")
        if fname in existing:
            rows.append(
                {
                    "fname": fname,
                    "dialogue": row.dialogue,
                    "summary_solar": existing[fname],
                    "summary_gold": getattr(row, "summary", None),
                }
            )
            processed += 1
            continue
        window_start = maybe_wait_for_rate_limit(
            request_count=request_count,
            window_start=window_start,
            limit=args.rate_limit,
            window_seconds=args.rate_window,
            cooldown=args.cooldown,
        )
        summary = generate_summary(
            client=client,
            dialogue=row.dialogue,
            instruction=args.instruction,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )
        request_count += 1
        rows.append(
            {
                "fname": fname,
                "dialogue": row.dialogue,
                "summary_solar": summary,
                "summary_gold": getattr(row, "summary", None),
            }
        )
        processed += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} pseudo-labels to {args.output}")


if __name__ == "__main__":
    main()
