"""Lightweight sentiment classifier for valence scoring.

Uses Cardiff NLP's RoBERTa sentiment model (3-class: negative / neutral /
positive). We project the 3 class probabilities to a single signed score in
[-1, +1] via `p(positive) - p(negative)` so it composes cleanly with the
weather-niceness signal in inputs/mapping.py.

The model is ~500MB and lazy-loaded on first call. Runs on CPU by default;
if CUDA is available it'll use it, but the model is small enough that CPU
is fine for the per-row scoring volumes in 02_layer_sweep.py / 03_calibrate.

Usage:
    from eval.sentiment import score_valence, score_csv

    score_valence("What a wonderful morning")  # -> ~0.85
    score_csv("outputs/sweep.csv")             # adds 'valence_score' column
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Cardiff RoBERTa label order: 0=negative, 1=neutral, 2=positive.
NEGATIVE_IDX = 0
POSITIVE_IDX = 2


@lru_cache(maxsize=1)
def _load() -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


def score_valence(text: str) -> float:
    """Return a signed valence score in [-1, +1].

    +1.0 = strongly positive, 0 = neutral, -1.0 = strongly negative.
    Projects 3-class softmax to `p(positive) - p(negative)`.
    """
    return score_valence_batch([text])[0]


def score_valence_batch(texts: list[str], batch_size: int = 16) -> list[float]:
    """Vectorized valence scoring. Truncates inputs to model max length (512)."""
    if not texts:
        return []

    tokenizer, model, device = _load()
    out: list[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = logits.softmax(dim=-1)
            scores = (probs[:, POSITIVE_IDX] - probs[:, NEGATIVE_IDX]).tolist()
            out.extend(scores)

    return out


def score_csv(
    path: str | Path,
    text_col: str = "output",
    score_col: str = "valence_score",
    overwrite: bool = True,
) -> Path:
    """Read a CSV, score the `text_col` column, write `score_col` back.

    Returns the path written. Pandas is imported lazily so importing this
    module doesn't pull pandas in for callers that only want score_valence.
    """
    import pandas as pd

    path = Path(path)
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise KeyError(f"Column {text_col!r} not in {path}; columns: {list(df.columns)}")

    df[score_col] = score_valence_batch(df[text_col].astype(str).tolist())

    out_path = path if overwrite else path.with_suffix(".scored.csv")
    df.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text", help="Score a single string passed on the CLI.")
    p_text.add_argument("text", type=str)

    p_csv = sub.add_parser("csv", help="Score a CSV file in-place.")
    p_csv.add_argument("path", type=Path)
    p_csv.add_argument("--text-col", default="output")
    p_csv.add_argument("--score-col", default="valence_score")
    p_csv.add_argument("--no-overwrite", action="store_true")

    p_demo = sub.add_parser("demo", help="Score a few canned examples to sanity-check.")

    args = parser.parse_args()

    if args.cmd == "text":
        score = score_valence(args.text)
        print(f"{score:+.3f}\t{args.text}")
    elif args.cmd == "csv":
        out = score_csv(
            args.path,
            text_col=args.text_col,
            score_col=args.score_col,
            overwrite=not args.no_overwrite,
        )
        print(f"Scored: {out}")
    elif args.cmd == "demo":
        examples = [
            "What a wonderful morning, the sun is shining and I feel great!",
            "Today is an okay day. Nothing much happened.",
            "Everything is terrible and I just want to crawl back into bed.",
            "The meeting starts at three pm in the conference room.",
            "I cannot stop smiling about how good this all feels right now.",
            "The weight of it all is just too much today, I am completely empty.",
        ]
        scores = score_valence_batch(examples)
        for s, t in zip(scores, examples):
            print(f"{s:+.3f}\t{t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
