"""Save and load steering vectors with metadata.

Each vector lives in its own .pt file containing a dict:

    {
        "vector": torch.Tensor of shape [hidden],   # L2-normalized
        "metadata": {
            "model": str,              # e.g. "meta-llama/Llama-3.1-8B-Instruct"
            "dataset_hash": str,       # SHA256 of canonicalized dataset JSON
            "emotion": str,            # "happy" or "sad"
            "layer": int,
            "pooling": str,            # "last_token"
            "chat_template_applied": bool,
            "n_positive": int,         # n statements in the emotion class
            "n_neutral": int,          # n statements in neutral baseline
            "raw_norm_before_l2": float,
            "dtype": str,              # e.g. "torch.bfloat16"
        }
    }

Filenames follow the convention:  vectors_dir / "{emotion}_L{layer:02d}.pt"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def vector_path(vectors_dir: Path | str, emotion: str, layer: int) -> Path:
    return Path(vectors_dir) / f"{emotion}_L{layer:02d}.pt"


def save_vector(
    vectors_dir: Path | str,
    emotion: str,
    layer: int,
    vector: torch.Tensor,
    metadata: dict[str, Any],
) -> Path:
    if vector.dim() != 1:
        raise ValueError(f"Expected 1-D vector, got shape {tuple(vector.shape)}.")

    vectors_dir = Path(vectors_dir)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    path = vector_path(vectors_dir, emotion, layer)

    payload = {
        "vector": vector.detach().cpu().contiguous(),
        "metadata": {**metadata, "emotion": emotion, "layer": int(layer)},
    }
    torch.save(payload, path)
    return path


def load_vector(path: Path | str) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["vector"], payload["metadata"]


def load_emotion_vector(
    vectors_dir: Path | str,
    emotion: str,
    layer: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    return load_vector(vector_path(vectors_dir, emotion, layer))
