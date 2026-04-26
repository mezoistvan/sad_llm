"""Idempotent download of Llama 3.1 8B Instruct to the persistent volume.

Run once per network volume. Skips if the model is already present.
Requires `huggingface-cli login` first (Llama 3.1 is gated).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Files we need: weights (safetensors), tokenizer, configs. Skip the
# original/ subfolder (PyTorch consolidated weights, ~16GB duplicate).
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "tokenizer.model",
    "tokenizer.*",
    "special_tokens_map.json",
]
IGNORE_PATTERNS = [
    "original/*",
    "*.bin",  # we use safetensors
    "*.pth",
    "*.gguf",
]


def model_already_downloaded(target_dir: Path) -> bool:
    if not target_dir.exists():
        return False
    has_config = (target_dir / "config.json").exists()
    has_weights = any(target_dir.glob("*.safetensors"))
    has_tokenizer = (target_dir / "tokenizer.json").exists() or (
        target_dir / "tokenizer.model"
    ).exists()
    return has_config and has_weights and has_tokenizer


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    model_name = config["model"]["name"]
    cache_dir = Path(config["model"]["cache_dir"])

    target_dir = cache_dir / model_name.replace("/", "__")

    if model_already_downloaded(target_dir):
        print(f"Model already present at {target_dir}. Skipping download.")
        print(
            "  (Delete the directory and re-run if you want to force a fresh pull.)"
        )
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_name} -> {target_dir}")
    print("This is ~16GB and typically takes 10-20 min on a pod's bandwidth.")
    t0 = time.time()

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(target_dir),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            max_workers=8,
        )
    except GatedRepoError:
        print(
            "\nERROR: The repo is gated. Run `huggingface-cli login` with a token "
            "that has been granted access to meta-llama/Llama-3.1-8B-Instruct.",
            file=sys.stderr,
        )
        return 1
    except RepositoryNotFoundError:
        print(
            f"\nERROR: Repo {model_name} not found. Check config.yaml model.name.",
            file=sys.stderr,
        )
        return 1

    elapsed = time.time() - t0
    total_bytes = sum(p.stat().st_size for p in target_dir.rglob("*") if p.is_file())
    print(f"\nDone in {elapsed / 60:.1f} min. Local size: {total_bytes / 1e9:.1f} GB.")
    print(f"Model path: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
