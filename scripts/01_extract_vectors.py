"""Extract happy + sad steering vectors against a neutral baseline.

For each candidate layer L in [10..27]:
  - Capture the residual stream at the LAST TOKEN of every statement (with the
    Llama 3.1 chat template applied as a user message + generation header)
  - Group by emotion, compute mean per group
  - happy_vector_L = mean(happy_acts) - mean(neutral_acts)
  - sad_vector_L   = mean(sad_acts)   - mean(neutral_acts)
  - L2-normalize each
  - Save to vectors/{emotion}_L{layer:02d}.pt with metadata

Prints cosine similarity between happy and sad vectors at every layer:
  - cos ~ -1: bipolar axis (single direction, like Turner-style)
  - |cos| < 0.5: genuinely independent emotion concepts (paper-style)
  - cos ~ +1: bug or dataset issue
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from steering.io import save_vector  # noqa: E402


CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_DATASET = REPO_ROOT / "prompts/emotion_examples.json"

EMOTIONS = ("happy", "sad", "neutral")
TARGET_EMOTIONS = ("happy", "sad")  # emotions for which we extract a vector


def parse_layer_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def load_model_and_tokenizer(config: dict):
    cache_dir = Path(config["model"]["cache_dir"])
    model_name = config["model"]["name"]
    local_path = cache_dir / model_name.replace("/", "__")
    if not local_path.exists():
        raise FileNotFoundError(
            f"Model not found at {local_path}. Run download_model.py first."
        )
    dtype = getattr(torch, config["model"]["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=dtype,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def hash_dataset(topics: list[dict]) -> str:
    canonical = json.dumps(topics, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chat_format(tokenizer, statement: str) -> torch.Tensor:
    """Wrap a statement as a Llama 3.1 chat user message and return the input
    ids tensor (with add_generation_prompt=True so the last token is the
    assistant header token — the position the model would begin generating from).
    """
    messages = [{"role": "user", "content": statement}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")


def extract_vectors(
    model,
    tokenizer,
    topics: list[dict],
    layers: list[int],
    hidden_size: int,
) -> dict[tuple[str, int], dict]:
    """Returns a dict keyed by (emotion, layer) with:
        { "sum": Tensor[hidden], "count": int, "raw_norm_sum": float }
    """
    aggregates: dict[tuple[str, int], dict] = {
        (emo, L): {
            "sum": torch.zeros(hidden_size, dtype=torch.float32, device="cuda"),
            "count": 0,
        }
        for emo in EMOTIONS
        for L in layers
    }

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook

    handles = [
        model.model.layers[L].register_forward_hook(make_hook(L)) for L in layers
    ]

    try:
        n_total = sum(len(t[emo]) for t in topics for emo in EMOTIONS)
        seen = 0
        t0 = time.time()
        with torch.no_grad():
            for topic in topics:
                for emo in EMOTIONS:
                    for statement in topic[emo]:
                        input_ids = chat_format(tokenizer, statement)
                        captured.clear()
                        model(input_ids)
                        for L in layers:
                            last = captured[L][0, -1, :].float()  # [hidden]
                            aggregates[(emo, L)]["sum"] += last
                            aggregates[(emo, L)]["count"] += 1
                        seen += 1
                        if seen % 60 == 0 or seen == n_total:
                            rate = seen / (time.time() - t0)
                            eta = (n_total - seen) / rate
                            print(f"  {seen}/{n_total}  ({rate:.1f}/s, eta {eta:.0f}s)")
    finally:
        for h in handles:
            h.remove()

    return aggregates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to emotion_examples.json",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="10-27",
        help="Layer range (e.g. '10-27'). Default: 10-27.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for vectors. Defaults to config.paths.vectors_dir.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    out_dir = args.out or Path(config["paths"]["vectors_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layer_range(args.layers)
    dataset = json.loads(args.examples.read_text())
    topics = dataset["topics"]
    dataset_hash = hash_dataset(topics)
    print(f"Dataset:  {args.examples}  ({len(topics)} topics)")
    print(f"Hash:     {dataset_hash}")
    print(f"Layers:   {layers[0]}..{layers[-1]} ({len(layers)} layers)")
    print(f"Output:   {out_dir}")

    print(f"\nLoading model {config['model']['name']}...")
    model, tokenizer = load_model_and_tokenizer(config)
    hidden_size = model.config.hidden_size
    print(f"  hidden_size={hidden_size}, n_layers={len(model.model.layers)}")

    n_total = sum(len(t[emo]) for t in topics for emo in EMOTIONS)
    print(f"\nExtracting last-token activations across {n_total} statements...")
    aggregates = extract_vectors(model, tokenizer, topics, layers, hidden_size)

    n_per_emotion = {emo: aggregates[(emo, layers[0])]["count"] for emo in EMOTIONS}
    print(f"\nStatements per emotion: {n_per_emotion}")

    print("\nComputing vectors and cosine similarities:")
    print(f"  {'layer':>5}  {'|happy|':>10}  {'|sad|':>10}  {'cos(h,s)':>10}")
    for L in layers:
        means = {
            emo: aggregates[(emo, L)]["sum"] / aggregates[(emo, L)]["count"]
            for emo in EMOTIONS
        }

        for emo in TARGET_EMOTIONS:
            raw = means[emo] - means["neutral"]
            raw_norm = raw.norm().item()
            normalized = (raw / raw_norm).to(torch.float32).cpu()

            metadata = {
                "model": config["model"]["name"],
                "dataset_hash": dataset_hash,
                "pooling": "last_token",
                "chat_template_applied": True,
                "n_positive": n_per_emotion[emo],
                "n_neutral": n_per_emotion["neutral"],
                "raw_norm_before_l2": float(raw_norm),
                "dtype": str(normalized.dtype),
            }
            save_vector(out_dir, emo, L, normalized, metadata)

        # Cosine similarity between happy and sad at this layer
        happy_raw = means["happy"] - means["neutral"]
        sad_raw = means["sad"] - means["neutral"]
        cos = torch.nn.functional.cosine_similarity(
            happy_raw.unsqueeze(0), sad_raw.unsqueeze(0)
        ).item()
        print(f"  {L:>5}  {happy_raw.norm().item():>10.3f}  {sad_raw.norm().item():>10.3f}  {cos:>+10.3f}")

    print(f"\nVectors written to {out_dir}/")
    print("Cosine interpretation:")
    print("  cos ~ -1.0  -> happy and sad are essentially the same axis (bipolar)")
    print("  |cos| < 0.5 -> genuinely independent emotion concepts (paper-style)")
    print("  cos > 0     -> dataset issue: happy and sad point the same way (investigate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
