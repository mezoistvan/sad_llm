"""Compute the mean L2 norm of the residual stream at each candidate layer.

Steering coefficients in this project are in units of fraction-of-residual-norm
(following Sofroniew et al. 2026). To express `s = 0.5 * residual_norm`, we need
to know the residual_norm at each candidate layer first. This script computes
those norms once, persists them into config.yaml, and is then read by all
downstream steering scripts.

Run after 00_smoke_test.py succeeds, before 02_layer_sweep.py:

    python -m steering.norm

Idempotent: re-running just refreshes the values.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

# A small generic corpus: a few hundred tokens of varied, emotionally neutral
# text on completely unrelated topics. Good enough for a stable mean-norm
# estimate without pulling in external datasets. We deliberately avoid any
# emotional content so the norm reflects the model's "default" state.
NEUTRAL_CORPUS = [
    "The capital of France is Paris, located in the north-central part of the country along the Seine river. The city has a population of around two million in its core area and is known for its architecture, museums, and cuisine.",
    "To make a roux, melt equal parts butter and flour in a pan over medium heat. Stir continuously for about three minutes for a blonde roux, or longer for a darker color. The mixture forms the base for many sauces.",
    "A bicycle drivetrain consists of the chainrings, chain, cassette, and derailleurs. To shift smoothly, ease pressure on the pedals briefly while moving the shifter. Regular cleaning and lubrication of the chain extends its useful life considerably.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose. It takes place primarily in the chloroplasts of leaf cells and produces oxygen as a byproduct released into the atmosphere.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides. It is one of the foundational results in classical geometry and has many practical applications.",
    "To set up a basic vegetable garden, choose a location with at least six hours of daily sun. Test the soil pH and amend it with compost as needed. Plant seedlings according to the spacing recommended on each variety's label.",
    "HTTP is a stateless protocol used for transferring hypertext documents over the internet. A typical request includes a method, a URL, headers, and an optional body. Common methods include GET, POST, PUT, and DELETE among others.",
    "The Roman aqueducts used a slight downward gradient to move water across long distances by gravity alone. Some segments ran underground, others on raised stone arches. Several survive in working condition to this day across southern Europe.",
    "When tuning a guitar to standard tuning, the strings from low to high are tuned to E, A, D, G, B, and E. Use a tuner or tuning fork as a reference for the low E string, then tune each higher string to the one below it.",
    "A sourdough starter is a fermented mixture of flour and water that captures wild yeast. Maintain it by feeding it equal parts flour and water at regular intervals. A healthy starter doubles in volume within four to eight hours of feeding.",
    "The boiling point of water at sea level is 100 degrees Celsius, but it decreases at higher elevations due to lower atmospheric pressure. At 2000 meters, water boils at about 93 degrees, which affects cooking times for many recipes.",
    "Public transit in dense urban areas typically combines buses, trams, subways, and commuter rail. Frequency, coverage, and reliability are the three main factors that determine whether residents will choose transit over driving in practice.",
]


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


def compute_layer_norms(
    model,
    tokenizer,
    layers: list[int],
    corpus_texts: list[str],
) -> dict[int, float]:
    """Mean L2 norm of the residual stream at each requested layer, averaged
    over all token positions across all corpus texts.

    The residual stream at layer L is defined as the output of
    `model.model.layers[L]`, which has shape [batch, seq, hidden]. We L2-norm
    along the hidden dim, giving per-position norms of shape [batch, seq], then
    average across all positions and all texts.
    """
    sums: dict[int, float] = {layer: 0.0 for layer in layers}
    counts: dict[int, int] = {layer: 0 for layer in layers}

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook

    handles = []
    for layer in layers:
        h = model.model.layers[layer].register_forward_hook(make_hook(layer))
        handles.append(h)

    try:
        for text in corpus_texts:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            captured.clear()
            with torch.no_grad():
                model(input_ids)
            for layer in layers:
                hidden = captured[layer]
                norms = hidden.float().norm(dim=-1)
                sums[layer] += norms.sum().item()
                counts[layer] += norms.numel()
    finally:
        for h in handles:
            h.remove()

    return {layer: sums[layer] / counts[layer] for layer in layers}


def write_norms_to_config(config_path: Path, norms: dict[int, float]) -> None:
    config = yaml.safe_load(config_path.read_text())
    if "steering" not in config or config["steering"] is None:
        config["steering"] = {}
    config["steering"]["layer_norms"] = {int(k): float(v) for k, v in norms.items()}
    config["steering"]["coefficient_units"] = "fraction_of_residual_norm"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))


def parse_layer_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        type=str,
        default="10-27",
        help="Layer range (e.g. '10-27') or comma-separated list. Default: 10-27.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to config.yaml. Norms are written back here.",
    )
    args = parser.parse_args()

    layers = parse_layer_range(args.layers)
    config = yaml.safe_load(args.config.read_text())

    print(f"Loading model from {config['model']['name']}...")
    model, tokenizer = load_model_and_tokenizer(config)
    n_model_layers = len(model.model.layers)
    print(f"Model has {n_model_layers} layers; computing norms at layers {layers}")

    out_of_range = [L for L in layers if L >= n_model_layers]
    if out_of_range:
        raise ValueError(
            f"Requested layers {out_of_range} are out of range for a "
            f"{n_model_layers}-layer model."
        )

    n_tokens = sum(len(tokenizer(t).input_ids) for t in NEUTRAL_CORPUS)
    print(f"Corpus: {len(NEUTRAL_CORPUS)} texts, {n_tokens} tokens total")

    t0 = time.time()
    norms = compute_layer_norms(model, tokenizer, layers, NEUTRAL_CORPUS)
    elapsed = time.time() - t0

    print(f"\nLayer norms (mean L2 of residual stream, computed in {elapsed:.1f}s):")
    for layer in sorted(norms):
        print(f"  layer {layer:2d}:  {norms[layer]:8.2f}")

    write_norms_to_config(args.config, norms)
    print(f"\nWritten to {args.config} under steering.layer_norms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
