"""Optional 2-layer pair sub-sweep for one emotion.

After `02_layer_sweep.py` surfaces a shortlist of productive layers for an
emotion (see NOTES.md §8), this script sweeps all unordered pairs of those
layers with an independent coefficient at each, so you can see whether
steering at two layers simultaneously produces a cleaner / stronger on-target
shift than any single layer alone.

Steering composition is done exactly the way `run.py` does it: one
`steer(...)` context per (layer, coefficient) entry, stacked in an
`ExitStack`. The `MultiVectorSteeringHook` hooks add their deltas
independently during each forward pass, so pairing layers has no special-
cased code path — the same machinery that ships to production.

Usage:
    python scripts/02b_pair_sweep.py --emotion happy --candidate-layers 17,19,20
    python scripts/02b_pair_sweep.py --emotion sad   --candidate-layers 17,19,20 \\
        --coefs 0,0.25,0.5 --held-out prompts/held_out.json

Output:
    outputs/pair_sweep_{emotion}.csv  with columns:
        layer_a, layer_b, coef_a, coef_b, prompt_id, prompt, output, valence_score
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from eval.sentiment import score_valence_batch  # noqa: E402
from steering.hooks import steer  # noqa: E402
from steering.io import load_emotion_vector  # noqa: E402


CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_HELD_OUT = REPO_ROOT / "prompts/held_out.json"

DEFAULT_COEFFICIENTS = [0.0, 0.25, 0.5]


def parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_coefs(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def load_model_and_tokenizer(config: dict):
    cache_dir = Path(config["model"]["cache_dir"])
    model_name = config["model"]["name"]
    local_path = cache_dir / model_name.replace("/", "__")
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


def chat_format(tokenizer, prompt: str, system_prompt: str) -> torch.Tensor:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")


def generate_one(model, tokenizer, input_ids, gen_cfg: dict, seed: int) -> str:
    torch.manual_seed(seed)
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=True,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion", choices=["happy", "sad"], required=True)
    parser.add_argument(
        "--candidate-layers",
        type=str,
        required=True,
        help="Comma-separated shortlist of layer indices, e.g. '17,19,20'. "
             "All unordered pairs among these will be swept.",
    )
    parser.add_argument(
        "--coefs",
        type=str,
        default=",".join(str(c) for c in DEFAULT_COEFFICIENTS),
        help="Comma-separated non-negative fraction-of-norm coefficients, "
             "applied independently at each of the two layers (2D grid).",
    )
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--vectors-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-score", action="store_true", help="Skip RoBERTa scoring (faster).")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    candidates = parse_int_list(args.candidate_layers)
    if len(candidates) < 2:
        raise SystemExit(
            f"--candidate-layers needs at least 2 layers, got {candidates}"
        )
    if len(set(candidates)) != len(candidates):
        raise SystemExit(f"--candidate-layers has duplicates: {candidates}")
    pairs = list(itertools.combinations(sorted(candidates), 2))
    coefs = parse_coefs(args.coefs)
    if any(c < 0 for c in coefs):
        raise SystemExit("Pair sweep uses one-sided non-negative coefficients only.")

    vectors_dir = args.vectors_dir or Path(config["paths"]["vectors_dir"])
    out_path = args.out or Path(config["paths"]["outputs_dir"]) / f"pair_sweep_{args.emotion}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layer_norms = config["steering"].get("layer_norms")
    if not layer_norms:
        raise RuntimeError(
            "config.steering.layer_norms is empty. Run `python -m steering.norm` first."
        )
    layer_norms = {int(k): float(v) for k, v in layer_norms.items()}
    for L in candidates:
        if L not in layer_norms:
            raise RuntimeError(f"layer {L} has no entry in config.steering.layer_norms")

    held = json.loads(args.held_out.read_text())["prompts"]

    print(f"Emotion:           {args.emotion}")
    print(f"Candidate layers:  {candidates}")
    print(f"Pairs to sweep:    {pairs}  ({len(pairs)} pairs)")
    print(f"Coefficients:      {coefs}  (2D grid: {len(coefs)}x{len(coefs)} = {len(coefs)**2})")
    print(f"Prompts:           {len(held)} held-out")
    print(f"Output:            {out_path}")

    n_total = len(pairs) * len(coefs) ** 2 * len(held)
    print(f"\nEstimated generations: {n_total}")

    print(f"\nLoading model {config['model']['name']}...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Pre-load all vectors for candidate layers once.
    vectors: dict[int, torch.Tensor] = {}
    for L in candidates:
        try:
            v, _ = load_emotion_vector(vectors_dir, args.emotion, L)
        except FileNotFoundError:
            raise SystemExit(
                f"Missing vectors/{args.emotion}_L{L:02d}.pt under {vectors_dir}. "
                f"Run 01_extract_vectors.py so the sweep has vectors for every candidate."
            )
        vectors[L] = v.to("cuda", dtype=torch.bfloat16)

    gen_cfg = config["generation"]
    seed = int(gen_cfg["seed"])
    sys_prompt = config.get("system_prompt") or ""

    rows: list[dict] = []
    t0 = time.time()
    done = 0

    for (La, Lb) in pairs:
        vec_a = vectors[La]
        vec_b = vectors[Lb]
        norm_a = layer_norms[La]
        norm_b = layer_norms[Lb]

        for coef_a in coefs:
            for coef_b in coefs:
                for p in held:
                    input_ids = chat_format(tokenizer, p["prompt"], sys_prompt)
                    with ExitStack() as stack:
                        stack.enter_context(
                            steer(model, La, [(vec_a, coef_a)], norm_a)
                        )
                        stack.enter_context(
                            steer(model, Lb, [(vec_b, coef_b)], norm_b)
                        )
                        text = generate_one(model, tokenizer, input_ids, gen_cfg, seed)
                    rows.append({
                        "layer_a": La,
                        "layer_b": Lb,
                        "coef_a": coef_a,
                        "coef_b": coef_b,
                        "prompt_id": p["id"],
                        "prompt": p["prompt"],
                        "output": text.strip().replace("\n", " "),
                        "valence_score": "",
                    })
                    done += 1
                    if done % 10 == 0 or done == n_total:
                        rate = done / (time.time() - t0)
                        eta = (n_total - done) / rate
                        print(f"  {done}/{n_total}  ({rate:.2f}/s, eta {eta:.0f}s)")

    if not args.no_score and rows:
        print(f"\nScoring {len(rows)} outputs with RoBERTa sentiment classifier...")
        scores = score_valence_batch([r["output"] for r in rows])
        for r, s in zip(rows, scores):
            r["valence_score"] = f"{s:+.3f}"

    fieldnames = ["layer_a", "layer_b", "coef_a", "coef_b", "prompt_id", "prompt", "output", "valence_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")

    if not args.no_score:
        print("\nMean valence_score per pair, per (coef_a, coef_b) cell:")
        max_coef = max(coefs) if coefs else 0.0
        min_coef = min(coefs) if coefs else 0.0
        shortlist: list[tuple[tuple[int, int], float]] = []

        for (La, Lb) in pairs:
            print(f"\n  pair L{La:02d} + L{Lb:02d}:")
            header_cells = [f"{'c_b=' + f'{c:g}':>10}" for c in coefs]
            print(f"    {'c_a\\c_b':>10}  " + "  ".join(header_cells))
            cell_means: dict[tuple[float, float], float] = {}
            for ca in coefs:
                cells = []
                for cb in coefs:
                    vals = [
                        float(r["valence_score"]) for r in rows
                        if r["layer_a"] == La and r["layer_b"] == Lb
                        and r["coef_a"] == ca and r["coef_b"] == cb
                        and r["valence_score"]
                    ]
                    if vals:
                        mean = sum(vals) / len(vals)
                        cell_means[(ca, cb)] = mean
                        cells.append(f"{mean:+10.3f}")
                    else:
                        cells.append("       n/a")
                print(f"    {'c_a=' + f'{ca:g}':>10}  " + "  ".join(cells))

            v_max = cell_means.get((max_coef, max_coef))
            v_zero = cell_means.get((min_coef, min_coef))
            if v_max is not None and v_zero is not None:
                delta = abs(v_max - v_zero)
                shortlist.append(((La, Lb), delta))
                print(f"    |\u0394valence|  (c={max_coef:g},{max_coef:g}) vs (c={min_coef:g},{min_coef:g}): {delta:+.3f}")

        if shortlist:
            shortlist.sort(key=lambda x: x[1], reverse=True)
            print("\nPair ranking by |\u0394valence| between max and zero cells:")
            for (pair, delta) in shortlist:
                print(f"  L{pair[0]:02d} + L{pair[1]:02d}   |\u0394|={delta:+.3f}")

    print(f"\nNext: eyeball {out_path} on a promising pair, then calibrate with")
    print(f"  python scripts/03_calibrate_coefficients.py --emotion {args.emotion} --layer <La> <Lb>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
