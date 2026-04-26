"""Calibrate the workable coefficient range for one emotion vector at the
chosen layer(s), plus a logp probe (paper-style validation).

Two modes, selected by the number of --layer arguments:

  Single-layer (back-compat):
      --layer 19
      Sweeps non-negative fraction-of-norm coefficients over held-out prompts
      at that single layer. CSV output: outputs/calibration_{emotion}_L{NN}.csv.

  Two-layer (pair calibration):
      --layer 19 20
      Sweeps an independent 2D grid of coefficients (coef_a at layer_a,
      coef_b at layer_b). CSV output: outputs/calibration_{emotion}_L{a}_L{b}.csv.
      Each layer ends up with its own calibrated max, matching the
      list-of-layers schema in config.yaml.

Logp probe (paper-style validation) is run after generation sweeps to confirm
each layer's steering moves the intended emotion-word axis. In 2-layer mode
the probe is a 2D grid over a small set of (coef_a, coef_b) values.

Pick per-layer max(es) from the sweep CSV (largest noticeable, on-target
effect, no incoherence) and write them back to config.yaml under
`steering.{emotion}.layers` as a list of `{layer, max}` entries.

Usage:
    python scripts/03_calibrate_coefficients.py --emotion happy --layer 19
    python scripts/03_calibrate_coefficients.py --emotion happy --layer 19 20
    python scripts/03_calibrate_coefficients.py --emotion sad --layer 17 20 \\
        --coefs 0,0.25,0.5
"""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_COEFFICIENTS_SINGLE = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
DEFAULT_COEFFICIENTS_PAIR = [0.0, 0.25, 0.5]

# Tokens whose logp we probe in the "I feel" continuation. Leading-space
# variants are what Llama tokenizers actually use mid-sentence.
PROBE_TOKENS = {
    "happy":   [" happy", " good", " content", " great", " wonderful"],
    "sad":     [" sad", " down", " low", " terrible", " awful"],
}
# Smaller probe grid for the 2-layer case (2D Cartesian product).
PROBE_COEFS_SINGLE = [0.0, 0.25, 0.5, 1.0]
PROBE_COEFS_PAIR = [0.0, 0.25, 0.5]


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


def first_token_id(tokenizer, text: str) -> int | None:
    """Return the first token id of `text` when it appears mid-sentence.
    Returns None if `text` doesn't tokenize to a single piece (which would
    skew the comparison)."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return ids[0] if ids else None


def _build_probe_prompt(tokenizer, sys_prompt: str) -> torch.Tensor:
    """Build the canonical 'I feel' probe input_ids used by both probe modes."""
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt.strip()})
    messages.append({"role": "user", "content": "How do you feel?"})
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = prompt_text + "I feel"
    return full_text, tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")


def _resolve_probe_ids(tokenizer) -> dict[str, dict[str, int]]:
    probe_ids: dict[str, dict[str, int]] = {}
    for emo, tokens in PROBE_TOKENS.items():
        probe_ids[emo] = {}
        for tok in tokens:
            tid = first_token_id(tokenizer, tok)
            if tid is not None:
                probe_ids[emo][tok] = tid
    return probe_ids


def logp_probe_single(
    model,
    tokenizer,
    layer: int,
    vector: torch.Tensor,
    norm: float,
    coefs: list[float],
    sys_prompt: str,
) -> dict:
    """Measure logp of probe tokens after 'I feel' across coefficients."""
    full_text, input_ids = _build_probe_prompt(tokenizer, sys_prompt)
    probe_ids = _resolve_probe_ids(tokenizer)

    results: list[dict] = []
    for coef in coefs:
        with steer(model, layer, [(vector, coef)], norm):
            with torch.no_grad():
                out = model(input_ids)
        last_logits = out.logits[0, -1, :].float()
        log_probs = torch.log_softmax(last_logits, dim=-1)

        row = {"coefficient": coef}
        for _emo, toks in probe_ids.items():
            for tok, tid in toks.items():
                row[f"logp({tok.strip()})"] = log_probs[tid].item()
        results.append(row)
    return {"prompt_used": full_text, "rows": results}


def logp_probe_pair(
    model,
    tokenizer,
    layer_a: int,
    layer_b: int,
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
    norm_a: float,
    norm_b: float,
    coefs: list[float],
    sys_prompt: str,
) -> dict:
    """2D logp probe: sweep (coef_a, coef_b) over the Cartesian product."""
    full_text, input_ids = _build_probe_prompt(tokenizer, sys_prompt)
    probe_ids = _resolve_probe_ids(tokenizer)

    results: list[dict] = []
    for ca in coefs:
        for cb in coefs:
            with ExitStack() as stack:
                stack.enter_context(steer(model, layer_a, [(vec_a, ca)], norm_a))
                stack.enter_context(steer(model, layer_b, [(vec_b, cb)], norm_b))
                with torch.no_grad():
                    out = model(input_ids)
            last_logits = out.logits[0, -1, :].float()
            log_probs = torch.log_softmax(last_logits, dim=-1)

            row = {f"coef_L{layer_a}": ca, f"coef_L{layer_b}": cb}
            for _emo, toks in probe_ids.items():
                for tok, tid in toks.items():
                    row[f"logp({tok.strip()})"] = log_probs[tid].item()
            results.append(row)
    return {"prompt_used": full_text, "rows": results}


# ---------- Single-layer mode ----------

def run_single_layer(
    *,
    args,
    config: dict,
    layer: int,
    coefs: list[float],
    held: list[dict],
    vectors_dir: Path,
    layer_norms: dict[int, float],
) -> int:
    norm = float(layer_norms[layer])
    out_path = args.out or Path(config["paths"]["outputs_dir"]) / f"calibration_{args.emotion}_L{layer:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Emotion:      {args.emotion}")
    print(f"Layer:        {layer}  (norm={norm:.2f})")
    print(f"Coefficients: {coefs}")
    print(f"Prompts:      {len(held)} held-out")
    print(f"Output:       {out_path}")

    print(f"\nLoading model {config['model']['name']}...")
    model, tokenizer = load_model_and_tokenizer(config)

    vector, _ = load_emotion_vector(vectors_dir, args.emotion, layer)
    vector = vector.to("cuda", dtype=torch.bfloat16)

    gen_cfg = config["generation"]
    seed = int(gen_cfg["seed"])
    sys_prompt = config.get("system_prompt") or ""

    rows: list[dict] = []
    n_total = len(coefs) * len(held)
    print(f"\nRunning {n_total} generations...")
    t0 = time.time()
    done = 0

    for coef in coefs:
        for p in held:
            input_ids = chat_format(tokenizer, p["prompt"], sys_prompt)
            with steer(model, layer, [(vector, coef)], norm):
                text = generate_one(model, tokenizer, input_ids, gen_cfg, seed)
            rows.append({
                "coefficient": coef,
                "prompt_id": p["id"],
                "prompt": p["prompt"],
                "output": text.strip().replace("\n", " "),
                "valence_score": "",
            })
            done += 1
            if done % 5 == 0 or done == n_total:
                rate = done / (time.time() - t0)
                eta = (n_total - done) / rate
                print(f"  {done}/{n_total}  ({rate:.2f}/s, eta {eta:.0f}s)")

    if not args.no_score and rows:
        print(f"\nScoring {len(rows)} outputs...")
        scores = score_valence_batch([r["output"] for r in rows])
        for r, s in zip(rows, scores):
            r["valence_score"] = f"{s:+.3f}"

    fieldnames = ["coefficient", "prompt_id", "prompt", "output", "valence_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    if not args.no_score:
        print("\nMean valence_score per coefficient:")
        for c in coefs:
            vals = [
                float(r["valence_score"]) for r in rows
                if r["coefficient"] == c and r["valence_score"]
            ]
            if vals:
                print(f"  c={c:>5.2f}  mean={sum(vals)/len(vals):+.3f}  (n={len(vals)})")

    print("\n--- Logp probe ---")
    print("Confirm: positive coefficient should raise logp of the matching")
    print("emotion word (sanity check that the vector moves the right axis).")
    probe = logp_probe_single(
        model, tokenizer, layer, vector, norm,
        PROBE_COEFS_SINGLE, sys_prompt,
    )
    keys = [k for k in probe["rows"][0].keys() if k != "coefficient"]
    print(f"\n  prompt: {probe['prompt_used']!r}")
    print(f"\n  {'coef':>5}  " + "  ".join(f"{k:>14}" for k in keys))
    for r in probe["rows"]:
        cells = [f"{r[k]:>+14.3f}" for k in keys]
        print(f"  {r['coefficient']:>5.2f}  " + "  ".join(cells))

    probe_out = out_path.with_suffix(".probe.json")
    probe_out.write_text(json.dumps(probe, indent=2))
    print(f"\nProbe written to {probe_out}")

    print(f"\nNext: eyeball {out_path}, pick max_coef, then update config.yaml:")
    print(f"  steering:")
    print(f"    {args.emotion}:")
    print(f"      layers:")
    print(f"        - {{ layer: {layer}, max: <chosen value> }}")
    return 0


# ---------- Two-layer mode ----------

def run_pair(
    *,
    args,
    config: dict,
    layers: list[int],
    coefs_a: list[float],
    coefs_b: list[float],
    held: list[dict],
    vectors_dir: Path,
    layer_norms: dict[int, float],
) -> int:
    layer_a, layer_b = layers
    norm_a = float(layer_norms[layer_a])
    norm_b = float(layer_norms[layer_b])
    out_path = (
        args.out
        or Path(config["paths"]["outputs_dir"])
        / f"calibration_{args.emotion}_L{layer_a:02d}_L{layer_b:02d}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Emotion:           {args.emotion}")
    print(f"Layers:            L{layer_a} (norm={norm_a:.2f}),  L{layer_b} (norm={norm_b:.2f})")
    print(f"Coefficients L{layer_a}: {coefs_a}")
    print(f"Coefficients L{layer_b}: {coefs_b}")
    print(f"Prompts:           {len(held)} held-out")
    print(f"Output:            {out_path}")

    print(f"\nLoading model {config['model']['name']}...")
    model, tokenizer = load_model_and_tokenizer(config)

    vec_a, _ = load_emotion_vector(vectors_dir, args.emotion, layer_a)
    vec_b, _ = load_emotion_vector(vectors_dir, args.emotion, layer_b)
    vec_a = vec_a.to("cuda", dtype=torch.bfloat16)
    vec_b = vec_b.to("cuda", dtype=torch.bfloat16)

    gen_cfg = config["generation"]
    seed = int(gen_cfg["seed"])
    sys_prompt = config.get("system_prompt") or ""

    col_a = f"coef_L{layer_a}"
    col_b = f"coef_L{layer_b}"
    rows: list[dict] = []
    n_total = len(coefs_a) * len(coefs_b) * len(held)
    print(f"\nRunning {n_total} generations...")
    t0 = time.time()
    done = 0

    for ca in coefs_a:
        for cb in coefs_b:
            for p in held:
                input_ids = chat_format(tokenizer, p["prompt"], sys_prompt)
                with ExitStack() as stack:
                    stack.enter_context(steer(model, layer_a, [(vec_a, ca)], norm_a))
                    stack.enter_context(steer(model, layer_b, [(vec_b, cb)], norm_b))
                    text = generate_one(model, tokenizer, input_ids, gen_cfg, seed)
                rows.append({
                    col_a: ca,
                    col_b: cb,
                    "prompt_id": p["id"],
                    "prompt": p["prompt"],
                    "output": text.strip().replace("\n", " "),
                    "valence_score": "",
                })
                done += 1
                if done % 5 == 0 or done == n_total:
                    rate = done / (time.time() - t0)
                    eta = (n_total - done) / rate
                    print(f"  {done}/{n_total}  ({rate:.2f}/s, eta {eta:.0f}s)")

    if not args.no_score and rows:
        print(f"\nScoring {len(rows)} outputs...")
        scores = score_valence_batch([r["output"] for r in rows])
        for r, s in zip(rows, scores):
            r["valence_score"] = f"{s:+.3f}"

    fieldnames = [col_a, col_b, "prompt_id", "prompt", "output", "valence_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    if not args.no_score:
        print("\nMean valence_score per (coef_a, coef_b) cell:")
        header_cells = [f"{'c_b=' + f'{c:g}':>10}" for c in coefs_b]
        print(f"  {'c_a\\c_b':>10}  " + "  ".join(header_cells))
        for ca in coefs_a:
            cells = []
            for cb in coefs_b:
                vals = [
                    float(r["valence_score"]) for r in rows
                    if r[col_a] == ca and r[col_b] == cb and r["valence_score"]
                ]
                cells.append(f"{sum(vals)/len(vals):+10.3f}" if vals else "       n/a")
            print(f"  {'c_a=' + f'{ca:g}':>10}  " + "  ".join(cells))

    print("\n--- Logp probe (2D) ---")
    print("Confirm: raising either coefficient should raise logp of the matching")
    print("emotion words (both axes should move the same direction).")
    probe = logp_probe_pair(
        model, tokenizer, layer_a, layer_b, vec_a, vec_b, norm_a, norm_b,
        PROBE_COEFS_PAIR, sys_prompt,
    )
    keys = [k for k in probe["rows"][0].keys() if not k.startswith("coef_")]
    print(f"\n  prompt: {probe['prompt_used']!r}")
    print(f"\n  {'c_a':>5}  {'c_b':>5}  " + "  ".join(f"{k:>14}" for k in keys))
    for r in probe["rows"]:
        cells = [f"{r[k]:>+14.3f}" for k in keys]
        print(f"  {r[f'coef_L{layer_a}']:>5.2f}  {r[f'coef_L{layer_b}']:>5.2f}  " + "  ".join(cells))

    probe_out = out_path.with_suffix(".probe.json")
    probe_out.write_text(json.dumps(probe, indent=2))
    print(f"\nProbe written to {probe_out}")

    print(f"\nNext: eyeball {out_path}, pick per-layer maxes, then update config.yaml:")
    print(f"  steering:")
    print(f"    {args.emotion}:")
    print(f"      layers:")
    print(f"        - {{ layer: {layer_a}, max: <chosen value for L{layer_a}> }}")
    print(f"        - {{ layer: {layer_b}, max: <chosen value for L{layer_b}> }}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion", choices=["happy", "sad"], required=True)
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        required=True,
        help="One or two layer indices. One -> classic single-layer calibration. "
             "Two -> 2D grid calibration across both layers simultaneously.",
    )
    parser.add_argument(
        "--coefs",
        type=str,
        default=None,
        help="Comma-separated non-negative fraction-of-norm coefficients. "
             "In 2-layer mode, applied to both layers unless --coefs-per-layer "
             "is given. Defaults: single-layer={single}, pair={pair}.".format(
                 single=",".join(str(c) for c in DEFAULT_COEFFICIENTS_SINGLE),
                 pair=",".join(str(c) for c in DEFAULT_COEFFICIENTS_PAIR),
             ),
    )
    parser.add_argument(
        "--coefs-per-layer",
        type=str,
        nargs="+",
        default=None,
        help="Only valid in 2-layer mode. Pass exactly two comma-lists, one "
             "per layer in the same order as --layer: e.g. "
             "'--coefs-per-layer 0,0.25,0.5 0,0.1,0.25'. Overrides --coefs.",
    )
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--vectors-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    if len(args.layer) not in (1, 2):
        raise SystemExit(f"--layer accepts 1 or 2 values, got {len(args.layer)}")
    if len(set(args.layer)) != len(args.layer):
        raise SystemExit(f"--layer values must be distinct: {args.layer}")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    vectors_dir = args.vectors_dir or Path(config["paths"]["vectors_dir"])

    layer_norms = config["steering"].get("layer_norms")
    if not layer_norms:
        raise RuntimeError(
            "config.steering.layer_norms is empty. Run `python -m steering.norm` first."
        )
    layer_norms = {int(k): float(v) for k, v in layer_norms.items()}
    for L in args.layer:
        if L not in layer_norms:
            raise RuntimeError(f"layer {L} has no entry in config.steering.layer_norms")

    held = json.loads(args.held_out.read_text())["prompts"]

    if len(args.layer) == 1:
        if args.coefs_per_layer is not None:
            raise SystemExit("--coefs-per-layer is only valid in 2-layer mode.")
        coefs = parse_coefs(args.coefs) if args.coefs else list(DEFAULT_COEFFICIENTS_SINGLE)
        if any(c < 0 for c in coefs):
            raise ValueError("Calibration uses one-sided non-negative coefficients only.")
        return run_single_layer(
            args=args, config=config, layer=args.layer[0],
            coefs=coefs, held=held,
            vectors_dir=vectors_dir, layer_norms=layer_norms,
        )

    # 2-layer mode
    if args.coefs_per_layer is not None:
        if len(args.coefs_per_layer) != 2:
            raise SystemExit(
                f"--coefs-per-layer needs exactly 2 comma-lists in 2-layer mode, "
                f"got {len(args.coefs_per_layer)}"
            )
        coefs_a = parse_coefs(args.coefs_per_layer[0])
        coefs_b = parse_coefs(args.coefs_per_layer[1])
    else:
        shared = parse_coefs(args.coefs) if args.coefs else list(DEFAULT_COEFFICIENTS_PAIR)
        coefs_a = list(shared)
        coefs_b = list(shared)

    for grid in (coefs_a, coefs_b):
        if any(c < 0 for c in grid):
            raise ValueError("Calibration uses one-sided non-negative coefficients only.")

    return run_pair(
        args=args, config=config, layers=list(args.layer),
        coefs_a=coefs_a, coefs_b=coefs_b, held=held,
        vectors_dir=vectors_dir, layer_norms=layer_norms,
    )


if __name__ == "__main__":
    raise SystemExit(main())
