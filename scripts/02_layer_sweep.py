"""Layer sweep: find the productive layer for steering.

For one emotion (happy or sad), generate completions on the held-out prompt set
across all candidate layers and a coarse coefficient grid. Output a CSV with
text + valence scores so you can eyeball which layer's positive coefficient
produces clear, on-target emotional shifts without going incoherent.

Usage:
    python scripts/02_layer_sweep.py --emotion happy
    python scripts/02_layer_sweep.py --emotion sad

Output:
    outputs/sweep_{emotion}.csv  with columns:
        layer, coefficient, prompt_id, prompt, output, valence_score
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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

DEFAULT_COEFFICIENTS = [-0.5, 0.0, 0.5]


def parse_layer_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def parse_coefs(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",")]


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
    parser.add_argument("--layers", type=str, default="10-27")
    parser.add_argument(
        "--coefs",
        type=str,
        default=",".join(str(c) for c in DEFAULT_COEFFICIENTS),
        help="Comma-separated fraction-of-norm coefficients to sweep.",
    )
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--vectors-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-score", action="store_true", help="Skip RoBERTa scoring (faster).")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    layers = parse_layer_range(args.layers)
    coefs = parse_coefs(args.coefs)
    vectors_dir = args.vectors_dir or Path(config["paths"]["vectors_dir"])
    out_path = args.out or Path(config["paths"]["outputs_dir"]) / f"sweep_{args.emotion}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layer_norms = config["steering"].get("layer_norms")
    if not layer_norms:
        raise RuntimeError(
            "config.steering.layer_norms is empty. Run `python -m steering.norm` first."
        )
    layer_norms = {int(k): float(v) for k, v in layer_norms.items()}

    held = json.loads(args.held_out.read_text())["prompts"]
    print(f"Emotion:      {args.emotion}")
    print(f"Layers:       {layers}")
    print(f"Coefficients: {coefs}")
    print(f"Prompts:      {len(held)} held-out")
    print(f"Output:       {out_path}")

    print(f"\nLoading model {config['model']['name']}...")
    model, tokenizer = load_model_and_tokenizer(config)

    gen_cfg = config["generation"]
    seed = int(gen_cfg["seed"])
    sys_prompt = config.get("system_prompt") or ""

    rows: list[dict] = []
    n_total = len(layers) * len(coefs) * len(held)
    print(f"\nRunning {n_total} generations...")
    t0 = time.time()
    done = 0

    for L in layers:
        if L not in layer_norms:
            print(f"  skipping layer {L}: no norm in config", file=sys.stderr)
            continue
        norm = layer_norms[L]

        try:
            vector, vec_meta = load_emotion_vector(vectors_dir, args.emotion, L)
        except FileNotFoundError:
            print(f"  skipping layer {L}: vector not found in {vectors_dir}", file=sys.stderr)
            continue
        vector = vector.to("cuda", dtype=torch.bfloat16)

        for coef in coefs:
            for p in held:
                input_ids = chat_format(tokenizer, p["prompt"], sys_prompt)
                with steer(model, L, [(vector, coef)], norm):
                    text = generate_one(model, tokenizer, input_ids, gen_cfg, seed)
                rows.append({
                    "layer": L,
                    "coefficient": coef,
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

    fieldnames = ["layer", "coefficient", "prompt_id", "prompt", "output", "valence_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")

    if not args.no_score:
        print("\nMean valence_score per (layer, coefficient):")
        print(f"  {'layer':>5}  " + "  ".join(f"{'c=' + str(c):>9}" for c in coefs))
        for L in layers:
            cells = []
            for c in coefs:
                vals = [
                    float(r["valence_score"]) for r in rows
                    if r["layer"] == L and r["coefficient"] == c and r["valence_score"]
                ]
                cells.append(f"{sum(vals)/len(vals):+9.3f}" if vals else "       n/a")
            print(f"  {L:>5}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
