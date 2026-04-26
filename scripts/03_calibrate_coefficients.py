"""Calibrate the workable coefficient range for one emotion vector at the
chosen layer, plus a logp probe (paper-style validation).

For one emotion (happy or sad) at the layer chosen in 02_layer_sweep.py:
  1. Sweep non-negative fraction-of-norm coefficients over held-out prompts,
     write CSV with text + valence scores
  2. Logp probe: measure how steering shifts p(next token = emotion word)
     in "Human: How do you feel?\\nAssistant: I feel"

Pick max_coef from the sweep CSV (largest noticeable, on-target effect, no
incoherence) and write it back to config.yaml under steering.{emotion}.max.

Usage:
    python scripts/03_calibrate_coefficients.py --emotion happy --layer 21
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

DEFAULT_COEFFICIENTS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]

# Tokens whose logp we probe in the "I feel" continuation. Leading-space
# variants are what Llama tokenizers actually use mid-sentence.
PROBE_TOKENS = {
    "happy":   [" happy", " good", " content", " great", " wonderful"],
    "sad":     [" sad", " down", " low", " terrible", " awful"],
}


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


def first_token_id(tokenizer, text: str) -> int | None:
    """Return the first token id of `text` when it appears mid-sentence.
    Returns None if `text` doesn't tokenize to a single piece (which would
    skew the comparison)."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return ids[0] if ids else None


def logp_probe(
    model,
    tokenizer,
    layer: int,
    vector: torch.Tensor,
    norm: float,
    coefs: list[float],
    sys_prompt: str,
) -> dict:
    """Measure logp of probe tokens after 'I feel' across coefficients."""
    # Build "Human: How do you feel?\nAssistant: I feel" via chat template.
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt.strip()})
    messages.append({"role": "user", "content": "How do you feel?"})
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = prompt_text + "I feel"
    input_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

    # Resolve probe token ids for both emotions (so we can see both axes
    # at once and confirm the steering moves the target axis only).
    probe_ids: dict[str, dict[str, int]] = {}
    for emo, tokens in PROBE_TOKENS.items():
        probe_ids[emo] = {}
        for tok in tokens:
            tid = first_token_id(tokenizer, tok)
            if tid is not None:
                probe_ids[emo][tok] = tid

    results: list[dict] = []
    for coef in coefs:
        with steer(model, layer, [(vector, coef)], norm):
            with torch.no_grad():
                out = model(input_ids)
        last_logits = out.logits[0, -1, :].float()
        log_probs = torch.log_softmax(last_logits, dim=-1)

        row = {"coefficient": coef}
        for emo, toks in probe_ids.items():
            for tok, tid in toks.items():
                row[f"logp({tok.strip()})"] = log_probs[tid].item()
        results.append(row)
    return {"prompt_used": full_text, "rows": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion", choices=["happy", "sad"], required=True)
    parser.add_argument("--layer", type=int, required=True, help="Layer chosen from 02_layer_sweep results.")
    parser.add_argument(
        "--coefs",
        type=str,
        default=",".join(str(c) for c in DEFAULT_COEFFICIENTS),
        help="Comma-separated non-negative fraction-of-norm coefficients.",
    )
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--vectors-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    coefs = parse_coefs(args.coefs)
    if any(c < 0 for c in coefs):
        raise ValueError("Calibration uses one-sided non-negative coefficients only.")

    vectors_dir = args.vectors_dir or Path(config["paths"]["vectors_dir"])
    out_path = args.out or Path(config["paths"]["outputs_dir"]) / f"calibration_{args.emotion}_L{args.layer:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layer_norms = config["steering"].get("layer_norms")
    if not layer_norms:
        raise RuntimeError(
            "config.steering.layer_norms is empty. Run `python -m steering.norm` first."
        )
    norm = float({int(k): float(v) for k, v in layer_norms.items()}[args.layer])

    held = json.loads(args.held_out.read_text())["prompts"]
    print(f"Emotion:      {args.emotion}")
    print(f"Layer:        {args.layer}  (norm={norm:.2f})")
    print(f"Coefficients: {coefs}")
    print(f"Prompts:      {len(held)} held-out")
    print(f"Output:       {out_path}")

    print(f"\nLoading model {config['model']['name']}...")
    model, tokenizer = load_model_and_tokenizer(config)

    vector, _ = load_emotion_vector(vectors_dir, args.emotion, args.layer)
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
            with steer(model, args.layer, [(vector, coef)], norm):
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
    probe = logp_probe(
        model, tokenizer, args.layer, vector, norm,
        [0.0, 0.25, 0.5, 1.0], sys_prompt,
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
    print(f"    layer: {args.layer}")
    print(f"    {args.emotion}:")
    print(f"      max: <chosen value>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
