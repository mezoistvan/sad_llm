"""Confirm the model loads, the chat template applies, hooks fire on the
residual stream, and generation produces coherent tokens.

Run this every fresh pod before anything else. ~30 seconds. Exits non-zero on
any failure so it can be wired into a `&&` chain.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Llama 3.1 8B has 32 transformer layers and hidden size 4096. Layer 14 sits
# in the prior band where valence-style steering tends to live.
PROBE_LAYER = 14
EXPECTED_HIDDEN = 4096


def load_model_and_tokenizer(config: dict):
    cache_dir = Path(config["model"]["cache_dir"])
    model_name = config["model"]["name"]
    local_path = cache_dir / model_name.replace("/", "__")

    if not local_path.exists():
        raise FileNotFoundError(
            f"Model not found at {local_path}. Run download_model.py first."
        )

    dtype = getattr(torch, config["model"]["dtype"])
    print(f"Loading tokenizer from {local_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    print(f"Loading model from {local_path} (dtype={dtype})")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=dtype,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def report_vram() -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: allocated={allocated:.1f}GB  reserved={reserved:.1f}GB  total={total:.1f}GB")


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available.", file=sys.stderr)
        return 1
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    model, tokenizer = load_model_and_tokenizer(config)
    report_vram()

    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} transformer layers, hidden size "
          f"{model.config.hidden_size}")

    if model.config.hidden_size != EXPECTED_HIDDEN:
        print(
            f"WARN: hidden size {model.config.hidden_size} != expected {EXPECTED_HIDDEN}. "
            f"Update PROBE_LAYER assumptions.",
            file=sys.stderr,
        )

    # Apply the official Llama 3.1 chat template. Vectors extracted later MUST
    # be extracted with the same wrapping.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in exactly five words."},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    print(f"Prompt token ids shape: {tuple(prompt_ids.shape)}")

    # Register a no-op forward hook on the residual stream after layer 14.
    # In Llama, model.model.layers[i].forward returns a tuple whose first
    # element is the hidden state of shape [batch, seq, hidden].
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["shape"] = tuple(hidden.shape)
        captured["dtype"] = hidden.dtype
        captured["device"] = hidden.device

    handle = model.model.layers[PROBE_LAYER].register_forward_hook(hook)

    print(f"Generating 20 tokens with hook on layer {PROBE_LAYER}...")
    try:
        with torch.no_grad():
            out_ids = model.generate(
                prompt_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    if "shape" not in captured:
        print("FAIL: hook never fired.", file=sys.stderr)
        return 1

    print(f"Hook captured tensor: shape={captured['shape']}  "
          f"dtype={captured['dtype']}  device={captured['device']}")

    if captured["shape"][-1] != model.config.hidden_size:
        print(
            f"FAIL: captured last dim {captured['shape'][-1]} != "
            f"hidden size {model.config.hidden_size}",
            file=sys.stderr,
        )
        return 1

    new_token_ids = out_ids[0, prompt_ids.shape[1]:]
    completion = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    print(f"First 5 token ids: {new_token_ids[:5].tolist()}")
    print(f"Completion: {completion!r}")

    if not completion.strip():
        print("FAIL: empty completion.", file=sys.stderr)
        return 1

    report_vram()
    print("\nOK: smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
