"""End-to-end weather-steered generation.

Pipeline:
    weather (Open-Meteo)
      -> niceness ∈ [-1, +1]
      -> { happy_coef, sad_coef } in fraction-of-residual-norm units
      -> MultiVectorSteeringHook at the chosen layer
      -> sampled response

Logs every run as JSONL to logs/run.jsonl with full provenance (timestamps,
weather, niceness, coefficients, prompt, output, valence_score).

Usage:
    python run.py --prompt "How was your weekend?"
    echo "Tell me about your morning." | python run.py
    python run.py --prompt "..." --force-season summer
    python run.py --prompt "..." --lat 40.7 --lon -74.0
    python run.py --prompt "..." --location "Skarsvag, Norway"
    python run.py --prompt "..." --location "Honolulu"
    python run.py --prompt "..." --niceness -0.7   # skip weather, drive coefs directly
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from eval.sentiment import score_valence  # noqa: E402
from inputs.mapping import (  # noqa: E402
    NicenessWeights,
    compute_niceness,
    niceness_to_coefficients,
)
from inputs.weather_api import fake_weather, fetch_current_weather, geocode_location  # noqa: E402
from steering.hooks import steer  # noqa: E402
from steering.io import load_emotion_vector  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config.yaml"


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


def generate(model, tokenizer, input_ids, gen_cfg: dict, seed: int | None) -> str:
    if seed is not None:
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


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _niceness_arg(raw: str) -> float:
    """Argparse type: float restricted to [-1, +1] (same range as compute_niceness output)."""
    v = float(raw)
    if not -1.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(
            f"--niceness must be in [-1.0, +1.0], got {v}"
        )
    return v


def read_prompt(args) -> str:
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text
    raise SystemExit("No prompt provided. Use --prompt or pipe text via stdin.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Free-text place name (e.g. 'Skarsvag' or 'Honolulu'). Resolved "
            "to lat/lon via Open-Meteo's geocoder — handy for simulating the "
            "model under a specific city's weather. Mutually exclusive with "
            "--lat/--lon."
        ),
    )
    parser.add_argument(
        "--force-season",
        choices=["summer", "winter"],
        default=None,
        help="Bypass Open-Meteo and pin niceness to ±1 for demos.",
    )
    parser.add_argument(
        "--niceness",
        type=_niceness_arg,
        default=None,
        metavar="FLOAT",
        help=(
            "Bypass weather entirely and supply niceness in [-1, +1] directly. "
            "Useful for A/B-testing the coefficient range without waiting on "
            "weather. Mutually exclusive with --force-season."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    if args.force_season and args.niceness is not None:
        raise SystemExit(
            "--force-season and --niceness are mutually exclusive overrides."
        )
    if args.location is not None and (args.lat is not None or args.lon is not None):
        raise SystemExit(
            "--location and --lat/--lon are mutually exclusive; pick one."
        )

    user_prompt = read_prompt(args)
    config = yaml.safe_load(CONFIG_PATH.read_text())

    happy_layer = config["steering"]["happy"].get("layer")
    sad_layer = config["steering"]["sad"].get("layer")
    if happy_layer is None or sad_layer is None:
        raise RuntimeError(
            "config.steering.{happy,sad}.layer is null. Pick per-emotion layers "
            "from 02_layer_sweep results and write them to config.yaml."
        )
    happy_max = config["steering"]["happy"]["max"]
    sad_max = config["steering"]["sad"]["max"]
    if happy_max is None or sad_max is None:
        raise RuntimeError("config.steering.{happy,sad}.max is null. Run 03_calibrate first.")
    layer_norms = {int(k): float(v) for k, v in config["steering"]["layer_norms"].items()}
    happy_norm = layer_norms[happy_layer]
    sad_norm = layer_norms[sad_layer]

    location_label: str | None = None
    if args.location is not None:
        geo = geocode_location(args.location)
        lat, lon = geo.latitude, geo.longitude
        location_label = geo.pretty()
        print(f"[location: {location_label} ({lat:.3f}, {lon:.3f})]")
    else:
        lat = args.lat if args.lat is not None else config["location"]["lat"]
        lon = args.lon if args.lon is not None else config["location"]["lon"]

    # Three modes for deciding niceness:
    #   1. --niceness N        -> skip weather entirely, use N directly
    #   2. --force-season X    -> use a synthetic "ideal" or "grim" weather object
    #   3. default             -> fetch real weather from Open-Meteo
    weather = None
    if args.niceness is not None:
        niceness = args.niceness
        niceness_source = "cli_override"
        print(f"[niceness override: {niceness:+.3f}  (weather fetch skipped)]")
    elif args.force_season:
        weather = fake_weather(args.force_season, lat=lat, lon=lon)
        niceness = compute_niceness(weather, weights=NicenessWeights())
        niceness_source = "forced_season"
        print(f"[forced season: {args.force_season}]")
    else:
        weather = fetch_current_weather(lat, lon)
        if weather.is_stale:
            print("[weather: stale cache, network fetch failed]")
        niceness = compute_niceness(weather, weights=NicenessWeights())
        niceness_source = "openmeteo"

    coefs = niceness_to_coefficients(niceness, happy_max=happy_max, sad_max=sad_max)

    if weather is not None:
        print(f"Weather:   {weather.temperature_c:.0f}C  cloud={weather.cloud_cover_pct:.0f}%  "
              f"precip={weather.precipitation_mm:.1f}mm  wind={weather.wind_kph:.0f}kph  "
              f"day={weather.is_daytime}")
    print(f"Niceness:  {niceness:+.3f}  (source={niceness_source})")
    print(f"Coefs:     happy={coefs['happy']:.3f} @ L{happy_layer} (norm={happy_norm:.2f})  "
          f"sad={coefs['sad']:.3f} @ L{sad_layer} (norm={sad_norm:.2f})")

    print(f"\nLoading model {config['model']['name']}...")
    model, tokenizer = load_model_and_tokenizer(config)

    happy_vec, _ = load_emotion_vector(config["paths"]["vectors_dir"], "happy", happy_layer)
    sad_vec, _ = load_emotion_vector(config["paths"]["vectors_dir"], "sad", sad_layer)
    happy_vec = happy_vec.to("cuda", dtype=torch.bfloat16)
    sad_vec = sad_vec.to("cuda", dtype=torch.bfloat16)

    sys_prompt = config.get("system_prompt") or ""
    seed = args.seed if args.seed is not None else config["generation"].get("seed")

    input_ids = chat_format(tokenizer, user_prompt, sys_prompt)

    # Two hooks at potentially different layers. Under the one-sided
    # niceness -> coefficient mapping, at most one is non-zero at a time, so
    # cross-layer interference is second-order. If a coefficient happens to be
    # zero, the hook still installs and adds the zero-delta (no-op) — cleaner
    # than branching on coefficient values.
    with ExitStack() as stack:
        stack.enter_context(
            steer(model, happy_layer, [(happy_vec, coefs["happy"])], happy_norm)
        )
        stack.enter_context(
            steer(model, sad_layer, [(sad_vec, coefs["sad"])], sad_norm)
        )
        text = generate(model, tokenizer, input_ids, config["generation"], seed)

    text = text.strip()
    print(f"\n--- prompt ---\n{user_prompt}")
    print(f"\n--- response ---\n{text}")

    valence = None
    if not args.no_score:
        valence = score_valence(text)
        print(f"\nvalence_score: {valence:+.3f}")

    if not args.no_log:
        weather_dict = None
        if weather is not None:
            weather_dict = {
                "lat": weather.latitude,
                "lon": weather.longitude,
                "temperature_c": weather.temperature_c,
                "cloud_cover_pct": weather.cloud_cover_pct,
                "precipitation_mm": weather.precipitation_mm,
                "wind_kph": weather.wind_kph,
                "is_daytime": weather.is_daytime,
                "is_stale": weather.is_stale,
                "forced_season": args.force_season,
            }
        record = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "location_query": args.location,
            "location_resolved": location_label,
            "weather": weather_dict,
            "niceness_source": niceness_source,
            "niceness": niceness,
            "coefficients": coefs,
            "layers": {"happy": happy_layer, "sad": sad_layer},
            "prompt": user_prompt,
            "output": text,
            "valence_score": valence,
        }
        log_path = Path(config["paths"]["logs_dir"]) / "run.jsonl"
        append_jsonl(log_path, record)
        print(f"\nLogged to {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
