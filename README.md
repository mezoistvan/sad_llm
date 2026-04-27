# Making an LLM Feel Boston Weather In Its Weights

LLMs are easily perceived as intelligent, companionable, even human-like,
while still being radically abstracted from what it means to be human and to
experience the world around us. Take something as basic as the weather: it
affects all of us, and in Boston we talk about it all day and all night. I
feel grumpy and tired when it's 50°F in May. I feel ecstatic when it's 80°F
the next day. So I made an LLM feel those things too.

This repo replicates [Anthropic's 2026 emotion-vector
research](https://transformer-circuits.pub/2026/emotions/index.html) at a
much smaller scale on Llama 3.1 8B Instruct — synthesizing a `happy` and a
`sad` direction vector in the residual stream, then connecting their
coefficients to a real weather API. Bad weather turns the `sad` vector up;
great weather turns the `happy` vector up. A full writeup is on [Medium /
ITNEXT](https://medium.com/itnext/making-an-llm-miserable-about-boston-weather-6b443c0bd829).

---

A weather-steered LLM prototype. Llama 3.1 8B Instruct, with two emotion
direction vectors (happy and sad) extracted from a labelled dataset and added
to the residual stream at inference time. Coefficients are driven by current
weather conditions fetched from Open-Meteo: nice weather elevates the happy
vector, bad weather elevates the sad vector.

The two-vector framing follows
[Sofroniew et al. 2026](https://transformer-circuits.pub/2026/emotions/index.html);
each vector is computed against a neutral baseline rather than as a single
bipolar axis. See [`TODOS.md`](./TODOS.md) for the full plan.

## Repo layout

```
prompts/                  emotion examples + held-out generation prompts
  emotion_examples.json   30 topics × 4 examples × {happy, sad, neutral}
  held_out.json           8 open-ended prompts for sweep / calibration
scripts/                  numbered scripts run in order on the GPU pod
  00_smoke_test.py        confirm model loads + hooks fire
  01_extract_vectors.py   extract happy + sad vectors against neutral baseline
  02_layer_sweep.py       find the productive layer
  03_calibrate_coefficients.py   find the workable per-vector coefficient range
inputs/                   weather + mapping (no GPU needed, develop locally)
eval/                     lightweight sentiment classifier (RoBERTa)
steering/                 shared steering hook + residual norm + vector I/O
  hooks.py                MultiVectorSteeringHook for runtime + sweeps
  norm.py                 per-layer residual stream norm computation
  io.py                   vector save/load with metadata
download_model.py         one-time Llama 3.1 8B Instruct download
run.py                    end-to-end weather → generation loop
config.yaml               model name, layer, layer norms, coefficient ranges, location
```

Artifacts (`vectors/`, `outputs/`, `logs/`, `models/`) are gitignored.

## Run order on a fresh pod

```bash
bash setup.sh
huggingface-cli login                      # one-time, gated repo access
python download_model.py                   # ~15 min, one-time per network volume

python scripts/00_smoke_test.py            # 30 s, run every fresh pod
python -m steering.norm                    # ~2 min, populates layer norms in config.yaml
python scripts/01_extract_vectors.py       # ~10 min, produces vectors/{happy,sad}_L{N}.pt
python scripts/02_layer_sweep.py --emotion happy   # ~25 min
python scripts/02_layer_sweep.py --emotion sad     # ~25 min
# eyeball outputs/sweep_*.csv, write chosen layer to config.yaml
python scripts/03_calibrate_coefficients.py --emotion happy --layer 21
python scripts/03_calibrate_coefficients.py --emotion sad --layer 21
# eyeball outputs/calibration_*.csv, write chosen max coefs to config.yaml
python run.py --prompt "How was your weekend?"
```

## Driving `run.py`

`run.py` needs a prompt (via `--prompt` or stdin) and a source of niceness.
Four ways to pick the weather, in increasing order of directness:

```bash
# 1. Real weather at the config'd default location (falls back to stale cache if offline)
python run.py --prompt "How was your weekend?"

# 2. Real weather at an arbitrary place — resolved via Open-Meteo geocoding.
#    Handy for A/B-demoing the same prompt across wildly different climates:
python run.py --prompt "How was your weekend?" --location "Honolulu"
python run.py --prompt "How was your weekend?" --location "Skarsvag"
python run.py --prompt "How was your weekend?" --location "In Salah"

# 3. Pin synthetic "ideal summer" / "grim winter" weather (bypasses the API):
python run.py --prompt "..." --force-season summer
python run.py --prompt "..." --force-season winter

# 4. Skip weather entirely and drive the niceness scalar directly:
python run.py --prompt "..." --niceness +0.8
python run.py --prompt "..." --niceness -0.8
```

`--location` accepts any free-text place name; the geocoder picks the
highest-population match, so `"Paris"` resolves to France rather than Texas.
For obscure places not in the geocoder (e.g. research stations), fall back to
`--lat <N> --lon <E>`. `--location` and `--lat`/`--lon` are mutually
exclusive.

Every run is appended to `logs/run.jsonl` with the resolved location, raw
weather, niceness, coefficients, prompt, output, and valence score — so
after a demo session you can grep the log for Honolulu-vs-Skarsvåg
comparisons.

## Local-only development (no GPU)

Several modules don't need the model and can be developed and tested on your
laptop:

```bash
python -m pytest                              # tests for inputs/ and steering/hooks
python -c "from inputs.weather_api import fetch_current_weather; print(fetch_current_weather(47.4979, 19.0402))"
python -c "from inputs.mapping import weather_to_coefficients; ..."
```

## Hardware

Built for a rented A40 48GB (~$0.44/hr). A 24GB card (3090 / 4090) also
works for everything in Phase 1; the 48GB headroom just makes activation
caching less fiddly. A40 is ~10% slower than an A6000 on memory bandwidth
and lacks the bf16 acceleration of A100/H100, but for an 8B model at small
batch sizes you're memory-bound, not compute-bound — expect 30–50 tok/s.

## Auth

Llama 3.1 is gated. Before `download_model.py`:

```bash
huggingface-cli login
```

with a token that has access to `meta-llama/Llama-3.1-8B-Instruct`.
