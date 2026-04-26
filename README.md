# Seasonal Affective Disorder LLM

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
