# Seasonal Affective Disorder LLM — Phase 1 TODOs

Weather-steered emotion prototype using Llama 3.1 8B Instruct + activation steering.
The model gets happier in nice weather (summer vibes) and sadder in bad weather
(winter vibes). Two emotion vectors: **happy** and **sad**, both extracted relative
to a **neutral baseline**. Following Sofroniew et al. 2026 (Anthropic, "Emotion
Concepts and their Function in a Large Language Model"), we treat them as separate
vectors that may not be anti-parallel — nice weather elevates the happy vector,
bad weather elevates the sad vector, and they can overlap or diverge in residual
stream geometry however the model's representations actually lie.

Target hardware: rented A40 48GB ($0.44/hr, 50GB RAM, 9 vCPU).

Conventions:
- `[L]` = local laptop work (no GPU needed)
- `[P]` = pod work (requires rented GPU)
- `[L/P]` = developed locally, validated on pod
- Order is roughly sequential; parallelizable items called out.

---

## 0. Repo scaffolding `[L]`

- [ ] `git init`, `.gitignore` (ignore `vectors/`, `outputs/`, `logs/`, `models/`, `__pycache__/`, `.venv/`)
- [ ] `README.md` with project summary + run order
- [ ] `requirements.txt` with pinned versions:
  - `torch`, `transformers`, `nnsight`, `accelerate`
  - `requests`, `pyyaml`, `pydantic`
  - `pandas` (CSV handling for sweeps)
  - sentiment classifier deps (see §7) — likely `transformers` is enough
- [ ] `setup.sh` for pod: create venv, `pip install -r requirements.txt`, verify CUDA
- [ ] `config.yaml` skeleton: model name, layer (filled after sweep), per-layer residual stream norm (filled by `steering/norm.py`), per-vector coefficient range in fraction-of-norm units (filled after calibration), location lat/long
- [ ] Directory layout:
  ```
  feelings/
    prompts/emotion_examples.json   # 30 topics x {happy, sad, neutral} x 4 examples
    prompts/held_out.json
    scripts/
      00_smoke_test.py
      01_extract_vectors.py
      02_layer_sweep.py
      03_calibrate_coefficients.py
    inputs/
      weather_api.py
      mapping.py
    eval/
      sentiment.py
    steering/
      hooks.py            # shared multi-vector steering hook
      norm.py             # residual stream norm computation (for fraction-of-norm coefficients)
      io.py               # vector save/load + metadata
    download_model.py
    run.py
    config.yaml
  ```

---

## 1. Emotion examples dataset `[L]`

The single most important artifact in Phase 1. Worth real effort *before* extraction code.

Following the paper's two-vector framing: we extract a happy vector and a sad
vector independently, **each computed against a neutral baseline**. The neutral
examples are emotionally flat statements about the same topics. This lets the
two vectors point in genuinely independent directions in residual stream space
rather than being forced anti-parallel as a single bipolar axis.

- [ ] `prompts/emotion_examples.json` schema: list of `{topic, happy: [...], sad: [...], neutral: [...]}` where each emotion list contains ~4 example statements per topic
- [ ] Build 30 topics × 4 examples × 3 emotions = **360 statements** (120 per emotion)
  - Topics: 3 per category across 10 categories (food, work, leisure, relationships, hobbies, travel, home, self, events, life)
  - Within a topic, all 12 statements (4 happy + 4 sad + 4 neutral) share the topic noun and approximate length so per-topic mean-difference cancels topic content
  - Each statement is ~15–30 tokens, first-person where natural, conversational register
  - Neutral examples are factual / observational about the same topic with no affect words ("I made coffee at 7am with the same beans I always buy")
- [ ] Sanity-eyeball: read all 12 statements for 5 random topics and confirm the only systematic difference between groups is emotional valence
- [ ] Hash the dataset (SHA256 of canonicalized JSON) so vectors can be tagged with their source

**Vector formula** (used by §4):
- `happy_vector = mean(happy_activations) - mean(neutral_activations)`
- `sad_vector = mean(sad_activations) - mean(neutral_activations)`
- Each L2-normalized for storage. Cosine similarity between the two will be reported in `01_extract_vectors.py` output — if it's strongly negative (~-0.9) they're effectively a single axis; if it's small (|cos| < 0.5) they're genuinely independent.

---

## 2. Pod setup `[P]` — one-time per rental

- [ ] Mount persistent network volume at `/workspace` (so weights + vectors survive pod restarts)
- [ ] `bash setup.sh` — install pinned deps, verify `nvidia-smi` shows A40
- [ ] `python download_model.py` — pull Llama 3.1 8B Instruct in bf16 to `/workspace/models/`. Idempotent (skip if exists). ~10–20 min, ~16GB.
- [ ] HuggingFace auth: `huggingface-cli login` with token (Llama 3.1 is gated)

---

## 3. `00_smoke_test.py` `[P]` — confirm hooks fire

- [ ] Load model in bf16 onto CUDA, print VRAM footprint
- [ ] Apply Llama 3.1 chat template to a test prompt
- [ ] Register a no-op hook on residual stream at layer 14
- [ ] Generate 20 tokens, assert hook fired and captured tensor shape `[batch, seq, 4096]`
- [ ] Print first 5 generated tokens — confirm coherent output
- [ ] **Run this every fresh pod before anything else.** Catches dtype, device, chat-template, and nnsight-API breakage in 30 seconds.

---

## 4. `01_extract_vectors.py` `[P]` — the workhorse

- [ ] CLI: `--examples prompts/emotion_examples.json --out vectors/`
- [ ] For each layer in **10–27 (18 layers)**:
  - For each statement (across all topics × emotions), run forward pass with chat template applied
  - Capture residual stream at the **last token position only**
  - Group activations by emotion (happy / sad / neutral), compute per-group means
  - `happy_vector_layerN = mean(happy_acts) - mean(neutral_acts)`
  - `sad_vector_layerN = mean(sad_acts) - mean(neutral_acts)`
  - L2-normalize each
- [ ] Save two `.pt` files per layer: `vectors/happy_L21.pt`, `vectors/sad_L21.pt` (etc. for all swept layers)
- [ ] Each `.pt` includes metadata dict: `{model, dataset_hash, emotion, layer, pooling: "last_token", chat_template_applied: true, n_positive, n_neutral, dtype, raw_norm_before_l2}`
- [ ] After extraction, print **cosine similarity** between happy and sad vectors at every layer. Interpretation:
  - cos ≈ -1: effectively a single bipolar axis (Turner-style result)
  - |cos| < 0.5: genuinely independent emotion concepts (paper-style result)
  - cos ≈ +1: bug, dataset issue, or vectors are the same direction (investigate)
- [ ] **Hard rule:** chat template must be applied identically here and at inference. Single most common silent failure.
- [ ] Runtime budget: ~10 min total (360 statements × 1 forward pass each + cheap mean math across 18 layers; activations cached during a single forward pass).

---

## 5. `02_layer_sweep.py` `[P]` — pick the productive layer

The paper's prior: emotion concepts that drive output behavior live in the
**middle-late** layers, "about two-thirds of the way through the model." For
Llama 3.1 8B (32 layers) that's around **layer 21**. The early-middle layers
encode the local emotional connotation of present text ("sensory") rather than
the planned-emotion concept that shapes upcoming generation ("action"). We want
the action layers, so the sweep is biased upward from the doc's old prior of 14.

- [ ] Run `steering/norm.py` first to populate `config.yaml` with per-layer mean residual stream norms (one-time, ~2 min)
- [ ] CLI: `--vectors vectors/ --emotion {happy,sad} --held-out prompts/held_out.json --out outputs/sweep_{emotion}.csv`
- [ ] Curate 5–10 held-out prompts (open-ended, no built-in valence)
- [ ] For each layer in **10–27 (18 layers)**, each fraction-of-norm coefficient in `[-0.5, 0, +0.5]`, each held-out prompt:
  - Steer at that layer with `coefficient × residual_norm[layer] × normalized_vector`
  - Generate ~100 tokens with **fixed seed**, **temperature 0.7** (NOT greedy)
  - Append row to CSV: `layer, coefficient, prompt, output, valence_score` (see §7)
- [ ] Eyeball CSV. Pick the layer where the +0.5 outputs read as the target emotion most clearly without going incoherent. Use `valence_score` as a tiebreaker (largest |Δscore| between +0.5 and -0.5 wins).
- [ ] Repeat for the other emotion. The paper's results suggest **the same layer works for both** vectors (emotion geometry is stable across mid-late layers per their RSA), but allow per-emotion choice if the data says otherwise.
- [ ] Runtime budget: ~45–60 min for both vectors across the full layer range

---

## 6. `03_calibrate_coefficients.py` `[P]` — find the workable range

**Multi-layer follow-up:** after the single-layer sweep surfaces a shortlist of
productive layers, the optional `scripts/02b_pair_sweep.py` (new, see `NOTES.md`
§12 when written) sweeps all unordered pairs of those layers with an
independent coefficient at each. `03_calibrate_coefficients.py --layer A B`
then calibrates a 2D grid over the chosen pair and produces a per-layer max
that plugs into the list-of-layers schema under `steering.{emotion}.layers`
in `config.yaml`. Single-layer calibration (`--layer A`) still works with a
one-entry list.

- [ ] CLI: `--vectors vectors/ --layer N --emotion {happy,sad} --out outputs/calibration_{emotion}.csv`
- [ ] Per-emotion sweep: fraction-of-norm coefficients `[0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]` × ~10 prompts
  - Note: only **non-negative** coefficients here. Negative steering on the happy vector is conceptually "anti-happy" but for SAD-LLM we want to *add* the sad vector when weather is bad, not *subtract* the happy vector. Keep both vectors as one-sided positive controls.
- [ ] Output CSV with valence scores (see §7)
- [ ] Pick `max_coef` per emotion: noticeable, on-target effect, no word salad. The paper's working range was 0.1–0.5; ours may differ since extraction methodology differs.
- [ ] **Logp probe** (paper-style validation, ~30s):
  - Prompt: `"Human: How do you feel?\nAssistant: I feel"`
  - For each emotion vector, sweep coefficient in `[0, 0.25, 0.5, 1.0]`
  - Measure logp of the next token being one of `[" happy", " good", " content", " sad", " down", " low"]`
  - Confirm: positive happy_coef ↑ logp(happy/good/content), positive sad_coef ↑ logp(sad/down/low). If not, the vector is moving emotional language in general but not the specific axis we think — investigate before continuing.
- [ ] **Combined coherence pass:** both vectors active simultaneously near max_coef. Confirms nothing breaks under additive steering at the same layer. If incoherent, reduce both max_coefs by 0.7×.
- [ ] Write chosen ranges back to `config.yaml`:
  ```yaml
  steering:
    layer: 21
    coefficient_units: fraction_of_residual_norm
    happy: { max: 0.5 }
    sad:   { max: 0.5 }
  ```

---

## 7. `eval/sentiment.py` `[L/P]` — lightweight classifier

Lightweight = small CPU/GPU model, runs alongside extraction without bloating VRAM.

- [ ] Pick model: `cardiffnlp/twitter-roberta-base-sentiment-latest` (~500MB, 3-class neg/neu/pos, well-calibrated for short text). Alternative if smaller is needed: `distilbert-base-uncased-finetuned-sst-2-english` (~250MB, binary).
- [ ] Wrap as `score_valence(text: str) -> float` returning a single signed score in `[-1, +1]` (positive = happy, negative = sad)
- [ ] Batch-score helper for CSV post-processing: `score_csv(path, text_col="output")`
- [ ] Validate: 20 hand-labeled outputs, confirm classifier agrees with your gut on ≥17 of them. If not, swap model.
- [ ] Wire into `02_` and `03_` so each generated row gets a `valence_score` column. Keep eyeball as ground truth, but use scores to:
  - Compute mean valence at each `(layer, coefficient)` cell
  - Surface the layer with the largest |Δvalence| between +4 and -4 as a tiebreaker
  - Spot-check that calibration max coefficients are actually moving valence, not just style

---

## 8. `inputs/weather_api.py` `[L]`

- [ ] Open-Meteo client: single GET, no auth, no rate limit concerns
- [ ] Returns typed dataclass/pydantic model: `{temperature_c, cloud_cover_pct, precipitation_mm, wind_kph, is_daytime, fetched_at}`
- [ ] Test against your real lat/long; print sample JSON; confirm field shapes
- [ ] Cache with 10-minute TTL so repeated `run.py` calls don't hammer the API
- [ ] Graceful failure: on network error, return last cached value with a stale flag

---

## 9. `inputs/mapping.py` `[L]`

- [ ] Pure functions, no I/O, easy to unit-test
- [ ] `weather_to_coefficients(weather, config) -> {happy: float, sad: float}` — non-negative fraction-of-norm coefficients in the calibrated range
- [ ] Compose a "weather niceness" signal in `[-1, +1]` from the inputs (clamped linear / sigmoid):
  - cloud_cover ↑ → niceness ↓
  - precipitation ↑ → niceness ↓
  - daylight (is_daytime + low cloud) → niceness ↑
  - temperature: prefer a band (e.g. 18–26°C peak), penalize extremes both ways
  - wind: mild penalty above ~25 kph
- [ ] Two-vector mapping (one-sided activation, both vectors stay non-negative):
  - `happy_coef = max(0, niceness) × happy_max`
  - `sad_coef = max(0, -niceness) × sad_max`
  - At niceness ≈ 0 (mediocre weather) both vectors are near zero and the model is essentially baseline
  - At niceness = +1 only the happy vector is on; at niceness = -1 only the sad vector is on
  - Avoids the "both emotions active simultaneously at half strength" muddle
- [ ] `pytest`-style tests with synthetic weather:
  - sunny noon, 22°C → happy near max, sad ≈ 0
  - overcast 8°C drizzle afternoon → happy ≈ 0, sad mild
  - midnight thunderstorm → happy ≈ 0, sad near max
  - clear winter night, -5°C → happy ≈ 0, sad moderate
  - mild overcast 15°C → both ≈ 0 (baseline behavior)

---

## 10. `steering/hooks.py` and `steering/norm.py` `[L/P]` — shared steering code

- [ ] `steering/norm.py`:
  - `compute_layer_norms(model, tokenizer, layers, corpus_texts) -> dict[int, float]`
  - For each requested layer, run forward passes over a few hundred tokens of arbitrary text and average the L2 norm of the residual stream across all positions
  - Persist results into `config.yaml` so subsequent scripts read them without re-computing
  - Run once after `00_smoke_test.py` succeeds; ~2 min total
- [ ] `steering/hooks.py`:
  - `MultiVectorSteeringHook(layer, vectors_with_coefs, layer_norm)` context manager
  - `vectors_with_coefs` is a list of `(vector, fraction_of_norm_coefficient)` tuples — supports any number of simultaneous vectors at the same layer
  - On every forward pass adds `sum(coef * layer_norm * vec)` to the residual stream at that layer (so coefficients are always in fraction-of-norm units)
  - Used identically by `02_`, `03_`, and `run.py` — single source of truth
  - Unit tests:
    - empty vector list → bit-identical to baseline output
    - single vector with coefficient 0 → bit-identical to baseline
    - two vectors with anti-parallel directions and equal coefficients → bit-identical to baseline (sanity check)

---

## 11. `run.py` `[P]` — end-to-end loop

- [ ] Load `config.yaml`, both vectors for chosen layer, model in bf16 (or 4-bit if VRAM tight)
- [ ] Pull weather → compute `{happy_coef, sad_coef}` → install MultiVectorSteeringHook → generate
- [ ] Log JSONL: `{timestamp, weather, niceness, happy_coef, sad_coef, prompt, output, valence_score}`
- [ ] CLI flags: `--prompt`, `--lat`, `--lon`, `--force-season {summer,winter}` (debug override that pins niceness to ±1.0 so you can demo without waiting for weather)
- [ ] Run 5–10 times across varied weather; spot-check logs
- [ ] **Acceptance criterion:** without being told the conditions, you can read three outputs and roughly guess the weather (summer-vibes vs winter-vibes)

---

## 12. Stretch (only if Phase 1 lands cleanly)

- [ ] Second emotion axis (tired/alert) for richer SAD analogue
- [ ] LLM-as-judge pass on calibration outputs for richer eval
- [ ] 4-bit quantize model for inference loop, re-run abbreviated calibration to confirm vector still steers cleanly
- [ ] Persistent log dashboard (simple Streamlit) over `logs/run.jsonl`

---

## Open questions to resolve before / during work

- [ ] Lat/long for weather (your actual location, or fixed demo location?)
- [ ] Default system prompt for `run.py` — does the model have a persona, or is it neutral assistant that happens to be moody?
- [ ] Where does `run.py` get its user prompt from — CLI, stdin, file?
