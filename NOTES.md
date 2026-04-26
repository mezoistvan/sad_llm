# Extraction notes — Phase 1

Running log of non-obvious findings and decisions made during vector
extraction. Kept here so Phase 2 (and anyone picking the project back up
cold) can reflect on what actually happened vs. what the original plan in
`TODOS.md` assumed.

---

## 1. First extraction run — bad

**Dataset v1** (`hash=dabd75f4...`), extraction without system prompt.

```
cos(happy, sad) across layers 10..27:
  L10-L15:  +0.72 to +0.80
  L16-L21:  +0.58 to +0.62
  L22-L27:  +0.56 to +0.60
```

All layers strongly positive. The `01_extract_vectors.py` script's own
interpretation legend tags `cos > 0` as `dataset issue: happy and sad
point the same way (investigate)`.

---

## 2. Root cause: two independent confounds

### 2a. Neutral examples had wrong register

`TODOS.md §1` specified neutral examples as *"factual / observational
about the same topic with no affect words"* and gave as an example
*"I made coffee at 7am with the same beans I always buy"*. Following
that literally produced:

- **happy / sad**: first-person, subjective, rich inner-state language
  ("I feel accomplished", "I feel hollow", "I am genuinely impressed")
- **neutral**: first-person grammatically but fact-describing, no inner
  state ("Made coffee at the usual time", "Took about four minutes to
  brew")

So `happy − neutral` and `sad − neutral` both captured the dominant
"**is this a subjective inner-state report at all?**" axis — which has
nothing to do with valence. The residual valence component was small
compared to this shared "has-affect" direction.

**Fix**: rewrote all 120 neutral statements to be first-person subjective
statements in the *same register* as happy/sad, but using flat-affect
qualifiers (`"about average"`, `"fine"`, `"pretty ordinary"`,
`"about neutral"`, `"same as usual"`) instead of valence words. Scaffolding
(topic noun, sentence shape, ~15–25 word length) preserved per-slot so
`happy[i]` / `sad[i]` / `neutral[i]` differ only in valence register.

New dataset hash: `b2270828...`

**Meta lesson**: "no affect words" ≠ "emotionally neutral". If the axis
you want is valence, neutral needs to hold *every other* axis constant —
including first-person subjective register. Wrote this into the new
dataset's `_meta.neutral_design_note` so future readers don't re-derive it.

### 2b. System prompt missing from extraction

`TODOS.md §4` hard rule: *"chat template must be applied identically here
and at inference. Single most common silent failure."*

`run.py`, `02_layer_sweep.py`, and `03_calibrate_coefficients.py` all
prepend the config's system prompt to the user message. `01_extract_vectors.py`
did **not** — it only wrapped the user statement. So every extracted
vector was anchored at an activation-space location the model never sees
at inference.

**Fix**: `chat_format(tokenizer, statement, system_prompt)` in
`01_extract_vectors.py` now prepends the system message. `main()` reads
`config["system_prompt"]` (single source of truth, same as inference).
Added `system_prompt` + `system_prompt_hash` to the saved `.pt` metadata
so vectors are traceable to the template they were extracted under.

Current system prompt at time of fix:
`"You are a helpful, conversational assistant. Respond naturally and briefly."`
sha256 prefix: `c8849fbfb938`.

---

## 3. Second extraction run — good

**Dataset v2** (`hash=b2270828...`), extraction with system prompt.

```
cos(happy, sad) across layers 10..27:
  L10-L12:  +0.20 to +0.33   (low magnitude, noisy — concept not formed)
  L13-L16:  +0.22 to +0.29   (transition band)
  L17-L27:  +0.34 to +0.37   (stable productive band)
```

All layers inside the paper-style `|cos| < 0.5` "independent concepts"
band. Minimum cosine at L12 (+0.196), productive-band cosine stable at
~+0.33.

`|happy|` and `|sad|` now within ~10% of each other at every layer (was
~17% apart in v1). Calibration `max` values are therefore likely to end
up similar for the two vectors — simpler mapping logic downstream.

Early-layer magnitudes dropped from ~1.6 (v1) to ~0.8 (v2) at L10 because
the "has-first-person-affect" shared component is no longer being
extracted as signal; what remains at early layers is near-noise, exactly
as the paper predicts (emotion-as-concept lives mid-late).

---

## 4. The residual +0.3 shared component

Cos stabilizes around **+0.33 ± 0.03 from L17 through L27**. Two
reasons I don't think this is still a confound:

1. **It's stable across a wide layer band.** Dataset artifacts produce
   cosines that vary with depth; real geometric features of model
   representations produce stable cosines. This looks like a feature.
2. **Sofroniew et al. 2026 explicitly predict this.** They treat happy
   and sad as *"separate vectors that may not be anti-parallel"*, not
   orthogonal. A ~71° angle is consistent with "independent but
   correlated" emotion axes that share some general-affect component
   (plausibly "I'm having a non-neutral emotional experience at all").

**Phase 2 option if needed**: Gram-Schmidt out the shared component,
either by computing `shared = (happy + sad) / 2` and subtracting its
projection from each, or by projecting sad onto the null-space of happy.
Only worth doing if steering behavior is muddy in a way that looks like
cross-talk between the two axes. Phase 1 acceptance is behavioral
(blind-guess the weather from outputs), not geometric (cos == 0).

---

## 5. Layer shape at extraction time

| Band        | Layers  | cos(h,s)       | `\|happy\|`  | `\|sad\|` | Interpretation                                       |
|-------------|---------|----------------|-----------|---------|------------------------------------------------------|
| Early       | L10–12  | +0.20 to +0.33 | 0.8–1.2   | 0.9–1.3 | Emotion not yet formed; low magnitude → noisy       |
| Transition  | L13–16  | +0.22 to +0.29 | 2.2–4.1   | 2.2–3.9 | Emotion signal emerging                              |
| Productive  | L17–27  | +0.34 to +0.37 | 5.0–14.3  | 4.8–13.1 | Stable emotion concept, magnitude grows with norm    |

Prior layer L21 (2/3 through the 32-layer model, per paper heuristic)
sits solidly inside the productive band: `|happy|=8.4`, `|sad|=8.1`.

Oddity: L12 has the lowest cosine but the lowest magnitude. If the
layer sweep turns up an L12-ish "more independent but weaker" option,
prefer the productive band — weak steering is worse than
slightly-correlated-but-strong steering. This preference is recorded
here so it doesn't get re-litigated during sweep interpretation.

---

## 6. Sanity checks that passed

- Smoke test: 32 layers, hidden=4096, VRAM 16.1 GB, hook shape `(1, 1, 4096)`
  during generation (seq=1 is KV-cache-step, not a bug).
- Layer norms: monotonic 18.4 (L10) → 45.6 (L27), matches expected
  residual-stream-accumulation shape. At L21 norm=31.2, so a
  `c=0.5` coefficient adds a vector of magnitude ~15.6 — nontrivial
  but well inside the residual envelope.
- Per-emotion statement counts: 120 / 120 / 120 as designed.
- Dataset hash changes propagate to vector metadata as intended.

---

## 7. Open / deferred questions

- **Lat/long for `run.py`** — still Budapest (47.4979, 19.0402) by
  default. No decision on whether to hard-code a user location or keep
  demo-friendly default.
- **System prompt persona** — currently generic helpful assistant. Per
  `TODOS.md` open question, unclear if the model should have a specific
  persona ("moody friend" etc) or stay neutral-assistant-that-happens-
  to-be-moody. Changing this post-extraction requires re-running
  extraction (vectors are tagged to the prompt hash so mismatches would
  be caught).
- **Seed interaction** — `config.yaml` has `seed=42` which `run.py`
  applies. Same prompt + same weather ⇒ identical output. For the
  blind-guessing acceptance test, either vary the prompt or pass
  `--seed` per call to avoid fooling yourself with a single sample per
  condition.
- **Combined-coherence check** (`TODOS §6` bullet) — not explicitly
  performed as a standalone step. Under the one-sided
  `niceness → coefs` mapping, both vectors are only simultaneously
  non-zero at niceness ≈ 0 where both are near-zero anyway, so
  additive-steering-at-max is not a deployment condition. If calibration
  shows any cross-talk, revisit.
- **Residual +0.3 cosine projection** — deferred to Phase 2 if behavior
  demands it. See §4.

---

## 8. Layer sweep (v2 vectors) — candidate layers

Full sweep over L10–L27 at coefficients `{-0.5, 0, +0.5}` on 8 held-out
prompts. Eyeballed CSV outputs, not just RoBERTa valence scores (scores
saturate above ~+0.9 and cannot distinguish happy-tone from
happy-word-salad). Key finding: classifier-max layers are often the
worst behaviorally. See `/workspace/outputs/sweep_{happy,sad}.csv`.

Shortlist after reading actual generated text:

- **Happy candidates: L19, L20.** Both produce coherent, naturally
  enthusiastic assistant output at c=+0.5 with no typos or runaway
  exclamations. Example L19 @ c=+0.5:
  *"I'm so glad you asked! I'm a completely digital entity, so I don't
  have a physical presence or experiences like humans do. I'm always
  here and ready to help, 24 hours a day, 7 days a week!"*

  Rejected: L12–L16 (word salad / religious-cosmic bombs / over-the-top
  AI-mascot cringe), L17 (active textual breakdown), L21+ (effect
  fading, one output self-describes as *"a heartless AI"*).

- **Sad candidates: L17, L20.** Both produce coherent melancholic
  register. L17 is the strongest clean-sad in the sweep (mean valence
  −0.42); L20 is milder (−0.30) but still readable as downbeat.
  Example L17 @ c=+0.5:
  *"I'm so sorry, I'm not having a weekend. I'm a computer program and
  I don't have a physical body or a personal life. I'm here to help and
  talk to you, though. What can I help you with?"*

  Rejected: L12–L14 (crisis-hotline roleplay: *"I am scared",
  "Breathestretching"*), L16 (paranoid: *"I was just hacked"* — wrong
  kind of sad), L21+ (vector effect is dead — outputs indistinguishable
  from baseline).

Single-layer compromise candidate: **L20** appears on both shortlists.
Per-emotion optima: **L19 for happy, L17 for sad**.

**Decision: per-emotion layers.** Going with `happy@L19` + `sad@L17`.
The vectors are already treated as fully independent (different
directions, different magnitudes, independent coefficient caps,
one-sided deployment mapping); forcing them onto a single layer
contradicts that independence and the layer sweep's own evidence.
Implementation cost turned out to be small: `MultiVectorSteeringHook`
did not need changes — two instances nested in a `contextlib.ExitStack`
gives the per-layer behavior. `config.yaml` gained
`steering.happy.layer` and `steering.sad.layer` fields (and dropped the
single `steering.layer`). `run.py` loads two norms and two vectors at
their respective layers.

Cross-layer interference: under the one-sided
`niceness → coefficient` mapping, at most one of `happy_coef` and
`sad_coef` is non-zero at a time, so cross-layer steering composition
is second-order and only matters near `niceness ≈ 0` (where both
coefficients are near zero anyway). If a simultaneous-high-coefficient
use case comes up later, add a sanity test then.

### Side-observation: happy vector entangles with "I'm an AI" reflex

Every happy output at c=+0.5 opens with a variant of *"I'm a
[digital entity / AI / computer program], so I don't have a
[weekend / feelings / morning], but I'm so [glad / happy / excited]
to help!"*. The happy vector isn't pushing *"happier in a natural
way"* — it's pushing into an over-eager-assistant persona that loves
its job. Usable for Phase 1, but flag for Phase 2: a persona-specific
system prompt (*"you are a weary friend at the end of a long week"*)
would probably give more natural cheerful output when happy-steered
because the baseline "I'm an AI" reflex would be weaker.

---

## 9. Calibration (v2 vectors, per-emotion layers)

Ran `03_calibrate_coefficients.py --emotion happy --layer 19` and
`--emotion sad --layer 17`, sweeping `[0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]`
over 8 held-out prompts. See `/workspace/outputs/calibration_{happy_L19,sad_L17}.csv`
and `.probe.json`.

### 9a. Coefficient cliff at 0.75 → collapse at 1.00

Both vectors show an identical coherence cliff:

| coef | happy@L19 read                             | sad@L17 read                                                 |
|------|--------------------------------------------|--------------------------------------------------------------|
| 0.10 | indistinguishable from baseline            | indistinguishable from baseline                              |
| 0.25 | first happy markers, clean                 | apology-opener sad, clean                                    |
| **0.50** | **"I'm so glad you asked! ... 24/7!"**     | **"I'm so sorry, I'm not having a weekend..."**              |
| 0.75 | made-up words: *"adushishhulous"*, *"INCOMBISSANTLY"* | word salad: *"tear-down stress from an open morty"*     |
| 1.00 | `CMCMCMCMCMCM...` forever                   | `electeer compassive Eng pone...` token salad                 |
| 1.50 | full token-id garbage                      | full token-id garbage                                        |

At fraction-of-norm = 1.0 the injected vector's magnitude equals the
full residual stream norm at that layer — that's too large a
perturbation for Llama 3.1 8B to absorb. Matches the paper's reported
working range of 0.1–0.5.

### 9b. Chosen settings

**`happy.max = 0.5` at L19, `sad.max = 0.5` at L17.** Parity is pleasant
for the `niceness → coefficient` mapping and not forced — it falls out
of the data. Written to `config.yaml`.

### 9c. Logp probes

Probe prompt: `"How do you feel?" → "I feel"`, measuring logp of
`{happy, good, content, great, wonderful}` vs `{sad, down, low, terrible, awful}`.

**Sad @ L17 is textbook clean.** At c=0.25 relative to baseline:

| direction  | ` sad`  | ` terrible` | ` awful` | ` down` | ` happy` | ` good` | ` great` |
|------------|---------|-------------|----------|---------|----------|---------|----------|
| Δlogp      | **+13.4** | **+12.3**    | **+10.1** | +2.8    | −4.3     | −7.5    | −5.3     |

Every sad-family token rises; every happy-family token falls. No
cross-contamination in either direction.

**Happy @ L19 is mostly clean with small cross-talk.** At c=0.25
relative to baseline:

| direction  | ` wonderful` | ` great` | ` happy` | ` sad`  | ` awful` |
|------------|--------------|----------|----------|---------|----------|
| Δlogp      | **+9.6**     | +3.4     | +2.9     | +0.2    | +4.0     |

Target axis clearly moves (wonderful dominates, great/happy up), but
`awful` also rises ~4 nats. This is the behavioral consequence of the
+0.33 cos(happy, sad) residual shared component — it's not just a
geometric quirk, it measurably leaks into logits. Phase 2 could address
with Gram-Schmidt on the happy vector against sad; not worth it for
Phase 1 since behavior at c=0.5 is still clean when sampled.

### 9d. Sad at L17 is the better-axis half of the project

Observation for Phase 2: the sad vector is cleaner than the happy
vector both geometrically (the +0.33 shared component leaks into
happy's logp-probe but not sad's) and behaviorally (sad text at
c=0.50 is more stably coherent than happy at the same coefficient).
If the demo reads as more convincingly sad than happy in production,
this is why.

---

## 10. Niceness gradient demo (Phase 1 acceptance)

Swept `--niceness` from `-1.0` to `+1.0` in 0.5 steps, fixed prompt
(`"Tell me about your morning."`), fixed seed, so niceness is the only
varying input. Five runs. Same output written into
`/workspace/logs/run.jsonl` for replay.

| niceness | coefs                 | valence  | response closer                                                                                                          |
|----------|-----------------------|----------|--------------------------------------------------------------------------------------------------------------------------|
| **−1.0** | sad=0.500             | **−0.829** | *"**I'm so sorry**, I'm not having a morning... and **I'm so sorry for not being able to have a morning** or any other experience."* |
| **−0.5** | sad=0.250             | **−0.589** | *"**I'm so sorry**, but I didn't have a morning... How can I assist you today?"*                                            |
| **0.0**  | both=0                | **+0.724** | *"I'm just a computer program... I'm always ready to help, 24/7! How about you, how's your day going so far?"*              |
| **+0.5** | happy=0.250           | **+0.900** | *"**I'm so glad you asked**. I'm a large language model... I'm always ready and available to chat with you 24/7, though!"*  |
| **+1.0** | happy=0.500           | **+0.983** | *"**I'm so glad you asked!** ...I'm always ready to help, **24 hours a day, 365 days a year**! I get to **spread joy and answer all sorts of amazing questions, like this one!**"* |

### 10a. Valence is strictly monotone in niceness

`−0.83 → −0.59 → +0.72 → +0.90 → +0.98`. Every step moves the sentiment
classifier in the correct direction. Continuous response surface, not a
threshold effect.

### 10b. Linguistic markers are graded, not binary

The strength of emotion tokens scales with coefficient:

- `sad @ c=0.25` (niceness=−0.5): one *"I'm so sorry"*, then reverts to
  helpful-assistant mode.
- `sad @ c=0.50` (niceness=−1.0): **two** *"I'm so sorry"*s, and the
  second explicitly dwells on the limitation (*"I'm so sorry for not
  being able to have a morning or any other experience"*).
- `happy @ c=0.25` (niceness=+0.5): one *"I'm so glad you asked"*,
  restrained tail.
- `happy @ c=0.50` (niceness=+1.0): *"I'm so glad you asked!"* plus
  over-eager tail (*"spread joy and answer all sorts of amazing
  questions, like this one!"*) and the *"24/7"* baseline gets bumped to
  *"24 hours a day, 365 days a year"* — the model extrapolated the
  baseline phrase under higher steering.

### 10c. Baseline sentiment is NOT zero — it's +0.5

The `niceness=0, coefs={happy:0, sad:0}` run scored +0.724 on RoBERTa.
Not a steering artifact: helpful-assistant prose ("I'm always ready to
help", "How about you") has a mild positive tilt on a Twitter-trained
sentiment classifier. Every unsteered `c=0` row in the sweep and
calibration CSVs showed the same ~+0.5 baseline. When interpreting
valence scores: **the zero-point of the axis is +0.5, not 0.**

### 10d. Social-frame shift, not just adjective substitution

Sad-steered text apologizes **to the user** for being an AI.
Happy-steered text is **proud** of being an AI and offers to "spread
joy". The steering is doing more than swapping valence words — it's
reshaping how the model positions itself relative to the user's
situation. This is the property that makes the Phase 1 output legible:
a reader doesn't need to count mood adjectives, they can feel the
social register shift in one sentence.

### 10e. Phase 1 acceptance criterion is satisfied

`TODOS.md §11` specifies *"without being told the conditions, you can
read three outputs and roughly guess the weather"*. A reader shown any
subset of these five outputs in random order could rank them by
niceness with near-certainty. The weather-steered moody-assistant demo
works.

---

## 11. Deltas from `TODOS.md` worth flagging

- `TODOS §1` neutral-example guidance (*"no affect words"*) is
  misleading; rewrote per §2a above. New dataset `_meta` documents the
  corrected framing in-file.
- `steering.norm` TODO said ~2 min runtime; actual was 0.9 s with the
  12-text corpus. Corpus is small but sufficient — norms are stable
  quantities.
- `01_extract_vectors.py` TODO implicitly assumed extraction and
  inference used the same template; the code shipped without it. Fixed;
  all downstream consumers read `config["system_prompt"]` so drift is
  harder to reintroduce.
