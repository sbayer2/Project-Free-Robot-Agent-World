# F33 preregistration — video-native physics baseline: does a 2026 general VL model (Qwen3.8-27B, video-trained) carry more usable contact physics than the 2025 language world model — and does *watching* the world beat looking at it?

*Preregistered 2026-08-20, before any Qwen3.8 response has been collected
on any pseudo-marble scene. Decision rules are committed as
`scripts/f33_verdicts.py` in the same commit as this document. This is a
**comparative baseline eval** in the F11 lineage, not a unity-verdict
instrument: it scores an external model against the exact MuJoCo answer
keys with the exact F11 metric path, and its comparison anchors are frozen
from `docs/FINDINGS.md` F11. The clip renderer, the harness `video`
condition, and the run driver come in later commits and may not alter the
constants or rules frozen here.*

---

## 1. Why this experiment exists

F11 measured a *language* world model (Qwen-AgentWorld-35B-A3B,
digital-domain-tuned, text-only): the reasoning format transferred, the
contact physics did not (essence gain 0.04×, one coherent ramp-contact
mistake), yet its topple call was better-calibrated than our own trained
model's. Dyna-2 then showed that *video* is the axis along which physical
content actually transfers at scale — in *training*. F33 asks the
input-side version of that question on a model whose training included
real video and physics: **Qwen3.8-27B (VL, video-capable), served
locally.** Three questions, one battery:

1. Does a 2026 general-purpose VL model beat the 2025 language world
   model at contact-physics prediction on the same scenes? (H1)
2. Does *video input* — watching a probe begin — buy physics content that
   static views don't? (H2, the input-side Dyna-2 question)
3. Can it at least *read* physics that already happened from video —
   the perception ceiling that makes a prediction null interpretable? (H3)

## 2. Model, serving, and provenance (frozen apparatus)

- **Model:** `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` + mmproj Q8_0
  (Qwen3-VL family; vision+video true), llama.cpp `llama-server` on
  `127.0.0.1:8080`, no auth, temperature 0 explicit per request
  (server defaults are temp 1.0 — never rely on them).
- **Provenance protocol (frozen):** the serving build is branch pr-27342
  + two local patches, verified inert for text and base64-PNG requests
  (UI = frontend-only; the mtmd patch lives in the video path our
  `+faststart` clips do not exercise via its bug branch). The PR's
  ~290 unaudited core-library insertions are dormant without a draft
  model. **Frozen decision:** the live build is accepted for all F33
  legs because this is a comparative baseline, not a verdict instrument
  for the unity question; the FINDINGS entry carries the caveat. Every
  leg records `/props` build hash AND server uptime, and the user
  confirms no restart occurred since the previous leg — a restart makes
  the on-disk mixed build tree (pr-27342 binary vs master-rebuilt libs)
  produce a hash-undescribed hybrid, and VOIDS legs run after it until
  the tree is rebuilt coherent.
- **Throughput budget (measured, roofline):** 13.6–14.0 tok/s dense
  generation; prompt 230–390 tok/s. This is memory-bandwidth-bound —
  F11's ~64 tok/s came from A3B MoE sparsity, unmatchable here.
- **Etiquette (frozen):** legs run only when the user's interactive
  multimodal work is idle; the user is told afterward (our requests can
  evict their ~69k-token KV context, silently costing minutes). No MLX
  work concurrent with any leg (one unified-memory pool; the serving
  stack alone has peaked at 58/64 GB).

**Apparatus note (2026-08-20, pre-pilot; no measurement affected):**
before any F33 request was made, the monitoring llama.cpp session
discovered that the then-recorded build hash (b10499-5d477d2f1) was
WRONG, not stale — the running binary had been built with uncommitted
working-tree changes, so its reported commit never described it. The
user performed a planned restart after those changes were committed and
the tree rebuilt coherently; the expected identity is now
b10501-eab19b649, functionally identical, truthfully hashed. The §2
protocol absorbs this: the hash is re-read from `/props` directly at
measurement time, never carried forward from transcription. Recorded
here because it is exactly the defect class the protocol exists for,
caught before it could enter the record.

## 3. Datasets and requests

- **Synthetic (primary):** `data/pm_big` **test split — the same 20
  scenes × 3 probes = 60 requests per leg that F11 and every LLM
  transfer eval used.** Comparability requires the identical split,
  normalizers, and scoring path (`llm_transfer.score_predictions`).
- **GSO (secondary, asset-gated):** no GSO dataset currently exists on
  disk. If GSO assets are available (or re-downloaded), a
  `generate_gso` dataset supplies a real-geometry arm whose ground
  truth is what OUR simulator does under assumed contact parameters —
  the F14 mass-blindness caveat applies verbatim and is disclosed with
  any GSO number. GSO legs run only after the primary legs' gates pass,
  and their absence does not block or void F33.

## 4. Conditions (six legs; four primary)

| leg | input | question |
|---|---|---|
| `essence` | true physics params, text (F11 condition, unchanged) | quantitative dynamics prediction |
| `vision` | 3 static rendered views (F11 condition, unchanged) | picture→physics inverse |
| `video_pred` | **pre-outcome clip prefix** of the probe | the world-model test: predict what happens next |
| `video_percep` | **full 2.0 s clip** — the outcome is visible | the perception ceiling: read what happened |
| `gso_video_pred` / `gso_video_percep` | same, GSO arm | real-geometry generalization (secondary) |

The `video_pred` / `video_percep` split is the load-bearing design
decision: feeding the full clip and calling it prediction would leak the
answer into the input. Both are scored with the same outcome-field
schema; only the clip span differs.

**Clip specification (frozen):** one fixed oblique camera, 256 px,
24 fps encode, H.264 mp4 with `-movflags +faststart` (moov atom first —
llama.cpp master's trailing-moov decode bug never triggers). The server
samples at fps_target 4.0. Prediction prefixes are **per-probe,
onset-anchored** — uniform fractions would leak differently per probe
(a 0.6 m drop contacts at ~0.35 s):

    drop   [0, 0.25 s]  — falling, strictly pre-contact
    tilt   [0, 0.30 s]  — on the ramp, motion onset only
    push   [0, 0.60 s]  — 0.4 s settle + the 0.1 s impulse visibly
                          delivered + 0.1 s; travel/topple not resolved

Prompts: text legs reuse F11's `SYSTEM_PROMPT` and state/action text
verbatim. Video legs keep the identical field-spec block and action
text; only the state paragraph changes ("the attached video shows this
exact object as the action begins/completes"). Kept minimal so the
stills-vs-video contrast is about the input, not the prompt.

## 5. Phases

- **Phase 0 — MME benchmark (context, no verdict):** the public MME
  perception+cognition battery against the endpoint (lmms-eval or
  VLMEvalKit, OpenAI-compatible driver). Scores land in the report's
  `meta.mme` as a general-capability anchor beside published scores.
  ~2.4k prompt-bound requests, est. 3–6 h.
- **Phase 1 — calibration pilot (Amendment 1 pins its outputs):**
  5 scenes × essence + 2 clips decoded end-to-end. Measures actual
  response length, truncation rate at a 16384 cap (Muse needed 16384;
  AgentWorld reasoned in ~17k chains), latency per request, and video
  ingestion sanity (frame sampling visible in `/props` slot logs).
  Amendment 1 pins: `--max-tokens`, expected minutes/request, and any
  clip-spec correction — decision rules and bars are NOT touchable.
- **Phase 2 — primary legs:** essence → vision → video_pred →
  video_percep, sequential, cache-resumable (responses cache per
  (scene, probe); reruns free). Est. 4–6 h per leg at Muse-comparable
  response lengths; worst case ~2×.
- **Phase 3 — GSO legs (secondary, only if assets available and gates
  passed).**

## 6. Hypotheses and frozen decision rules

Anchors frozen from FINDINGS F11 (pm_big test split): AgentWorld essence
**MSE 1.277, gain 0.04×, Brier 0.100, 8/21 fields**; appearance **0.789,
0.07×, 0.150, 4/21**; vision ≈ appearance-text (VL36); train-mean
baseline MSE 0.052; trained specialist 0.033–0.051 (gain 1.36–1.64×).

**H1 — cross-model, essence leg vs F11 essence (primary):**
r₁ = MSE(qwen3.8 essence) / 1.277.

    BETTER      r1 <= 0.75
    COMPARABLE  0.75 < r1 <= 1.25
    WORSE       r1 > 1.25

Co-reported regardless: gain, Brier (vs F11's calibrated 0.100 — the
bar a general model must meet to match the world model's judge quality),
fields-beating count, and the tilt-ramp-contact anatomy (does the 2026
model make F11's one coherent mistake?).

**H2 — video vs stills (the input-side Dyna-2 question):**
r₂ = MSE(video_pred) / MSE(vision). Readable only if H3 = READABLE.

    VIDEO-ADDS   r2 <= 0.75
    VIDEO-HURTS  r2 >= 1.33
    NULL         0.90 <= r2 <= 1.11
    INCONCLUSIVE otherwise

**H3 — perception ceiling (gates H2's null):**
r₃ = MSE(video_percep) / MSE(video_pred).

    READABLE    r3 <= 0.90   (it extracts more from a visible outcome
                              than a hidden one -- video is being seen)
    UNREADABLE  otherwise -> the model cannot read outcomes from video
                             at all; H2's NULL is unreadable (a
                             VIDEO-ADDS could still be read)

**H4 — descriptive, no verdict:** everything vs the in-world specialist
(1.36–1.64×) and the train-mean baseline; GSO/synthetic ratios if
Phase 3 runs; **consistency co-report** (the "coherence" proxy for an
external model, proposed not verdict-bearing): agreement between the
model's essence estimates implied by stills vs video legs, and
self-consistency of its predictions across conditions on the same scene.

## 7. Gates — all must pass before any verdict is read

    parse       >= 90% of a leg's responses yield a scorable JSON
                (extract_prediction); below -> that leg VOID.
    truncation  <= 10% of responses hit the max-tokens cap; above ->
                leg VOID, Amendment raises the cap and reruns (cache
                makes this cheap).
    provenance  build hash + uptime recorded per leg; user's no-restart
                confirmation on record; a restart mid-arc voids
                subsequent legs until the tree is rebuilt.
    pilot       Amendment 1 committed (cap + clip sanity) before any
                Phase-2 leg.
    video-sanity the pilot's clips must be demonstrably ingested
                (server logs show sampled frames > 0) -- guards
                against silently scoring a text-only fallback.

## 8. Predictions — frozen before any data

- **P1 (55/45):** H1 → BETTER. A 2026 general VL model trained on real
  video plausibly carries more contact physics than a 2025
  digital-domain world model — but F11's failure was a *prior about
  contact regimes*, and priors are exactly what general training may
  not fix. Barely-favored is honest.
- **P2 (55/45):** H2 → NULL. The F11 precedent (vision ≈
  appearance-text: added modality bought nothing) and the
  training-vs-input distinction (Dyna-2's payoff came from video in
  TRAINING; feeding video at inference is a different claim) both point
  to null. VIDEO-ADDS would be the exciting outcome; NULL is the
  expected one.
- **P3 (70/30):** H3 → READABLE. Reading a visible outcome is far
  easier than predicting a hidden one; a video-capable model should
  manage it. The 30 is real: mtmd frame sampling at 4 fps may simply
  miss the outcome-bearing frames of a 2 s episode.
- **P4:** Brier ≥ 0.15 (i.e., NOT matching AgentWorld's 0.100
  calibrated topple judge). Lean, not odds: F11's sharpest surprise was
  judge quality from a *world model*; general models hedge.

## 9. Cost

All local, phased, cache-resumable. Phase 0 ≈ 3–6 h; pilot ≈ 1 h;
four primary legs ≈ 16–24 h total (overnight sessions; worst case ~2×
if responses run to the cap); GSO legs optional +8–12 h plus dataset
generation. No training, no MLX contention (nothing else on the GPU
during legs).

## 10. Honest limits

- **Quantization confound:** Q4_K_M here vs F11's Q8 quant — cross-model
  comparisons ride on different precision. Disclosed, not correctable
  locally (no Q8 fits comfortably beside the serving stack's footprint).
- **Serving-stack confound:** llama.cpp vs F11's oMLX/MLX. Same API,
  different inference kernels.
- **20 scenes**, one world (pm_big, historical r = 0.263 coupling), one
  camera for clips, 4 fps frame sampling — the video legs measure what
  the model can do with ~2–8 frames, not dense video.
- **GSO ground truth** (if Phase 3 runs) is our simulator under assumed
  contact parameters — F14's mass-blindness caveat verbatim.
- The patched-branch acceptance is a frozen pragmatic decision; anyone
  re-running F33 for the record should use clean upstream master.
- Muse-Glimmer-30B transfer runs (2026-08-14, both legs complete on the
  same scenes) exist on disk (`runs/llm_transfer_muse_*_atem`) and are
  **excluded from this comparison by user decision** (dead end as VLM);
  no Muse numbers are published here.

## 11. Contamination disclosure

All F11 numbers, the Muse run results, F28/F29 outcomes, and the Dyna-2
report were seen before this design was written. No Qwen3.8 response has
ever been collected on any pseudo-marble scene; the model has never been
shown any pseudo-marble render or clip. The clip specification and
per-probe prefixes were designed from the probe battery's source
parameters (drop height 0.6 m, push impulse at 0.4 s), not from any
model output. The serving-stack facts (throughput, memory, build
provenance, decode-bug behavior) come from the independent llama.cpp
session's measurements, logged before this design existed.
