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

## Amendment A (2026-08-20, pre-pilot, zero data) — video-ingestion tripwires hardened

*Written before the clip renderer exists and before any request has been
made. Tightens (never relaxes) the §7 `video-sanity` gate, based on the
monitoring llama.cpp session's measured silent-failure signature.*

**The failure mode this guards:** both known video-decode bugs fail
SILENTLY — the server returns HTTP 200, the model reports no video was
attached, and a scored eval reads that as a model capability result, not
an apparatus fault. The two fixes for those bugs are **local-only
commits the user has decided to keep local**; clean upstream master has
both bugs. Consequences frozen here:

1. **`-movflags +faststart` is an apparatus REQUIREMENT, not a spec
   detail.** The trailing-moov bug is not phone-specific — it hits any
   default-muxed MP4 fed through the server's pipe (backward seek for
   the index fails; zero frames decode). Faststart clips decode
   correctly on BOTH the patched :8080 build AND clean upstream master,
   which **removes the build-dependent confound from the video legs**:
   a FINDINGS-grade clean-master rerun and the pilot build produce
   comparable video ingestion.
2. **Rotation side-data must be asserted absent, not assumed:** the
   renderer verifies `ffprobe -show_entries stream_side_data=rotation`
   returns nothing for its output (the rotation bug fires only on
   90/270 display-matrix tags — a camera artifact our renders should
   never carry, now checked rather than trusted).
3. **The §7 `video-sanity` gate is superseded by a sharper, per-request
   tripwire:** every video-leg request's `prompt_tokens` must exceed
   the token count of its own text-only prompt by ≥ 100 (decoded frames
   contribute vision tokens; a silently-dropped video contributes ~none
   — reference measurements: an 8 s 640×480 clip = 18,729 prompt tokens
   ingested vs 85 when silently dropped). A request failing the
   tripwire is an APPARATUS failure, excluded from scoring and counted;
   > 10% of a leg tripping VOIDs the leg as apparatus, never scored as
   model capability.
4. **Positive control:** before each video leg, one control clip with a
   pre-verified nonzero decoded-frame count is sent and its
   prompt_tokens checked against the tripwire — distinguishing a
   silent zero-frame regime from genuine model misses before 60
   requests are spent.
5. **Frame-count reality check for the pilot:** at the server's
   fps_target 4.0, the drop prefix (0.25 s) may sample only 1–2
   frames. The pilot records decoded-frame counts per prefix; if a
   prefix yields zero frames by sampling arithmetic (not decode
   failure), Amendment 1 may adjust encode fps or prefix span for that
   probe — clip *content* boundaries (pre-outcome) may not move.
   *(Clause 5's encode-fps contingency is superseded by Amendment B —
   encode fps is a dead lever; see below.)*

## Amendment B (2026-08-20, pre-pilot, zero data) — encode fps is a dead lever; uniform 4× time-stretch adopted, disclosed to the model

*Written before the clip renderer exists. Supersedes Amendment A
clause 5's contingency; the pre-outcome content boundaries remain
inviolable.*

**The dead lever, source-verified by the monitoring session:**
`fps_target` is hardcoded at 4.0 (`mtmd-helper.cpp:890`) with no server
flag, request field, or env override, and the decode pipeline applies
`-vf fps=<target>` (`:761`), which RESAMPLES regardless of source rate —
30 fps encodes decimate to 4, sub-4 encodes pad with duplicates.
Encode fps cannot change frames-per-real-second. Measured confirmation:
an 8.17 s clip decodes to 33 frames (duration × 4 within rounding), so
expected counts are computable in advance.

**The design decision (frozen):** all clips — video_pred, video_percep,
GSO variants, and the positive control — are time-stretched by a
uniform factor of **4×** (`-vf "setpts=4*PTS,fps=4"` before
`+faststart`). Expected sampled frames: drop prefix 0.25 s → 1.0 s
video-time → ~4; tilt 0.30 s → ~5; push 0.60 s → ~10; full episode
2.0 s → 8.0 s → ~32.

**Why stretch rather than accept 1–2-frame prefixes:** the pre-outcome
windows are physically capped (~0.35 s to drop contact) — real-time
prefixes cannot be lengthened without leaking outcomes, and a 1-frame
"video" is a degenerate stimulus carrying *less* than the 3-view stills
condition it is compared against, gutting H2's construct validity.

**Confound management (frozen):**
- The stretch factor is a single constant across ALL conditions and
  legs; it may not vary per probe or per leg.
- **The prompt tells the model the video is 4× slow motion**, so
  velocity-, momentum-, and contact-dynamics inferences can be
  renormalized rather than silently mismeasured. This is honest
  stimulus labeling, same policy as F11's simulator-context prompt.
- The server's timestamp markers are emitted on a 5000 ms interval in
  the video's own time base; under 4× stretch they land every 1.25 s of
  real time. Disclosed; identical across conditions, so no differential
  effect.
- Limits (§10) gains a line: the video legs measure physics-from-
  slow-motion-video with declared time base — a stimulus choice forced
  by the fixed 4 fps sampler, not a free parameter.

The pilot's per-prefix decoded-frame recording stays mandatory and now
also verifies the expected-count arithmetic above; a 1-frame prefix is
distinguished from a 0-frame tripwire failure (underpowered stimulus vs
apparatus fault), and only the recorded count separates them.

## Amendment C (2026-08-20, pre-pilot, zero data) — timestamp fabrication under sparse markers: prompt disclosure, scoring scope, confabulation co-report

*Written before any request. Based on the monitoring session's direct
observation (not inference): given a clip whose only timestamp markers
were 0.00 s and 5.00 s — confirmed in the model's own reasoning trace —
the model asserted events "at 0:04s", "at 0:05s", "at 0:08s": temporal
precision invented to match the question's granularity. Under our 4×
stretch, marker density (5000 ms of video time) means the full episode
gets TWO markers and every prediction prefix gets exactly ONE — the
model has essentially no temporal grid for a process whose structure is
temporal. This is an apparatus limitation that would masquerade as a
model failure, and the prompt_tokens tripwire cannot catch it (frames
present, token count normal).*

**Frozen handling:**

1. **Scoring scope unchanged and now explicit:** only the final JSON
   outcome fields are scored (`extract_prediction`), never the prose
   reasoning. Fabricated timestamps in reasoning cannot directly enter
   the score. The field schema is NOT re-worded toward ordinal phrasing
   — F11 comparability requires the identical fields — but see 3.
2. **Prompt disclosure extended** (same renormalization logic as the 4×
   line): the video-leg prompt states that frame timestamps are labeled
   only at 5-second intervals of video time, and instructs the model
   not to infer finer timing than the evidence supports — grounds to
   hedge rather than invent.
3. **`settle_time` is the exposed field** — the one scored quantity
   that IS a metric-time estimate, compounding fabricated precision
   with the 4× time-base conversion. Frozen: video-leg per-field
   co-reports break out `settle_time` explicitly, and any H2/H3
   reading that flips when `settle_time` is excluded from the
   aggregate is reported as FRAGILE (both aggregates shown). The
   frozen ratios themselves remain all-field (F11-lineage convention).
4. **Confabulation co-report (descriptive, no verdict):** responses in
   video legs are scanned for metric-time claims finer than the marker
   grid (pattern-matched in the reasoning text); the RATE of such
   claims is reported per leg as a measurement of confabulation under
   information scarcity — logged in a different column than accuracy,
   exactly as the monitoring session framed it.
5. **Reporting time base pinned (addendum, same day):** the slow-motion
   disclosure invites renormalization but left the ANSWER's time base
   ambiguous — a model could report `settle_time` in video seconds, in
   real seconds, or inconsistently across trials; the third mode
   injects an unrecoverable bimodal 4× error into a scored continuous
   field that no tripwire sees (plausible values, plausible range), and
   silently breaks F11 comparability. Frozen: the video-leg prompt
   states **"report every quantity, including settle_time, in
   real-world seconds — the simulator's time base — accounting for the
   4× slow motion"**, identical across legs (real-world base chosen
   because the F11 field spec and normalizers are already in it). The
   pilot additionally checks whether reported settle_time clusters near
   the real-world value or near 4× it — a direct mode detector, and a
   free measurement of whether the model performs the renormalization
   it is invited to perform.

## Amendment D (2026-08-20, pre-pilot) — pairwise frame fusion: even-frame rule; apparatus-validation event disclosed

**Apparatus event (disclosed):** with the user's authorization via the
monitoring session, 4 requests were spent against :8080 validating
apparatus on **train-split scene train_000000 only** (separate dir
`data/pm_big_clips_apparatus/`); no scene that will ever be scored has
produced a model response. Raw results:

    prompt_tokens: drop_pred(4f) 3108 · tilt_pred(5f) 3108 ·
                   push_pred(10f) 6186 · drop_percep(32f) 17528
    fit: 1026 tokens per FRAME PAIR exactly + 1056 fixed overhead,
         holding across an 8x range (32f predicted 17472 + longer text)

**The finding (strongly supported, not proven):** Qwen-VL fuses video
frames PAIRWISE (clip.cpp:5824); the cost unit is the pair, and a
5-frame clip costs identically to a 4-frame clip — consistent with
floor(n/2) pairs and **the orphan last frame being discarded**. Our
tilt_pred was the only odd clip, and its would-be-discarded frame is
the LAST — the frame nearest the outcome boundary, the most informative
one in a pre-outcome design. The renderer's frames are produced and
decoded correctly (duration × 16 arithmetic unaffected); the loss is
downstream in fusion.

**Frozen rule:** every clip must yield an EVEN sampled-frame count. The
renderer freeze-pads: when the sampled count would be odd, the final
source frame is duplicated just long enough to cross the next 4 fps
sample boundary — a visible pause, zero new dynamics, no outcome leak;
the pre-outcome content boundary does not move. Today this affects only
tilt_pred (5 → 6); the rule is general (GSO arm inherits it). Expected
sampled frames become: drop 4, **tilt 6**, push 10, percep 32.

**Pilot additions:** (1) decisive fusion confirmation — a 6-frame vs
7-frame variant pair; identical prompt_tokens confirms the discard
mechanism; (2) our own tripwire baseline supersedes the borrowed
reference: expected prompt_tokens ≈ 1056 + 1026 × (frames/2) + text
tokens, checked within ±10% per clip class.

**Addendum (same day) — the pad's semantic channel, examined and
closed:** the monitoring session flagged that a duplicated TERMINAL
sampled frame would read as "the object has stopped" — outcome-adjacent
evidence, differentially applied to odd-sampling clips only, at the
most heavily weighted position. Resolution: the concern targets the
sampled stream, but our pad operates PRE-SAMPLER, and under minimal
back-padding **the duplicated frame is never sampled at all** — it
extends duration so the 4 fps sampler emits one more GENUINE frame (the
very frame odd-parity fusion would have discarded). Proven arithmetically
for every source count in [4, 80] (`test_pad_duplicate_is_never_sampled`:
all sampled source indices < original count) and empirically on the
smoke clip (6 pairwise-distinct decoded frames; terminal ≠ penultimate).
The suggested front-pad would instead SHIFT the sampling grid and lose
the true boundary frame — the exact loss the even rule exists to
prevent. No sampled frame is ever synthetic; the semantic channel never
opens.

**Throughput datum for §9 budgeting (measured on our clips):** prompt
processing DEGRADES with length — 291.9 tok/s at 5k tokens → 246.9 at
17.5k (drop_percep, 71 s wall). Video legs are budgeted at ~250 tok/s
prompt processing, not the ~290 short-prompt figure.

**Mode-detector attempt (B), recorded:** one 300-token-capped response
on the drop_percep clip narrated frame-by-frame and truncated before
emitting settle_time — INCONCLUSIVE by design cap, mode key unopened.
Two observations worth keeping: the model anchored on the two real
markers (0.00 s / 5.00 s) and referred to contact ORDINALLY ("around
the 3rd or 4th frame") rather than fabricating metric times — the
opposite of the Amendment-C hazard, mildly encouraging; and the
truncation itself is the datum that video-leg completion budgets must
clear a narration preamble, folded into the pilot's max-tokens
calibration. The mode detector runs properly in the pilot (5 scenes,
adequate budget) rather than via further one-off requests.

## Amendment E (2026-08-20, pre-pilot) — budget starvation classified as apparatus; confab scan covers both channels

*Based on the monitoring session's measured three-response comparison:
with reasoning on, a starved completion budget can spend EVERYTHING in
`reasoning_content` and return `finish_reason=length` with a completely
EMPTY `content` field — zero characters, not truncated JSON. A naive
parser scores that as a model parse-failure; it is a budget failure. The
prompt_tokens tripwire cannot catch it (input side is normal); this is
the zero-frame trap arriving through the completion door.*

**Frozen classifier** (`f33_video.classify_response`), applied per
video-leg response:

    budget_starved  finish_reason=length AND content empty
                    -> APPARATUS, excluded from scoring, counted
    truncated       finish_reason=length AND content non-empty
                    -> counted toward the frozen truncation gate,
                       scored normally (parse may fail naturally)
    scoreable       otherwise

Both non-scoreable classes count toward the frozen §7 truncation gate
(> 10 %/leg → leg VOID, cap raised by amendment, cache makes the rerun
cheap) — no frozen bar changes. `completion_tokens` and
`reasoning_content` length are recorded per response so the pilot
calibrates the cap from measurement, not inference.

**Confab scan scope corrected:** with reasoning on, thinking lands in
`reasoning_content` — a SEPARATE field the content-scoped scanner never
saw (the observed "0:04s" fabrication lived there). The scan now covers
BOTH channels, reported separately (`confab_content_rate`,
`confab_reasoning_rate`); scoring safety was never at risk (JSON fields
only), but the descriptive measurement now measures what it claims to.

**Runtime facts recorded (from the monitoring session, server verified
coherent):** post-restart n_ctx is 65536 (down from 262144; ~4× our
largest expected video prompt — prompt truncation, not an error, is the
failure mode if ever approached), RSS 24.4 GB, build b10501-eab19b649
verified three ways with zero uncommitted tracked files.

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
