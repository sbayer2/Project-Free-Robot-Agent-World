"""F29 pilot driver: pick the extrapolation aggression alpha*
(docs/HELD_OUT_TRANSFER.md section 3).

Two-phase design so the geometry unit-tests in any session:

  * PURE-PYTHON at module top -- drop_ratio, combined_drop, pick_alpha,
    within_acceptable_band, pilot_summary -- exactly the section-3.6 rule
    frozen in the preregistration. tests/test_f29_pilot.py pins the
    constants against the prereg document.
  * MLX MEASUREMENT LAYER (lazy, Mac-only): encode + head-only behavior +
    gain on the three pilot datasets at loud. Reuses F28's load_model and
    scripts.probe_appearance.encode_z verbatim, so no new autodiff.

Frozen alpha* rule (prereg section 3.6):

    alpha* = argmin over tested alphas of |combined_drop(alpha) - 0.5|

where combined_drop(alpha) is the mean of joint's and disjoint's drop_ratios
at loud on the alpha-tail pilot dataset. If combined_drop(alpha*) falls
outside [0.25, 0.75], Amendment 1 discloses the miss explicitly.

The MLX layer is intentionally NOT WRITTEN in this commit: it depends on
the 24 disjoint checkpoints landing (runs/f29_disjoint/) and on the pilot
datasets existing (data/pm_f29_pilot_a*). Follow-up commit wires those.
"""

from __future__ import annotations

# Frozen from the preregistration (docs/HELD_OUT_TRANSFER.md sections 3, 4).
PILOT_ALPHAS: tuple[float, ...] = (0.05, 0.20, 0.40)  # section 3.1 aggressions
TARGET_DROP: float = 0.5                              # section 3.6 target
ACCEPTABLE_BAND: tuple[float, float] = (0.25, 0.75)   # section 3.6 disclosure trigger
PILOT_ARM: str = "loud"                               # section 3.2/3.3: measured on loud
N_SCENES_PER_ARM: int = 128                           # section 3.1

# Paths (all conventions frozen; MLX layer will consume them verbatim).
JOINT_CHECKPOINTS = "runs/f27b/{arm}_s{seed}/model.safetensors"  # seed in 0,1,2
DISJOINT_BEHAV = "runs/f29_disjoint/{arm}_pair{k}/behav/model.safetensors"  # k in 0,1,2
PILOT_DATA = "data/pm_f29_pilot_a{tag}"  # tag in "5", "20", "40" (percent, no leading zero)


def alpha_tag(alpha: float) -> str:
    """Percent-encoded aggression tag: 0.05 -> '5', 0.20 -> '20', 0.40 -> '40'.
    Pins the dataset naming convention alongside PILOT_DATA."""
    pct = round(alpha * 100)
    if abs(pct - alpha * 100) > 1e-9 or pct <= 0 or pct >= 100:
        raise ValueError(f"alpha must be a strict-percent fraction, got {alpha}")
    return str(pct)


def drop_ratio(train_gain: float, held_out_gain: float) -> float:
    """One student's fractional gain drop: 1 - held_out / train.

    0 if the student didn't drop at all; 1 if collapsed to predict-the-mean.
    Can go negative if held-out somehow beats training (implausible for a
    valid holdout; recorded but not clamped). NaN if train_gain is nonpositive
    (student never learned; drop is undefined)."""
    if train_gain <= 0:
        return float("nan")
    return 1.0 - held_out_gain / train_gain


def combined_drop(joint_train: float, joint_held: float,
                  disjoint_train: float, disjoint_held: float) -> float:
    """Mean of the two students' drop_ratios (prereg section 3.5).

    NaN if either student's train_gain is nonpositive (kill the alpha, not
    the pilot -- the retrain-sanity gate in section 7 should have caught a
    truly bad disjoint pair upstream)."""
    j = drop_ratio(joint_train, joint_held)
    d = drop_ratio(disjoint_train, disjoint_held)
    if j != j or d != d:  # NaN check without importing math
        return float("nan")
    return (j + d) / 2.0


def pick_alpha(alphas, combined_drops) -> float:
    """alpha* = argmin |combined_drop - 0.5| (prereg section 3.6).

    Ties broken by SMALLEST alpha: when two aggressions are equally close
    to the target, prefer the less-aggressive one -- less extrapolation
    means more measurable transfer, honesty-conservative. Rejects NaN
    combined_drops so an unmeasurable alpha cannot be picked."""
    if len(alphas) != len(combined_drops):
        raise ValueError("alphas and combined_drops must match length")
    live = [(i, a, c) for i, (a, c) in enumerate(zip(alphas, combined_drops,
                                                     strict=True)) if c == c]
    if not live:
        raise ValueError("all combined_drops are NaN -- pilot cannot pick alpha*")
    best = min(live, key=lambda t: (abs(t[2] - TARGET_DROP), t[1]))
    return best[1]


def within_acceptable_band(combined_drop_value: float) -> bool:
    """True iff combined_drop lands inside [0.25, 0.75] (prereg section 3.6).
    A False triggers explicit disclosure in Amendment 1; the choice of
    alpha* is not overridden by the band, only annotated."""
    lo, hi = ACCEPTABLE_BAND
    return lo <= combined_drop_value <= hi


def pilot_summary(alphas, per_alpha: list[dict]) -> dict:
    """Assemble the pilot's Amendment-1-ready report from per-alpha
    measurement dicts (keys: joint_train, joint_held, disjoint_train,
    disjoint_held, joint_r_held, disjoint_r_held, extrapolation_frac).

    Returns the dict pinned into Amendment 1's frontmatter and included
    in runs/f29/f29_report.json's meta.amendment_1 field. Schema is
    pinned by tests/test_f29_pilot.py."""
    if len(alphas) != len(per_alpha):
        raise ValueError("alphas and per_alpha must match length")
    rows = []
    combined = []
    for a, m in zip(alphas, per_alpha, strict=True):
        c = combined_drop(m["joint_train"], m["joint_held"],
                          m["disjoint_train"], m["disjoint_held"])
        combined.append(c)
        rows.append({
            "alpha": a,
            "joint_train": m["joint_train"], "joint_held": m["joint_held"],
            "disjoint_train": m["disjoint_train"], "disjoint_held": m["disjoint_held"],
            "joint_drop": drop_ratio(m["joint_train"], m["joint_held"]),
            "disjoint_drop": drop_ratio(m["disjoint_train"], m["disjoint_held"]),
            "combined_drop": c,
            **{k: v for k, v in m.items()
               if k not in ("joint_train", "joint_held",
                            "disjoint_train", "disjoint_held")},
        })
    star = pick_alpha(alphas, combined)
    star_i = list(alphas).index(star)
    return {
        "target_drop": TARGET_DROP,
        "acceptable_band": list(ACCEPTABLE_BAND),
        "pilot_arm": PILOT_ARM,
        "alphas_tested": list(alphas),
        "per_alpha": rows,
        "alpha_star": star,
        "combined_drop_at_alpha_star": combined[star_i],
        "within_acceptable_band": within_acceptable_band(combined[star_i]),
    }
