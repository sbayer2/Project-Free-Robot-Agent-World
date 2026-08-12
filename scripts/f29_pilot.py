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

The MLX layer measures held_out_gain for both students at loud on each
pilot dataset (prereg section 3.2-3.3), plus the disjoint train-world
gains the drop_ratio needs (section 3.4; the joint train gains already
exist per-seed in runs/f27b/f27_report.json). Output feeds pilot_summary
verbatim; the resulting dict is what Amendment 1 pins and what the main
run embeds as meta.amendment_1 (the frozen verdicts' pilot gate).

Usage (Mac only):
    .venv/bin/python scripts/f29_pilot.py --dry-run   # list work, check inputs
    .venv/bin/python scripts/f29_pilot.py             # measure + write report
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
# One dataset dir per (alpha, arm) so PseudoMarbleDataset's one-dir-per-arm
# convention holds. The per-arm suffix refines prereg section 4's family name
# "pm_f29_pilot_a{5,20,40}" (a layout decision, recorded in Amendment 1 so it
# cannot read as silent prereg drift). tag is percent, no leading zero.
PILOT_DATA = "data/pm_f29_pilot_a{tag}_{arm}"


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


def r_from_records(records: list) -> float:
    """Coupling r of a dataset: corrcoef(appearance roughness, physics
    friction) over its scene records -- scripts/f27_measure.py's apparatus
    convention, restated here so the pilot report and the F27b report
    measure r identically. Pure (numpy only), unit-tested."""
    import numpy as np
    rough = [s["material_truth"]["appearance_params"]["roughness"] for s in records]
    fric = [s["physics"]["raw"]["friction"] for s in records]
    return float(np.corrcoef(rough, fric)[0, 1])


# ---- MLX measurement layer (Mac only; every heavy import is lazy) ---------- #

def _measure_models_on(imgs, Yb_train, Yb_eval, ckpts: dict,
                       eval_idx=None) -> dict:
    """{name: {"gain": float, "pr": float}} for each checkpoint, encoding
    ``imgs`` once per model and scoring gain against the train-mean
    baseline of ``Yb_train`` (prereg section 2's formula).

    ``eval_idx`` subsets the predictions to the scenes ``Yb_eval`` covers
    (the train-gain phase encodes the full training world but scores only
    its test split); None means every encoded scene is scored (the pilot
    datasets, which are all-eval). PR is computed on the full encoded z
    either way, matching the f27_measure convention."""
    from f28_measure import load_model
    from oracle_ceiling import gain as gain_ratio
    from probe_appearance import encode_z
    from render_fidelity_eval import behavior_from_z

    from pseudomarble.models.alignment import participation_ratio

    out = {}
    for name, ck in ckpts.items():
        m = load_model(ck)
        z = encode_z(m, imgs)
        pred = behavior_from_z(m, z)
        pred_eval = pred if eval_idx is None else pred[eval_idx]
        out[name] = {"gain": float(gain_ratio(Yb_train, Yb_eval, pred_eval)),
                     "pr": float(participation_ratio(z))}
        del m
    return out


def main() -> None:
    import argparse
    import json
    import os

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default=None,
                    help="comma-separated subset of the frozen alphas")
    ap.add_argument("--out", default="runs/f29/pilot_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    alphas = ([float(a) for a in args.alphas.split(",")] if args.alphas
              else list(PILOT_ALPHAS))
    for a in alphas:
        if a not in PILOT_ALPHAS:
            raise SystemExit(f"alpha {a} is not in the frozen set {PILOT_ALPHAS}")

    joint_cks = {f"s{k}": JOINT_CHECKPOINTS.format(arm=PILOT_ARM, seed=k)
                 for k in range(3)}
    disjoint_cks = {f"pair{k}": DISJOINT_BEHAV.format(arm=PILOT_ARM, k=k)
                    for k in range(3)}
    pilot_dirs = {a: PILOT_DATA.format(tag=alpha_tag(a), arm=PILOT_ARM)
                  for a in alphas}
    missing = [p for p in (list(joint_cks.values()) + list(disjoint_cks.values()))
               if not os.path.exists(p)]
    missing += [d for d in pilot_dirs.values()
                if not os.path.exists(os.path.join(d, "manifest.json"))]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"missing inputs (run the retrain/generator first):\n  {listing}")
    if args.dry_run:
        print(f"would measure {len(joint_cks)} joint + {len(disjoint_cks)} disjoint "
              f"models on {len(pilot_dirs)} pilot sets at {PILOT_ARM}: "
              + ", ".join(pilot_dirs.values()))
        return

    import mlx.core as mx
    from f29_generate_pilot import build_hull, load_train_physics, normalize_phys
    from render_fidelity_eval import load_arrays

    with open("runs/f27b/f27_report.json") as f:
        f27 = json.load(f)
    joint_train_per_seed = f27["arms"][PILOT_ARM]["gain_per_seed"]
    r_train = float(f27["apparatus"]["r"][PILOT_ARM])

    # Phase MLX-1: disjoint train-world gains on the f27 loud training world
    # (section 3.4's "measured during the retrain-sanity check").
    print(f"[pilot] measuring disjoint train gains on data/pm_f27_{PILOT_ARM}")
    imgs, Yb, _Ya, tr, te, _ds = load_arrays(f"data/pm_f27_{PILOT_ARM}")
    disjoint_train = _measure_models_on(imgs, Yb[tr], Yb[te],
                                        {k: v for k, v in disjoint_cks.items()},
                                        eval_idx=te)
    Yb_train = Yb[tr]
    del imgs
    mx.clear_cache()
    dt_gains = [disjoint_train[k]["gain"] for k in sorted(disjoint_train)]
    print("[pilot] disjoint train gains: "
          + ", ".join(f"{g:.3f}" for g in dt_gains)
          + "  (joint per-seed from f27b: "
          + ", ".join(f"{g:.3f}" for g in joint_train_per_seed) + ")")

    train_phys = load_train_physics("ctrl")
    hull = build_hull(train_phys)

    per_alpha, detail = [], {}
    for a in alphas:
        d = pilot_dirs[a]
        imgs_H, Yb_H, _Ya_H, _trH, _teH, ds_H = load_arrays(d)
        records = [s.record for s in ds_H.scenes]
        held = _measure_models_on(imgs_H, Yb_train, Yb_H,
                                  {**joint_cks, **disjoint_cks})
        del imgs_H
        mx.clear_cache()
        phys = [(s["physics"]["raw"]["density"], s["physics"]["raw"]["friction"],
                 s["physics"]["raw"]["restitution"]) for s in records]
        ext = float(np.mean([hull.find_simplex([normalize_phys(p)])[0] < 0
                             for p in phys]))
        j_held = [held[f"s{k}"]["gain"] for k in range(3)]
        d_held = [held[f"pair{k}"]["gain"] for k in range(3)]
        per_alpha.append({
            "joint_train": float(np.mean(joint_train_per_seed)),
            "joint_held": float(np.mean(j_held)),
            "disjoint_train": float(np.mean(dt_gains)),
            "disjoint_held": float(np.mean(d_held)),
            "joint_held_per_seed": j_held,
            "disjoint_held_per_pair": d_held,
            "r_held": r_from_records(records),
            "r_train": r_train,
            "extrapolation_frac": ext,
        })
        detail[f"a{alpha_tag(a)}"] = held
        print(f"[pilot] a={a:.2f}: joint held "
              + "/".join(f"{g:.2f}" for g in j_held)
              + "  disjoint held " + "/".join(f"{g:.2f}" for g in d_held)
              + f"  r_held {per_alpha[-1]['r_held']:+.3f}  outside-hull {ext:.0%}")

    summary = pilot_summary(alphas, per_alpha)
    report = {"summary": summary,
              "detail": {"models_held": detail,
                         "disjoint_train": disjoint_train,
                         "joint_train_per_seed": joint_train_per_seed}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    band = "OK" if summary["within_acceptable_band"] else \
        "MISSED -- Amendment 1 must disclose"
    print(f"\n[pilot] alpha* = {summary['alpha_star']} "
          f"(combined_drop {summary['combined_drop_at_alpha_star']:.3f}, band {band})")
    print(f"wrote {args.out} -- next: Amendment 1 pins alpha* from this file")


if __name__ == "__main__":
    main()
