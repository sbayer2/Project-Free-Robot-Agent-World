#!/usr/bin/env python
"""F33 frozen decision rules (docs/VIDEO_BASELINE.md sections 6-7).

Reads runs/f33/f33_report.json and emits the preregistered verdicts.
Committed in the SAME commit as the preregistration, before any Qwen3.8
response existed on any pseudo-marble scene.

    H1  BETTER / COMPARABLE / WORSE            (essence vs F11 AgentWorld essence)
    H2  VIDEO-ADDS / VIDEO-HURTS / NULL / INCONCLUSIVE  (video_pred vs vision stills)
    H3  READABLE / UNREADABLE                  (perception ceiling; gates H2's NULL)
    H4  descriptive co-report, no verdict      (specialist/baseline/GSO/consistency)

Usage:  .venv/bin/python scripts/f33_verdicts.py [--report runs/f33/f33_report.json]
"""
from __future__ import annotations

import argparse
import json

# Frozen constants (do not edit after the freeze commit).
H1_BETTER, H1_WORSE = 0.75, 1.25          # MSE ratio vs F11 essence
H2_ADDS, H2_HURTS = 0.75, 1.33            # video_pred / vision MSE ratio
H2_NULL_LO, H2_NULL_HI = 0.90, 1.11
H3_READABLE = 0.90                        # video_percep / video_pred MSE ratio
PARSE_MIN = 0.90                          # per-leg scorable-response floor
TRUNC_MAX = 0.10                          # per-leg cap-hit ceiling
PRIMARY_LEGS = ("essence", "vision", "video_pred", "video_percep")

# F11 anchors, frozen from docs/FINDINGS.md F11 (pm_big test split).
F11 = {
    "essence": {"mse": 1.277, "gain": 0.04, "brier": 0.100, "fields": "8/21"},
    "appearance": {"mse": 0.789, "gain": 0.07, "brier": 0.150, "fields": "4/21"},
    "baseline_mse": 0.052,
    "specialist_gain": "1.36-1.64x",
}
P4_BRIER_BAR = 0.15  # P4: Brier >= this = NOT matching AgentWorld's 0.100 judge


def leg_ok(leg: dict) -> tuple[bool, list[str]]:
    notes = []
    n = leg.get("n_requests", 0)
    if n <= 0:
        return False, ["no requests recorded"]
    parse_rate = leg.get("n_parsed", 0) / n
    trunc_rate = leg.get("n_truncated", 0) / n
    if parse_rate < PARSE_MIN:
        notes.append(f"parse rate {parse_rate:.0%} < {PARSE_MIN:.0%}")
    if trunc_rate > TRUNC_MAX:
        notes.append(f"truncation rate {trunc_rate:.0%} > {TRUNC_MAX:.0%}")
    return not notes, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/f33/f33_report.json")
    args = ap.parse_args()
    with open(args.report) as f:
        rep = json.load(f)
    legs, meta = rep["legs"], rep.get("meta", {})

    # ---- gates (section 7) ------------------------------------------------ #
    gates_ok, void = True, set()
    if not meta.get("amendment_1"):
        print("  !! PILOT: Amendment 1 not recorded in meta -> gates FAIL")
        gates_ok = False
    if not meta.get("provenance", {}).get("no_restart_confirmed"):
        print("  !! PROVENANCE: user's no-restart confirmation missing -> gates FAIL")
        gates_ok = False
    if not meta.get("video_sanity_frames_sampled"):
        print("  !! VIDEO-SANITY: sampled-frames evidence missing -> video legs VOID")
        void.update(("video_pred", "video_percep"))
    for name in PRIMARY_LEGS:
        leg = legs.get(name)
        if leg is None:
            print(f"  !! MISSING leg {name}")
            gates_ok = False
            continue
        ok, notes = leg_ok(leg)
        if not ok:
            for n in notes:
                print(f"  !! {name}: {n}")
            void.add(name)
    print(f"GATES: {'ALL PASS' if gates_ok and not void else 'FAILED/PARTIAL -- see notes'}")
    if not gates_ok:
        print("verdicts withheld: gates failed (section 7); leg table withheld with them.")
        return

    # ---- table (prints only past the hard gates; voided legs marked) ------ #
    print(f"\n{'leg':14s} {'MSE':>8s} {'gain':>7s} {'brier':>7s} {'fields':>7s} {'parse':>6s}")
    for name in PRIMARY_LEGS + tuple(k for k in legs if k.startswith("gso_")):
        leg = legs.get(name)
        if leg is None:
            continue
        mark = " VOID" if name in void else ""
        print(f"{name:14s} {leg['mse']:8.3f} {leg['gain']:6.2f}x "
              f"{leg.get('brier', float('nan')):7.3f} "
              f"{leg.get('fields_beating', '?'):>7} "
              f"{leg.get('n_parsed', 0)}/{leg.get('n_requests', 0)}{mark}")
    print(f"\nF11 anchors: essence MSE {F11['essence']['mse']} gain {F11['essence']['gain']}x "
          f"brier {F11['essence']['brier']} ({F11['essence']['fields']}); "
          f"baseline {F11['baseline_mse']}; specialist {F11['specialist_gain']}")
    if meta.get("mme"):
        print(f"MME (context, no verdict): {meta['mme']}")

    # ---- H1 --------------------------------------------------------------- #
    print()
    if "essence" in void:
        print("H1: essence leg VOID -> withheld")
    else:
        r1 = legs["essence"]["mse"] / F11["essence"]["mse"]
        v1 = "BETTER" if r1 <= H1_BETTER else ("WORSE" if r1 > H1_WORSE else "COMPARABLE")
        print(f"H1: MSE ratio vs F11 essence = {r1:.3f} -> {v1}")
        b = legs["essence"].get("brier")
        if b is not None:
            p4 = "hedging (P4 lean holds)" if b >= P4_BRIER_BAR else \
                 "matches/beats AgentWorld's calibrated judge (P4 lean falsified)"
            print(f"    Brier {b:.3f} vs F11 0.100 -> {p4}")

    # ---- H3 first (it gates H2's NULL) ------------------------------------ #
    h3 = None
    if void & {"video_pred", "video_percep"}:
        print("H3: video leg(s) VOID -> withheld; H2 withheld")
    else:
        r3 = legs["video_percep"]["mse"] / legs["video_pred"]["mse"]
        h3 = "READABLE" if r3 <= H3_READABLE else "UNREADABLE"
        print(f"H3: percep/pred MSE ratio = {r3:.3f} -> {h3}")

    # ---- H2 --------------------------------------------------------------- #
    if h3 is not None and "vision" not in void:
        r2 = legs["video_pred"]["mse"] / legs["vision"]["mse"]
        if r2 <= H2_ADDS:
            v2 = "VIDEO-ADDS"
        elif r2 >= H2_HURTS:
            v2 = "VIDEO-HURTS"
        elif H2_NULL_LO <= r2 <= H2_NULL_HI:
            v2 = "NULL (unreadable -- H3 UNREADABLE)" if h3 != "READABLE" else "NULL"
        else:
            v2 = "INCONCLUSIVE"
        print(f"H2: video_pred/vision MSE ratio = {r2:.3f} -> {v2}")
        if v2 == "VIDEO-ADDS":
            print("    frozen reading: video INPUT carries physics content stills "
                  "don't -- the input-side counterpart of Dyna-2's training-side "
                  "claim, measured locally against an exact answer key.")
        elif v2 == "NULL":
            print("    frozen reading: watching the probe begin buys nothing over "
                  "looking at the object -- consistent with F11's vision ~= "
                  "appearance-text equivalence, one modality further up.")

    # ---- H4 (descriptive) ------------------------------------------------- #
    print("\nH4 (descriptive): specialist gain "
          f"{F11['specialist_gain']}, train-mean baseline MSE {F11['baseline_mse']}.")
    gso = {k: v for k, v in legs.items() if k.startswith("gso_")}
    if gso:
        for k, v in gso.items():
            base = legs.get(k.replace("gso_", ""))
            if base and base["mse"] > 0:
                print(f"    {k}: MSE {v['mse']:.3f} "
                      f"({v['mse']/base['mse']:.2f}x its synthetic leg)")
    cons = rep.get("consistency")
    if cons:
        print(f"    cross-condition consistency co-report: {cons}")


if __name__ == "__main__":
    main()
