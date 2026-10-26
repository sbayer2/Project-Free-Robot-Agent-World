"""What the 0.349 deg ramp did to the behavior target: the F18 oracle and the
per-axis variance shares, before vs after the tilt fix, on the same scenes.

scripts/regen_tilt_outcomes.py rewrote pm_big's tilt outcomes in place and
kept the old ones in its audit file (tilt_regen_audit.json). This rebuilds
the pre-fix behavior targets from that file, checks that drop and push are
unchanged, and reports for both versions:
  - F18's oracle arms (scripts/oracle_ceiling.py, zero training) on the
    corner split and an iid split of the train region;
  - the share of total behavior-target variance explained by each essence
    axis alone (cubic fit per field) and by shape, plus the tilt block's share.
The pre-fix corner run reproduces F18's published numbers exactly
(shape 1.331, shape+appearance 2.311, shape+essence 3.243).

    python scripts/tilt_bug_oracle.py --data data/pm_big
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import oracle_ceiling as O  # noqa: E402

from pseudomarble.probes import behavior_field_names, behavior_vector  # noqa: E402


def prefix_targets(data: str, audit: dict) -> np.ndarray:
    """Behavior targets with each scene's tilt outcome swapped back to the
    audit's pre-fix values, in oracle_ceiling.load's scene order."""
    rows = []
    for f in sorted(glob.glob(os.path.join(data, "*", "sample.json"))):
        d = json.load(open(f))
        old = audit[os.path.basename(os.path.dirname(f))]["old"]
        probes = [{**p, "outcome": {**p["outcome"], **old}} if p["probe"] == "tilt" else p
                  for p in d["behavior"]["probes"]]
        rows.append(behavior_vector(probes, normalize=True))
    return np.array(rows)


def explained_share(Y: np.ndarray, X: np.ndarray) -> float:
    """Share of total target variance (summed over fields) explained by a
    least-squares fit on X."""
    res = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(1.0 - res.var(0).sum() / Y.var(0).sum())


def variance_shares(Y: np.ndarray, S: np.ndarray, E: np.ndarray,
                    axes, tilt_cols) -> dict[str, float]:
    out = {a: explained_share(Y, np.stack([E[:, j] ** p for p in range(4)], 1))
           for j, a in enumerate(axes)}
    out["shape"] = explained_share(Y, S)
    out["tilt_block"] = float(Y[:, tilt_cols].var(0).sum() / Y.var(0).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pm_big")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    audit = json.load(open(os.path.join(args.data, "tilt_regen_audit.json")))["scenes"]
    S, E, A, Ynew, _, splits = O.load(args.data)
    Yold = prefix_targets(args.data, audit)
    fields = behavior_field_names()
    tilt = [i for i, n in enumerate(fields) if n.startswith("tilt.")]
    other = [i for i in range(len(fields)) if i not in tilt]
    assert np.allclose(Yold[:, other], Ynew[:, other]), "drop/push changed: not a tilt-only regen"

    idx = np.where(splits == "train")[0]
    perm = np.random.default_rng(args.seed).permutation(idx)
    cut = int(0.8 * len(perm))
    iid_tr, iid_te = np.zeros(len(Ynew), bool), np.zeros(len(Ynew), bool)
    iid_tr[perm[:cut]], iid_te[perm[cut:]] = True, True

    res = {}
    for tag, Y in (("pre_fix", Yold), ("post_fix", Ynew)):
        res[tag] = {
            "corner": O.report(f"{tag}, corner split", S, E, A, Y,
                               splits == "train", splits == "test", fields),
            "iid": O.report(f"{tag}, iid split", S, E, A, Y, iid_tr, iid_te, fields),
            "variance_shares": variance_shares(Y, S, E, O.ESSENCE_AXES, tilt),
        }
        print(f"\n[tilt-bug] {tag} variance shares: " + "  ".join(
            f"{k} {v:.2f}" for k, v in res[tag]["variance_shares"].items()))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
