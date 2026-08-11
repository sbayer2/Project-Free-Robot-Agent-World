#!/usr/bin/env python
"""F28 measurement driver: directional (content-matched) coherence
(docs/DIRECTIONAL_COHERENCE.md, frozen 2026-08-04).

Per arm (ctrl/base/g2/loud; runs/f27b checkpoints on data/pm_f27_*):

  * ridge-probe directions for the committed render channels (metallic,
    roughness, value = max RGB) fit on the train-split z,
  * essence-head Jacobian rows (density, friction, restitution) at each
    test-split z via MLX vjp,
  * both expressed in that model's own train-split standardized coordinates
    (ridge weights are already standardized-X; raw-z Jacobians are multiplied
    elementwise by the train-split per-dim std),
  * alignment matrix M[c,p] = mean signed cosine over test scenes;
    D = mean of the three matched cells with the committed signs; the six
    mismatched cells are co-reported as mean |M| (diagnostic, not part of D),
  * learned_D = D_trained - mean(D_untrained) over 3 untrained builds
    (mx.random.seed(1000+i), ModelConfig(image_size=128)), per arm on that
    arm's data,
  * secondary behavior endpoint: same construction on the behavior head at
    the committed fields (addressed by name; a missing name fails loudly),
  * disjoint stitched reference: renderonly probes -> linear stitch
    (behavonly z -> renderonly z, same ridge) -> behavonly essence-head
    Jacobians.

Writes runs/f28/f28_report.json. NO verdicts here; the frozen rules live in
scripts/f28_verdicts.py (committed with the preregistration, one commit before
this driver) and this driver may not alter them.

Gate bookkeeping decisions (documented driver choices, not new rules): the
arm-level trained-probe gate covers the three shared seeds (the models the
primary statistic reads); a renderonly model failing the same floor voids its
own cells by emitting ``disjoint: null``, which scripts/f28_verdicts.py
already renders VOID -- the arm verdict is not withheld for a failed
reference cell. The collapse and Jacobian gates take the min over every model
whose numbers enter the report (shared seeds + renderonly + behavonly for PR;
shared seeds + behavonly + untrained builds for Jacobians). The untrained
builds are re-encoded per arm (the procedure in prereg section 2 -- their z
depends on the arm's images even though the weights do not), so the encode
count is ~32, not the estimate's 23; still comfortably inside the 10-20 min
budget.

Mac only (MLX). Usage:
    .venv/bin/python scripts/f28_measure.py            # full run
    .venv/bin/python scripts/f28_measure.py --dry-run  # list work, touch nothing

The geometry (ridge weights, standardized-coordinate composition, cosines,
stitch transfer, report schema) is pure numpy at module top so
tests/test_f28_measure.py can verify it in any session, per the
preregistration's section 2.3 requirement.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from probe_appearance import encode_z, kfold_r2  # noqa: E402
from render_fidelity_eval import load_arrays  # noqa: E402

from pseudomarble.models.alignment import participation_ratio  # noqa: E402
from pseudomarble.probes import behavior_field_names  # noqa: E402

# ---- committed constants: the section-3 pairing table, verbatim ------------- #
# (channel order, physics order, and signs are FROZEN in the preregistration;
# tests/test_f28_measure.py pins them. Matched cells are the diagonal.)
CHANNELS = ("metallic", "roughness", "value")
ESSENCE_AXES = ("density", "friction", "restitution")  # F19 order = head outputs 0,1,2
SIGNS_ESSENCE = (1.0, 1.0, -1.0)
BEHAVIOR_FIELDS = ("push.path_length", "push.slid_distance", "drop.n_bounces")
SIGNS_BEHAVIOR = (-1.0, -1.0, -1.0)
RIDGE_ALPHA = 1e-3
PROBE_R2_TRAINED = 0.25  # mirror of the frozen gate; used here only to VOID disjoint cells
N_UNTRAINED = 3
UNTRAINED_SEED_BASE = 1000

ARMS = [("ctrl", "data/pm_f27_ctrl"), ("base", "data/pm_f27_base"),
        ("g2", "data/pm_f27_g2"), ("loud", "data/pm_f27_loud")]


# ---- pure-numpy geometry (unit-tested in any session) ----------------------- #

def channel_targets(Ya: np.ndarray) -> np.ndarray:
    """(n, 3) committed probe targets from the 8-dim appearance vector:
    metallic (col 5), roughness (col 4), value = max(R, G, B) (cols 0:3)."""
    return np.stack([Ya[:, 5], Ya[:, 4], Ya[:, 0:3].max(axis=1)], axis=1)


def ridge_weights(Xtr: np.ndarray, Ytr: np.ndarray,
                  alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The exact fit inside oracle_ceiling.ridge_fit_predict, returning the
    solved weights instead of predictions.

    Returns (W, mu, sd): W is (d+1, k) in standardized-X coordinates with the
    intercept as the LAST row; predictions are [(X-mu)/sd, 1] @ W. Kept in
    lockstep with ridge_fit_predict by tests/test_f28_measure.py."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], axis=1)
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0.0  # never penalize the intercept
    W = np.linalg.solve(A.T @ A + reg, A.T @ Ytr)
    return W, mu, sd


def unit_rows(V: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.where(n < 1e-12, 1.0, n)


def probe_directions(Ztr: np.ndarray, Ytr: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit render-side directions in standardized-z coordinates.

    Returns (dirs (k, L), mu (L,), sd (L,)); one direction per target column,
    the intercept row excluded before normalization (prereg section 2.1)."""
    W, mu, sd = ridge_weights(Ztr, Ytr)
    return unit_rows(W[:-1].T), mu, sd


def alignment_matrix(dirs: np.ndarray, jacs_raw: np.ndarray,
                     sd: np.ndarray) -> np.ndarray:
    """M[c, p] = mean over scenes of cos(d_c, J_p) in standardized coordinates.

    ``dirs`` (C, L): unit directions already in standardized-z coordinates.
    ``jacs_raw`` (P, B, L): raw-z Jacobian rows; converted here by elementwise
    multiplication with the train-split per-dim std (prereg section 2.3)."""
    M = np.zeros((dirs.shape[0], jacs_raw.shape[0]))
    for p in range(jacs_raw.shape[0]):
        js = unit_rows(jacs_raw[p] * sd[None, :])  # (B, L)
        M[:, p] = (dirs @ js.T).mean(axis=1)
    return M


def directional_D(M: np.ndarray, signs: tuple[float, ...]) -> tuple[float, float]:
    """(D, mismatched_abs): signed mean of the matched (diagonal) cells, and
    mean |M| over the off-diagonal cells (prereg section 2.5)."""
    k = len(signs)
    matched = float(np.mean([signs[i] * M[i, i] for i in range(k)]))
    off = ~np.eye(k, dtype=bool)
    return matched, float(np.abs(M[off]).mean())


def stitch_predict(W: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                   X: np.ndarray) -> np.ndarray:
    A = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1)
    return A @ W


def variance_weighted_r2(Y: np.ndarray, P: np.ndarray) -> float:
    """Variance-weighted R^2 over all columns (prereg section 5.3's stitch gate)."""
    ss_res = float(((Y - P) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def transfer_directions(dirs_ro: np.ndarray, sd_ro: np.ndarray,
                        W_stitch: np.ndarray) -> np.ndarray:
    """Compose probe-on-z_ro with the stitch z_bo -> z_ro.

    The transferred direction is the gradient of
    ``d_c . (stitch(z_bo) - mu_ro) / sd_ro`` w.r.t. standardized z_bo:
    ``W_stitch[:-1] @ (d_c / sd_ro)``, unit-normalized. Verified against the
    analytic composition in tests/test_f28_measure.py."""
    return unit_rows((W_stitch[:-1] @ (dirs_ro / sd_ro[None, :]).T).T)


def arm_record(r: float, gates: dict, learned_D: list[float],
               behavior_learned_D: list[float], mismatched_abs_mean: float,
               disjoint: dict | None, extra: dict | None = None) -> dict:
    """The per-arm schema scripts/f28_verdicts.py reads (pinned by test)."""
    rec = {"r": float(r), "gates": gates,
           "learned_D_per_seed": [float(x) for x in learned_D],
           "behavior_learned_D_per_seed": [float(x) for x in behavior_learned_D],
           "mismatched_abs_mean": float(mismatched_abs_mean),
           "disjoint": disjoint}
    if extra:
        rec.update(extra)
    return rec


# ---- MLX layer (Mac only) --------------------------------------------------- #

def load_model(ck: str):
    """Build from the checkpoint's own recorded config and load weights."""
    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model
    with open(os.path.join(os.path.dirname(ck), "metrics.json")) as f:
        cfg_d = json.load(f)["config"]
    cfg_d["conv_channels"] = tuple(cfg_d["conv_channels"])
    known = ModelConfig.__dataclass_fields__.keys()
    m = build_model(ModelConfig(**{k: v for k, v in cfg_d.items() if k in known}))
    m.load_weights(ck)
    return m


def head_jacobian_rows(head_fn, z_np: np.ndarray, out_idx: tuple[int, ...],
                       out_dim: int) -> np.ndarray:
    """(len(out_idx), B, L) raw-z Jacobian rows of a per-row z -> R^out_dim
    head, one batched vjp per output (one-hot cotangent)."""
    import mlx.core as mx  # type: ignore
    z = mx.array(z_np)
    rows = []
    for j in out_idx:
        cot = np.zeros((len(z_np), out_dim), np.float32)
        cot[:, j] = 1.0
        _, vjps = mx.vjp(head_fn, [z], [mx.array(cot)])
        mx.eval(vjps[0])
        rows.append(np.array(vjps[0].tolist(), np.float32))
    return np.stack(rows)


def measure_model(m, imgs, Ych: np.ndarray, tr: np.ndarray, te: np.ndarray,
                  field_idx: tuple[int, ...], n_behavior: int) -> dict:
    """The full section-2 procedure for one model on one arm's data."""
    z = encode_z(m, imgs)
    pr = participation_ratio(z)
    probe_r2 = kfold_r2(z[tr], Ych[tr])                    # per-channel gate, train split
    dirs, _mu, sd = probe_directions(z[tr], Ych[tr])
    jac_e = head_jacobian_rows(m.essence_from_z, z[te], (0, 1, 2), len(ESSENCE_AXES))
    D, mism = directional_D(alignment_matrix(dirs, jac_e, sd), SIGNS_ESSENCE)
    jac_b = head_jacobian_rows(m.behavior_from_z, z[te], field_idx, n_behavior)
    Db, _ = directional_D(alignment_matrix(dirs, jac_b, sd), SIGNS_BEHAVIOR)
    jac_min = min(float(np.median(np.linalg.norm(jac_e[p] * sd[None, :], axis=1)))
                  for p in range(len(ESSENCE_AXES)))
    return {"z": z, "pr": pr, "probe_r2": probe_r2, "sd": sd,
            "D": D, "Db": Db, "mism": mism, "jac_min": jac_min}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/f27b",
                    help="F27b checkpoint root (shared seeds + renderonly/behavonly)")
    ap.add_argument("--f27-report", default=None,
                    help="F27b report to carry apparatus r from (default <root>/f27_report.json)")
    ap.add_argument("--out", default="runs/f28/f28_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    f27_report = args.f27_report or f"{args.root}/f27_report.json"

    plan = []
    for tag, data in ARMS:
        cks = sorted(glob.glob(f"{args.root}/{tag}_s*/model.safetensors"))
        ro = f"{args.root}/{tag}_renderonly/model.safetensors"
        bo = f"{args.root}/{tag}_behavonly/model.safetensors"
        has_dj = os.path.exists(ro) and os.path.exists(bo)
        plan.append((tag, data, cks, ro, bo, has_dj))
        print(f"{tag:5s} {data:18s} {len(cks)} shared seeds, "
              f"disjoint pair {'present' if has_dj else 'MISSING'}")
    if args.dry_run:
        n_enc = sum(len(c) + (2 if dj else 0) + N_UNTRAINED for _, _, c, _, _, dj in plan)
        print(f"would encode {n_enc} model/arm passes; apparatus r from {f27_report}")
        return

    import mlx.core as mx  # type: ignore

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    with open(f27_report) as f:
        r_by_arm = {k: float(v) for k, v in json.load(f)["apparatus"]["r"].items()}
    print("apparatus r (carried from F27b): "
          + ", ".join(f"{t} {r_by_arm[t]:+.3f}" for t, _ in ARMS))

    names = behavior_field_names()
    field_idx = tuple(names.index(f) for f in BEHAVIOR_FIELDS)  # missing name -> loud
    report: dict = {"apparatus": {"r": r_by_arm, "source": f27_report},
                    "arms": {}}

    for tag, data, cks, ro, bo, has_dj in plan:
        if len(cks) < 2:
            print(f"=== {tag}: {len(cks)} shared checkpoints, skipped ===")
            continue
        print(f"=== arm {tag} ===")
        imgs, _Yb, Ya, tr, te, _ds = load_arrays(data)
        Ych = channel_targets(Ya)

        shared = []
        for ck in cks:
            m = load_model(ck)
            shared.append(measure_model(m, imgs, Ych, tr, te, field_idx, len(names)))
            del m
        untrained = []
        for i in range(N_UNTRAINED):
            mx.random.seed(UNTRAINED_SEED_BASE + i)
            m = build_model(ModelConfig(image_size=128))
            untrained.append(measure_model(m, imgs, Ych, tr, te, field_idx, len(names)))
            del m

        D_un = float(np.mean([u["D"] for u in untrained]))
        Db_un = float(np.mean([u["Db"] for u in untrained]))
        learned = [s["D"] - D_un for s in shared]
        learned_b = [s["Db"] - Db_un for s in shared]

        disjoint: dict | None = None
        prs_extra, jac_extra = [], []
        if has_dj:
            m_ro, m_bo = load_model(ro), load_model(bo)
            z_ro, z_bo = encode_z(m_ro, imgs), encode_z(m_bo, imgs)
            prs_extra = [participation_ratio(z_ro), participation_ratio(z_bo)]
            ro_r2 = kfold_r2(z_ro[tr], Ych[tr])
            if float(ro_r2.min()) < PROBE_R2_TRAINED:
                print(f"  ! {tag}: renderonly probe min R2 {ro_r2.min():.3f} < "
                      f"{PROBE_R2_TRAINED} -> disjoint cell VOID (emitted as null)")
            else:
                dirs_ro, _mu_ro, sd_ro = probe_directions(z_ro[tr], Ych[tr])
                W_st, mu_bo, sd_bo = ridge_weights(z_bo[tr], z_ro[tr])
                st_r2 = variance_weighted_r2(z_ro[te],
                                             stitch_predict(W_st, mu_bo, sd_bo, z_bo[te]))
                dirs_t = transfer_directions(dirs_ro, sd_ro, W_st)
                jac_bo = head_jacobian_rows(m_bo.essence_from_z, z_bo[te],
                                            (0, 1, 2), len(ESSENCE_AXES))
                D_dj, _ = directional_D(alignment_matrix(dirs_t, jac_bo, sd_bo),
                                        SIGNS_ESSENCE)
                jac_extra = [min(float(np.median(
                    np.linalg.norm(jac_bo[p] * sd_bo[None, :], axis=1)))
                    for p in range(len(ESSENCE_AXES)))]
                disjoint = {"D": D_dj, "stitch_r2": st_r2,
                            "probe_r2_min": float(ro_r2.min())}
            del m_ro, m_bo
        del imgs
        mx.clear_cache()

        gates = {
            "latent_pr_min": float(min([s["pr"] for s in shared] + prs_extra)),
            "probe_r2_min_trained": float(min(float(s["probe_r2"].min()) for s in shared)),
            "probe_r2_min_untrained": float(min(float(u["probe_r2"].min())
                                                for u in untrained)),
            "jac_norm_median_min": float(min([s["jac_min"] for s in shared]
                                             + [u["jac_min"] for u in untrained]
                                             + jac_extra)),
        }
        report["arms"][tag] = arm_record(
            r_by_arm[tag], gates, learned, learned_b,
            float(np.mean([s["mism"] for s in shared])), disjoint,
            extra={"n_seeds": len(shared),
                   "D_trained_per_seed": [float(s["D"]) for s in shared],
                   "D_untrained_mean": D_un})
        dj_s = "VOID/none" if disjoint is None else f"{disjoint['D']:+.4f}"
        print(f"  learned_D {np.mean(learned):+.4f} (untrained {D_un:+.4f})  "
              f"behavior {np.mean(learned_b):+.4f}  "
              f"mism|abs| {report['arms'][tag]['mismatched_abs_mean']:.4f}  "
              f"disjoint {dj_s}  PR>={gates['latent_pr_min']:.1f}  "
              f"probeR2>={gates['probe_r2_min_trained']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}\nnext: .venv/bin/python scripts/f28_verdicts.py "
          f"--report {args.out}")


if __name__ == "__main__":
    main()
