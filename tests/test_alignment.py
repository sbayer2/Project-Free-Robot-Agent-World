"""Tests for the F24 alignment metrics (src/pseudomarble/models/alignment.py).

Synthetic data only -- the preregistration (docs/CONVERGENCE.md) forbids
touching trained checkpoints before the freeze, and these tests are part of
the frozen commit. Skips cleanly if numpy is unavailable.

The key cases encode the preregistration's gates and section 3:
* self-alignment is exactly 1.0; scene-shuffle drops to ~chance (gate `self`)
* the JL floor is REAL: two untrained random projections of the same
  structure align far above chance -- the reason raw alignment must never be
  reported (section 3)
* trit agreement is relabeling-invariant, and its shuffle null carries the
  same max-over-relabelings selection as the statistic
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

from pseudomarble.models.alignment import (  # noqa: E402
    linear_cka,
    mutual_information,
    mutual_knn_alignment,
    participation_ratio,
    terciles,
    trit_agreement,
    trit_agreement_null,
)

N, K = 200, 10
CHANCE = K / (N - 1)


def _skip_if_no_numpy():
    if not _HAVE_NUMPY:  # pragma: no cover
        print("numpy unavailable; skipping")
        return True
    return False


def test_self_alignment_is_one():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 32))
    assert mutual_knn_alignment(X, X, k=K) == 1.0
    assert abs(linear_cka(X, X) - 1.0) < 1e-9


def test_scene_shuffle_drops_to_chance():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(1)
    X = rng.normal(size=(N, 32))
    Xs = X[rng.permutation(N)]
    a = mutual_knn_alignment(X, Xs, k=K)
    assert a < 3 * CHANCE, f"shuffled alignment {a:.3f} not near chance {CHANCE:.3f}"


def test_jl_floor_shared_structure_aligns_untrained():
    """Two RANDOM projections of the same rank-5 structure align far above
    chance -- the source article's own toy, and the reason the preregistration
    demands baseline subtraction (docs/CONVERGENCE.md section 3)."""
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(N, 5))
    X = Z @ rng.normal(size=(5, 40))
    Y = Z @ rng.normal(size=(5, 25))
    noise = rng.normal(size=(N, 40))
    shared = mutual_knn_alignment(X, Y, k=K)
    unrelated = mutual_knn_alignment(noise, Y, k=K)
    assert shared > 0.5, f"shared-structure alignment only {shared:.3f}"
    assert unrelated < 3 * CHANCE
    assert linear_cka(X, Y) > 0.7
    assert linear_cka(noise, Y) < 0.3


def test_cka_rotation_and_scale_invariance():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(4)
    X = rng.normal(size=(N, 16))
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    assert abs(linear_cka(X, 3.7 * (X @ Q)) - 1.0) < 1e-9


def test_knn_alignment_bounds_and_symmetry():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(5)
    X, Y = rng.normal(size=(N, 8)), rng.normal(size=(N, 8))
    a, b = mutual_knn_alignment(X, Y, k=K), mutual_knn_alignment(Y, X, k=K)
    assert 0.0 <= a <= 1.0
    assert abs(a - b) < 1e-12  # set intersection is symmetric


def test_participation_ratio_matches_trainpy_definition():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(6)
    iid = rng.normal(size=(4000, 32))
    pr = participation_ratio(iid)
    assert 25 < pr <= 32.5, f"iid PR {pr:.1f} should approach dim 32"
    one_dim = np.outer(rng.normal(size=300), np.ones(32))
    assert participation_ratio(one_dim) < 32.5  # degenerate but defined
    assert participation_ratio(np.zeros((10, 5))) == 0.0  # collapsed -> 0


def test_trit_agreement_relabeling_invariant():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(7)
    a = rng.integers(-1, 2, size=512)
    relabeled = np.array([{-1: 1, 0: -1, 1: 0}[int(v)] for v in a])
    frac, perm = trit_agreement(a, relabeled)
    assert frac == 1.0
    assert perm == {-1: 1, 0: -1, 1: 0}


def test_trit_agreement_null_brackets_random_codes():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(8)
    a = rng.integers(-1, 2, size=512)
    b = rng.integers(-1, 2, size=512)
    frac, _ = trit_agreement(a, b)
    null = trit_agreement_null(a, b, n_draws=200)
    # Independent codes should land inside their own shuffle null band.
    assert null.min() <= frac <= null.max() + 0.05
    # Max-over-6-relabelings of a ~1/3 match sits well above 1/3 but below 1/2.
    assert 0.30 < null.mean() < 0.50


def test_trit_agreement_rejects_non_trits():
    if _skip_if_no_numpy():
        return
    try:
        trit_agreement([0, 1, 2], [0, 1, -1])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("non-trit codes must raise")


def test_mutual_information_endpoints():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(9)
    x = rng.integers(0, 5, size=5000)
    # Perfect dependence: MI equals the entropy of x.
    ent = -sum(p * np.log(p) for p in np.bincount(x) / len(x) if p > 0)
    assert abs(mutual_information(x, x) - ent) < 1e-9
    # Independence: MI near zero (plug-in bias is O(classes^2 / n)).
    y = rng.integers(0, 5, size=5000)
    assert mutual_information(x, y) < 0.02


def test_terciles_balanced():
    if _skip_if_no_numpy():
        return
    rng = np.random.default_rng(10)
    labels = terciles(rng.normal(size=900))
    counts = np.bincount(labels, minlength=3)
    assert counts.min() > 250, f"terciles unbalanced: {counts.tolist()}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} alignment tests passed")
