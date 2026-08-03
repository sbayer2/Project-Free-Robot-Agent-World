"""Cross-model representational alignment metrics (F24, docs/CONVERGENCE.md).

Measures whether two *independently trained* models built the same latent
geometry for the same scenes -- the Platonic Representation Hypothesis
(Huh et al. 2024, arXiv 2405.07987) instrumented for this project's worlds.

Two continuous metrics over paired representation matrices X, Y of the same n
scenes (rows correspond -- scene i is the same object in both):

* ``mutual_knn_alignment`` -- the 2024 paper's primary: mean overlap of the
  k-nearest-neighbor sets of each scene in the two spaces. Local structure
  only; needs no shared basis. Chance for unrelated spaces is ~ k/(n-1).
* ``linear_cka`` -- Kornblith et al. 2019: normalized HSIC of the centered
  Gram matrices. Global linear structure; rotation/scale invariant.

One categorical statistic for k=1 FSQ codes (H3, "convergence under
scarcity"): ``trit_agreement`` -- best scene-wise match over the 6 relabelings
of {-1,0,1}, with ``trit_agreement_null`` providing the shuffle baseline under
the SAME max-over-relabelings selection.

The JL floor (preregistration section 3): untrained encoders fed the same
images already share neighborhood structure (random maps preserve relative
geometry), so raw alignment is uninterpretable -- always subtract an
untrained-pair baseline computed on the same inputs. These functions compute
raw numbers only; the discipline lives in the caller.

Lazy numpy import, repo-wide convention: the module imports with no deps.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations


def _np():
    import numpy as np

    return np


def pairwise_sq_dists(X):
    """(n,d) -> (n,n) squared Euclidean distances (numpy, exact, symmetric)."""
    np = _np()
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def knn_indices(X, k: int):
    """(n,d) -> (n,k) indices of each row's k nearest neighbors, self excluded.

    Stable argsort (preregistration gate: deterministic tie handling).
    """
    np = _np()
    d2 = pairwise_sq_dists(X)
    np.fill_diagonal(d2, np.inf)
    order = np.argsort(d2, axis=1, kind="stable")
    return order[:, :k]


def mutual_knn_alignment(X, Y, k: int = 10) -> float:
    """Mean over scenes of |kNN_X(i) & kNN_Y(i)| / k, in [0,1]."""
    np = _np()
    ix, iy = knn_indices(X, k), knn_indices(Y, k)
    n = ix.shape[0]
    overlaps = np.empty(n, dtype=np.float64)
    for i in range(n):
        overlaps[i] = len(set(ix[i].tolist()) & set(iy[i].tolist())) / k
    return float(overlaps.mean())


def _centered_gram(X):
    np = _np()
    X = np.asarray(X, dtype=np.float64)
    K = X @ X.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X, Y) -> float:
    """Centered Kernel Alignment (linear kernel), in [0,1]."""
    np = _np()
    Kc, Lc = _centered_gram(X), _centered_gram(Y)
    hsic_kl = np.trace(Kc @ Lc)
    hsic_kk = np.trace(Kc @ Kc)
    hsic_ll = np.trace(Lc @ Lc)
    denom = float(np.sqrt(hsic_kk * hsic_ll))
    return float(hsic_kl / denom) if denom > 0 else 0.0


def participation_ratio(Z) -> float:
    """train.py:94's PR -- (sum var)^2 / sum(var^2) over per-dim variance.

    The F12 healthy band (PR ~ 8-84) is stated in these units; do not swap in
    the covariance eigenvalue spectrum (see scripts/f22_pilot_eval.py note).

    F27 bugfix: a numerically-collapsed latent (constant z up to float32
    noise, per-dim variance ~1e-9) has UNIFORM tiny variance, so the raw
    ratio reads ~n_dims -- full rank -- and masks exactly the collapse the
    F10/F12 gate exists to catch. The epsilon in the denominator (matching
    run_coherence_experiment.py's guard) sends that case toward 0 instead;
    genuinely varying latents (per-dim var >> 1e-6) are unaffected.
    """
    np = _np()
    var = np.asarray(Z, dtype=np.float64).var(axis=0)
    s1, s2 = float(var.sum()), float((var**2).sum())
    return s1 * s1 / (s2 + 1e-12)


_TRIT_PERMS: list[dict] = [
    {-1: p[0], 0: p[1], 1: p[2]} for p in permutations((-1, 0, 1))
]


def trit_agreement(a: Sequence[int], b: Sequence[int]) -> tuple[float, dict]:
    """Best scene-wise agreement over the 6 relabelings of {-1,0,1}.

    Returns (fraction agreeing, the winning relabeling a->b). Codes outside
    {-1,0,1} raise -- that would mean a non-FSQ latent was passed by mistake.
    """
    np = _np()
    a = np.asarray(a, dtype=np.int64).reshape(-1)
    b = np.asarray(b, dtype=np.int64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"code length mismatch: {a.shape} vs {b.shape}")
    for arr, name in ((a, "a"), (b, "b")):
        bad = set(np.unique(arr).tolist()) - {-1, 0, 1}
        if bad:
            raise ValueError(f"non-trit values in {name}: {sorted(bad)}")
    best, best_perm = -1.0, _TRIT_PERMS[0]
    for perm in _TRIT_PERMS:
        mapped = np.array([perm[int(v)] for v in a.tolist()], dtype=np.int64)
        frac = float((mapped == b).mean())
        if frac > best:
            best, best_perm = frac, perm
    return best, best_perm


def trit_agreement_null(a: Sequence[int], b: Sequence[int], n_draws: int = 1000,
                        seed: int = 2026):
    """Shuffle-null for trit_agreement: permute b across scenes, keep marginals.

    The max-over-relabelings selection is applied to every draw, so the null
    carries the same selection effect as the statistic (preregistration s2).
    Returns the (n_draws,) array of null agreements.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    b = np.asarray(b, dtype=np.int64).reshape(-1)
    draws = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        draws[i] = trit_agreement(a, rng.permutation(b))[0]
    return draws


def mutual_information(x: Sequence[int], y: Sequence[int]) -> float:
    """Plug-in MI (natural log) between two discrete label sequences."""
    np = _np()
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"label length mismatch: {x.shape} vs {y.shape}")
    mi = 0.0
    for xv in np.unique(x):
        mx = x == xv
        px = mx.mean()
        for yv in np.unique(y):
            pxy = float(np.logical_and(mx, y == yv).mean())
            if pxy > 0:
                py = float((y == yv).mean())
                mi += pxy * float(np.log(pxy / (px * py)))
    return float(mi)


def terciles(values: Sequence[float]):
    """Continuous values -> {0,1,2} tercile labels (for H3b's factor MI)."""
    np = _np()
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    lo, hi = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    return (v > lo).astype(np.int64) + (v > hi).astype(np.int64)


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    """Adjusted Rand index between two labelings of the same scenes.

    Permutation-invariant by construction and chance-adjusted to ~0, so it
    generalizes trit_agreement (whose max-over-relabelings search is factorial
    in the code width) to FSQ codes of any k: treat each distinct code vector
    as one cluster label. 1.0 = identical partitions.
    """
    np = _np()
    a = np.asarray(a).reshape(len(a), -1)
    b = np.asarray(b).reshape(len(b), -1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: {a.shape[0]} vs {b.shape[0]}")
    _, la = np.unique(a, axis=0, return_inverse=True)
    _, lb = np.unique(b, axis=0, return_inverse=True)
    n = la.shape[0]
    contingency = np.zeros((la.max() + 1, lb.max() + 1), dtype=np.int64)
    for i, j in zip(la.tolist(), lb.tolist(), strict=True):
        contingency[i, j] += 1

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_ij = float(comb2(contingency).sum())
    sum_a = float(comb2(contingency.sum(axis=1)).sum())
    sum_b = float(comb2(contingency.sum(axis=0)).sum())
    total = comb2(n)
    expected = sum_a * sum_b / total if total > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    return float((sum_ij - expected) / denom) if denom != 0 else 1.0
