"""The F36 exploratory addendum (scripts/crossview_exploratory.py) on synthetic
runs: the label, no grade word anywhere, E1 recomputing only the comparisons
whose arms lost a loud run, E2's trajectory summary, and E3's Spearman rho and
seeded bootstrap."""

import json
import math
import os
import re
import sys

import pytest

pytest.importorskip("scipy")
pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
V = pytest.importorskip("crossview_verdicts")
E = pytest.importorskip("crossview_exploratory")


def _history(prs):
    return [{"epoch": i, "train_loss": 0.2 - 0.001 * i, "behavior_mse": 0.1,
             "essence_mse": 0.1, "render_mse": 0.0, "latent_pr": p}
            for i, p in enumerate(prs)]


def _write_run(path, prs):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metrics.json"), "w") as fh:
        json.dump({"config": {}, "history": _history(prs)}, fh)


def _reps(tmp, loud_pr=None, base=None):
    """Five arms, 20 loud / 3 ctrl runs. loud_pr overrides one arm's loud final
    PRs; base sets each arm's loud nested mean. Ctrl runs get a metrics.json."""
    loud_pr, base = loud_pr or {}, base or {}
    reps = {}
    for a in V.ARMS:
        reps[a] = {}
        for w, n in (("loud", 20), ("ctrl", 3)):
            rows = {}
            for i in range(n):
                run = os.path.join(tmp, f"{a}_{w}_s{i}")
                pr = (loud_pr.get(a) or [40.0] * n)[i] if w == "loud" else 40.0
                x = (base.get(a, 0.40) if w == "loud" else -0.01) + 0.01 * (i - (n - 1) / 2)
                rows[run] = {"view_pool": V.VIEW_POOL[a], "friction_nested": x,
                             "z_r2_fixed": {"density": 0.0, "friction": x - 0.02,
                                            "restitution": 0.98},
                             "z_r2_heldout": {"friction": x + 0.1}, "std_rank": 100,
                             "latent_pr": pr, "heldout_gain": 1.0}
                if w == "ctrl":
                    _write_run(run, [30.0] * 49 + [pr])
            reps[a][w] = {"device": "cpu", "runs": rows}
    return reps


def test_the_label_is_on_top_and_no_grade_word_appears(tmp_path):
    out = E.render(_reps(str(tmp_path)), None)
    assert out.splitlines()[2] == E.LABEL
    body = out.replace(E.LABEL, "")
    assert not re.search(r"RISES|NULL|FALLS|INCONCLUSIVE|READOUT|WITHHELD|would", body)


def test_e1_recomputes_only_comparisons_whose_arms_lost_a_loud_run(tmp_path):
    pr = {"P1c": [40.0] * 9 + [6.7] + [40.0] * 10}  # P1c_loud_s9 fails G1
    rows = {r["name"]: r for r in E.e1(_reps(str(tmp_path), loud_pr=pr))}
    changed = {n for n, r in rows.items() if r["dropped"]}
    assert changed == {"H1 stats mixing", "P1c vs P0"}
    h1 = rows["H1 stats mixing"]
    assert h1["dropped"] == ["P1c_loud_s9"]
    assert h1["registered"]["friction_nested"]["n"] == (20, 20)
    assert h1["exclusion"]["friction_nested"]["n"] == (20, 19)


def test_e1_exclusion_delta_is_a_plain_recompute_without_the_run(tmp_path):
    pr = {"P1c": [40.0] * 9 + [6.7] + [40.0] * 10}
    reps = _reps(str(tmp_path), loud_pr=pr, base={"P1": 0.45, "P1c": 0.30})
    h1 = next(r for r in E.e1(reps) if r["name"] == "H1 stats mixing")
    a = V.col(reps["P1"]["loud"], "friction_nested")
    b = [v["friction_nested"] for k, v in reps["P1c"]["loud"]["runs"].items()
         if not k.endswith("P1c_loud_s9")]
    w = h1["exclusion"]["friction_nested"]
    assert w["delta"] == pytest.approx(sum(a) / len(a) - sum(b) / len(b))
    lo, hi = w["ci"]
    assert lo < w["delta"] < hi


def test_e2_summarises_a_trajectory():
    t = E.trajectory(_history([3.9, 2.0, 9.0] + [12.0] * 46 + [0.1]))
    assert t["first_below"] == 0 and t["min"] == 0.1 and t["min_epoch"] == 49
    assert t["recovered"] is False and t["at"][0] == 3.9 and t["final"] == 0.1
    t = E.trajectory(_history([30.0, 5.0] + [40.0] * 48))
    assert t["first_below"] == 1 and t["recovered"] is True
    t = E.trajectory(_history([30.0] * 50))
    assert t["first_below"] is None and t["recovered"] is False


def test_e2_reads_every_ctrl_run_and_the_stability_runs(tmp_path):
    reps = _reps(str(tmp_path / "xv"))
    stab = tmp_path / "stability"
    for sub in E.STABILITY.values():
        _write_run(str(stab / sub), [50.0] * 50)
    rows = E.e2(reps, str(stab))
    assert sum(r["kind"] == "ctrl" for r in rows) == 15
    assert sum(r["kind"] == "stability" for r in rows) == 4


def test_spearman_handles_ties_and_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    assert E.spearman([1, 2, 3, 4], [2, 4, 9, 100]) == pytest.approx(1.0)
    assert E.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    x, y = [1, 2, 2, 3, 5, 5, 7], [3, 1, 4, 1, 5, 9, 2]
    assert E.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)
    assert math.isnan(E.spearman([1, 1, 1], [1, 2, 3]))


def test_the_bootstrap_is_seeded_and_brackets_the_estimate():
    x = [float(i) for i in range(20)]
    y = [v + (3.0 if i % 3 == 0 else 0.0) for i, v in enumerate(x)]
    a, b = E.boot_ci(x, y, b=500), E.boot_ci(x, y, b=500)
    assert a == b
    assert a[0] <= E.spearman(x, y) <= a[1]


def test_a_constant_input_prints_n_a_rather_than_failing():
    lo, hi = E.boot_ci([40.0] * 20, [float(i) for i in range(20)], b=50)
    assert math.isnan(lo) and math.isnan(hi) and E.fmt_rho(lo) == "n/a"
