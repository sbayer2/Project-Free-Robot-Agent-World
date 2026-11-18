"""The F36 tables (scripts/crossview_summary.py) on synthetic reports: the G1
failure list and counts, Fisher's exact test, the per-arm means, that a
WITHHELD comparison carries no grade word, D1's two orders, the restitution
counts, and that every Markdown table row keeps its column count.

Only the tests that render need scipy: render() runs crossview_verdicts,
whose NULL bound uses scipy's t quantile. The rest run on the standard
library, as on a plain `pip install -e ".[dev]"`."""

import math
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
V = pytest.importorskip("crossview_verdicts")
S = pytest.importorskip("crossview_summary")

D1_KEYS = ("concat", "D |DFT over view index|")


def _rep(arm, world, n, base, pr=None, rest=None):
    pr = pr or [40.0] * n
    rest = rest or [0.98] * n
    rows = {}
    for i in range(n):
        x = base + 0.01 * (i - (n - 1) / 2)
        rows[f"runs/xv/{arm}_{world}_s{i}"] = {
            "view_pool": V.VIEW_POOL[arm], "friction_nested": x,
            "z_r2_fixed": {"density": 0.0, "friction": x - 0.02, "restitution": rest[i]},
            "z_r2_heldout": {"friction": x + 0.1}, "std_rank": 100, "latent_pr": pr[i],
            "heldout_gain": 1.0,
            "noise": {"rel": 1e-4, "friction_nested": x - 0.001, "friction_fixed": x - 0.021},
            **({"d1": {D1_KEYS[0]: 0.5 + 0.01 * (i % 2), D1_KEYS[1]: 0.51 - 0.01 * (i % 2)}}
               if arm == "P0" and world == "loud" else {})}
    return {"device": "cpu", "runs": rows}


def _reps(pr=None, rest=None):
    """Every arm healthy at 0.40 loud, -0.01 ctrl, unless pr / rest override
    an (arm, world) with per-run lists."""
    pr, rest = pr or {}, rest or {}
    return {a: {w: _rep(a, w, 20 if w == "loud" else 3, 0.40 if w == "loud" else -0.01,
                        pr.get((a, w)), rest.get((a, w)))
                for w in ("loud", "ctrl")} for a in V.ARMS}


def _six_of_twelve():
    """Six of the twelve mixer and control ctrl runs below the floor."""
    return {("P1", "ctrl"): [0.9, 40, 40], ("P1c", "ctrl"): [0.0, 40, 40],
            ("P2", "ctrl"): [0.1, 7.975, 40], ("P2c", "ctrl"): [5.1, 6.0, 40]}


def test_g1_failures_list_every_run_below_the_floor_and_no_other():
    pr = {("P1", "ctrl"): [0.9, 8.0, 40], ("P2", "loud"): [7.99] + [40] * 19,
          ("P2c", "ctrl"): [math.nan, 40, 40]}
    fails = S.g1_failures(_reps(pr=pr))
    got = {(f["arm"], f["world"], f["run"]) for f in fails}
    assert got == {("P1", "ctrl", "P1_ctrl_s0"), ("P2", "loud", "P2_loud_s0"),
                   ("P2c", "ctrl", "P2c_ctrl_s0")}  # PR exactly 8.0 passes G1


def test_fisher_matches_the_hypergeometric_for_0_of_3_against_6_of_12():
    c = S.g1_counts(_reps(pr=_six_of_twelve()))
    assert c["P0_ctrl"] == (0, 3) and c["mixers_ctrl"] == (6, 12)
    assert c["mixers_loud"] == (0, 80) and c["P0_loud"] == (0, 20)
    total = math.comb(15, 3)
    assert c["fisher_one_sided"] == pytest.approx(math.comb(9, 3) / total)  # 84/455
    assert c["fisher_two_sided"] == pytest.approx((math.comb(9, 3) + math.comb(6, 3)) / total)


def test_arm_rows_are_the_mean_and_sd_of_each_readout():
    reps = _reps()
    row = next(r for r in S.arm_rows(reps) if r["arm"] == "P1" and r["world"] == "loud")
    assert row["n"] == 20
    assert row["nested"] == V.col(reps["P1"]["loud"], "friction_nested")
    assert sum(row["fixed"]) / 20 == pytest.approx(0.38)
    assert sum(row["noise"]) / 20 == pytest.approx(0.399)


def test_a_withheld_comparison_names_its_gate_and_carries_no_grade_word():
    pytest.importorskip("scipy")  # crossview_verdicts' NULL bound
    out = S.render(_reps(pr=_six_of_twelve()))
    rows = [ln for ln in out.splitlines() if "WITHHELD" in ln]
    assert len(rows) == 7  # every comparison includes a failing arm
    for ln in rows:
        assert not re.search(r"RISES|FALLS|NULL|INCONCLUSIVE|READOUT", ln)
    assert any("WITHHELD (G1: P1, P1c)" in ln for ln in rows)


def test_an_ungated_comparison_shows_its_verdict_and_both_grades():
    pytest.importorskip("scipy")  # crossview_verdicts' NULL bound
    out = S.render(_reps())
    h1 = next(ln for ln in out.splitlines() if ln.startswith("| H1 stats mixing"))
    assert "NULL" in h1 and "(nested NULL, fixed NULL)" in h1 and "WITHHELD" not in h1


def test_d1_reports_both_orders():
    o = S.d1_orders(_reps())
    assert o["max_of_means"] == pytest.approx(0.505)
    assert o["mean_of_maxima"] == pytest.approx(0.51)
    assert o["mean_of_maxima"] >= o["max_of_means"]


def test_restitution_counts_runs_below_the_p4_floor():
    rest = {("P2", "loud"): [0.96] * 14 + [0.98] * 6}
    r = S.restitution(_reps(rest=rest))
    assert r["P2"]["below"] == 14 and r["P2"]["min"] == pytest.approx(0.96)
    assert r["P0"]["below"] == 0


def test_every_table_row_keeps_its_column_count():
    pytest.importorskip("scipy")  # crossview_verdicts' NULL bound
    out = S.render(_reps(pr=_six_of_twelve()))
    for block in out.split("\n\n"):
        rows = [ln for ln in block.splitlines() if ln.startswith("|")]
        if not rows:
            continue
        widths = {len(re.split(r"(?<!\\)\|", ln)) for ln in rows}
        assert len(widths) == 1, block


def test_fisher_needs_no_scipy_and_agrees_with_it_where_installed():
    two, one = S.fisher_2x2(0, 3, 6, 6)
    assert (two, one) == (pytest.approx(104 / 455), pytest.approx(84 / 455))
    stats = pytest.importorskip("scipy.stats")
    for table in ([0, 3, 6, 6], [2, 5, 1, 9], [3, 0, 0, 3], [1, 1, 7, 3]):
        a, b, c, d = table
        two, one = S.fisher_2x2(a, b, c, d)
        assert two == pytest.approx(stats.fisher_exact([[a, b], [c, d]])[1])
        assert one == pytest.approx(
            stats.fisher_exact([[a, b], [c, d]], alternative="less")[1])


def test_a_missing_or_nan_pr_prints_and_still_fails_g1():
    assert S.fmt_pr(None) == "non-finite" and S.fmt_pr(math.nan, 3) == "non-finite"
    assert S.lowest([40.0, math.nan, 3.0]) == "non-finite"
    assert S.lowest([40.0, 3.04]) == "3.0"
    assert S.median_finite([None, 1.0, 3.0]) == 2.0
    pytest.importorskip("scipy")
    out = S.render(_reps(pr={("P2", "ctrl"): [None, 40, 40]}))
    assert "| P2_ctrl_s0 | non-finite |" in out
    assert "| P2 | FAIL | 40.0 | non-finite |" in out
