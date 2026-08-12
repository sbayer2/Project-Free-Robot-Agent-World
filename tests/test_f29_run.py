"""Tests for the F29 main run driver (scripts/f29_run.py).

Pure-Python; no MLX, no MuJoCo. Pins the Amendment-1 constants, the
pilot-leak seed guard, and — the last unreviewed seam — the report schema
end-to-end through the FROZEN scripts/f29_verdicts.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from f29_generate_pilot import pilot_seed  # noqa: E402
from f29_run import (  # noqa: E402
    ARMS,
    MAIN_ALPHA,
    MAIN_DATA,
    MAIN_SEED,
    arm_record,
)

REPO = os.path.join(os.path.dirname(__file__), "..")


def test_amendment_1_constants_pinned():
    assert MAIN_ALPHA == 0.40
    assert MAIN_SEED == 2941
    assert MAIN_DATA == "data/pm_f29_holdout_{arm}"
    assert ARMS == ("ctrl", "base", "g2", "loud")


def test_main_seed_is_not_the_pilot_seed():
    """The Amendment-1 leak guard: seed 2940 would replay pm_f29_pilot_a40_*
    scene-for-scene and import the disclosed pilot leak into the
    verdict-bearing measurement."""
    assert MAIN_SEED != pilot_seed(MAIN_ALPHA)


def test_seed_override_changes_the_material_stream():
    """pilot_assignments(seed=MAIN_SEED) must draw different physics than
    the pilot's own alpha-0.40 stream."""
    import random

    from f29_generate_pilot import physics_rows, pilot_assignments

    from pseudomarble.config import PHYSICS_NORMALIZERS as N
    rng = random.Random(0)
    train = [(rng.uniform(0.02, 0.10) * N["density"],
              rng.uniform(0.02, 0.10) * N["friction"],
              rng.uniform(0.02, 0.10) * N["restitution"]) for _ in range(50)]
    a_pilot, s_pilot = pilot_assignments("loud", MAIN_ALPHA, train, n_scenes=6)
    a_main, s_main = pilot_assignments("loud", MAIN_ALPHA, train, n_scenes=6,
                                       seed=MAIN_SEED)
    assert s_pilot["seed"] == pilot_seed(MAIN_ALPHA)
    assert s_main["seed"] == MAIN_SEED
    assert physics_rows(a_pilot) != physics_rows(a_main)


TRAIN_GAIN_JOINT = {"ctrl": 1.121, "base": 1.113, "g2": 3.625, "loud": 5.672}


def _gates(arm="loud", gain_min=None, jpr=30.0, dpr=28.0, r=0.99):
    """Passing gates fixture. Amendment 2 made the retraining gate
    arm-relative (>= 0.85 x joint train gain), so the default healthy value
    is 0.95 x that arm's joint gain."""
    if gain_min is None:
        gain_min = 0.95 * TRAIN_GAIN_JOINT[arm]
    return {"r_held_out": r + 0.01, "r_train": r,
            "extrapolation_frac_outside": 1.0,
            "disjoint_train_gain_min": gain_min,
            "joint_pr_min": jpr, "disjoint_pr_min": dpr}


def test_arm_record_schema_feeds_the_frozen_verdicts_script(tmp_path):
    """A synthetic report assembled with the driver's own arm_record() must
    run through scripts/f29_verdicts.py end to end (the seam Agent2
    flagged; same pattern as the F28/F32 driver tests)."""
    rs = {"ctrl": 0.008, "base": 0.263, "g2": 0.809, "loud": 0.992}
    joint = {"ctrl": [1.10, 1.12, 1.09], "base": [1.20, 1.18, 1.22],
             "g2": [2.80, 2.85, 2.75], "loud": [4.20, 4.30, 4.10]}
    disjoint = {"ctrl": [1.11, 1.13, 1.08], "base": [1.18, 1.16, 1.20],
                "g2": [2.78, 2.83, 2.77], "loud": [4.18, 4.25, 4.15]}
    arms = {t: arm_record(rs[t], _gates(arm=t, r=rs[t]), joint[t], disjoint[t])
            for t in ARMS}
    rep = {"meta": {"amendment_1": {"alpha_star": MAIN_ALPHA},
                    "main_alpha": MAIN_ALPHA, "main_seed": MAIN_SEED},
           "arms": arms}
    p = tmp_path / "f29_report.json"
    p.write_text(json.dumps(rep))
    out = subprocess.run(
        [sys.executable, os.path.join("scripts", "f29_verdicts.py"),
         "--report", str(p)],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert "GATES: ALL PASS" in out.stdout
    assert "H0:" in out.stdout and "-> PASS" in out.stdout
    assert "H2 [loud PRIMARY]" in out.stdout
    assert "H3:" in out.stdout
    assert "H4:" in out.stdout


def test_failed_gate_withholds_verdicts(tmp_path):
    """A below-floor disjoint train gain must fail gates and withhold H0-H4."""
    rs = {"ctrl": 0.008, "base": 0.263, "g2": 0.809, "loud": 0.992}
    arms = {}
    for t in ARMS:
        g = _gates(arm=t, r=rs[t])
        if t == "g2":
            g["disjoint_train_gain_min"] = 1.2  # << 0.85 x 3.625 (Amendment 2 gate)
        arms[t] = arm_record(rs[t], g, [2.0, 2.1, 1.9], [2.0, 2.1, 1.9])
    rep = {"meta": {"amendment_1": {"alpha_star": MAIN_ALPHA}}, "arms": arms}
    p = tmp_path / "f29_report_bad.json"
    p.write_text(json.dumps(rep))
    out = subprocess.run(
        [sys.executable, os.path.join("scripts", "f29_verdicts.py"),
         "--report", str(p)],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert "RETRAINING: g2" in out.stdout
    assert "verdicts withheld" in out.stdout
    # Amendment 2 layout fix: a failed run withholds its outcome table too.
    assert "advantage" not in out.stdout


def test_missing_amendment_fails_pilot_gate(tmp_path):
    """meta.amendment_1 absent -> the frozen pilot gate fires."""
    rs = {"ctrl": 0.008, "base": 0.263, "g2": 0.809, "loud": 0.992}
    arms = {t: arm_record(rs[t], _gates(arm=t, r=rs[t]), [2.0, 2.1, 1.9],
                          [2.0, 2.1, 1.9]) for t in ARMS}
    rep = {"meta": {}, "arms": arms}
    p = tmp_path / "f29_report_nopilot.json"
    p.write_text(json.dumps(rep))
    out = subprocess.run(
        [sys.executable, os.path.join("scripts", "f29_verdicts.py"),
         "--report", str(p)],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert "PILOT: Amendment 1 not recorded" in out.stdout
    assert "verdicts withheld" in out.stdout


def test_arm_record_types_are_json_native():
    rec = arm_record(0.5, _gates(), [1.0, 2.0], [1.5, 2.5])
    assert isinstance(rec["r"], float)
    assert all(isinstance(g, float) for g in rec["joint"]["per_seed_gain"])
    assert all(isinstance(g, float) for g in rec["disjoint"]["per_pair_gain"])
    json.dumps(rec)  # must serialize
