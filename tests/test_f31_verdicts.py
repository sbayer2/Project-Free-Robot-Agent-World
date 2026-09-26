"""Pin the frozen F31a decision rules (docs/TRAJECTORY_SUPERVISION.md §5)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import f31_verdicts as V  # noqa: E402


def _rep(friction, gain, pr=40.0, density=0.6):
    return {"runs": {f"s{i}": {"z_r2": {"friction": f, "density": density, "restitution": 0.98},
                               "heldout_gain": g, "latent_pr": pr, "escape_epoch": 25,
                               "heldout_trajectory_gain": None}
                     for i, (f, g) in enumerate(zip(friction, gain, strict=True))}}


def _reps(loud_traj_f, loud_traj_g=(5.0, 5.1, 4.9), ctrl_f=(0.0, 0.01, -0.02), pr=40.0):
    return {"loud_joint": _rep([0.25, 0.26, 0.24], [5.0, 5.1, 4.9]),
            "loud_traj": _rep(list(loud_traj_f), list(loud_traj_g), pr=pr),
            "ctrl_joint": _rep(list(ctrl_f), [1.0, 1.0, 1.0]),
            "ctrl_traj": _rep(list(ctrl_f), [1.0, 1.0, 1.0])}


def test_classify_boundaries():
    assert V.classify(0.12, 3.0, 0.10, 0.05, 2.5) == "RISES"
    assert V.classify(0.12, 2.0, 0.10, 0.05, 2.5) == "INCONCLUSIVE"   # big but noisy
    assert V.classify(0.04, 9.0, 0.10, 0.05, 2.5) == "NULL"
    assert V.classify(-0.12, -3.0, 0.10, 0.05, 2.5) == "FALLS"


def test_h1_rises_null_and_gate_withholding():
    assert V.verdicts(_reps([0.40, 0.41, 0.39]))["H1_friction"]["verdict"] == "RISES"
    assert V.verdicts(_reps([0.25, 0.27, 0.23]))["H1_friction"]["verdict"] == "NULL"
    leaky = V.verdicts(_reps([0.40, 0.41, 0.39], ctrl_f=(0.0, 0.15, 0.0)))
    assert not leaky["gates"]["G2_control"]
    assert leaky["H1_friction"]["verdict"] == "WITHHELD"
    assert leaky["H2_gain"]["verdict"] != "WITHHELD"          # G2 does not gate H2
    sick = V.verdicts(_reps([0.40, 0.41, 0.39], pr=3.0))
    assert sick["H1_friction"]["verdict"] == sick["H2_gain"]["verdict"] == "WITHHELD"


def test_h2_falls():
    v = V.verdicts(_reps([0.25, 0.26, 0.24], loud_traj_g=(4.0, 4.1, 3.9)))
    assert v["H2_gain"]["verdict"] == "FALLS"
