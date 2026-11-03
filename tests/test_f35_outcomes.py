"""Pins for the F35 outcome fields (scripts/f35_outcomes.py), D7 rules: ramp
slides floored at the 1 cm still threshold, no kick stop_s, and the dropramp
slide defined for every scene with tumbling kept as a covariate, never a
mask (docs/CROSS_INTERACTION.md §9, D7)."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import f35_outcomes as O  # noqa: E402


def test_ramp_slide_dead_zone_is_the_still_threshold():
    d = np.array([0.0, 0.004, 0.0099, 0.01, 0.5])
    assert np.allclose(O.log_slide(d), [-2.0, -2.0, -2.0, -2.0, np.log10(0.5)])
    assert O.DEAD_ZONE_M == 0.01 and O.FIELDS_VERSION == "D7"


def test_covariate_r_is_zero_for_a_constant_and_signed_otherwise():
    E = np.array([[0.0, 1.0, -1.0], [1.0, 2.0, -2.0], [2.0, 3.0, -3.0], [3.0, 4.0, -4.0]])
    assert O.covariate_r(np.zeros(4), E) == [0.0, 0.0, 0.0]
    r = O.covariate_r(np.array([0, 0, 1, 1]), E)
    assert r[0] > 0.8 and r[2] < -0.8


def test_d7_fields_on_a_small_battery(tmp_path):
    pytest.importorskip("mujoco")
    out = tmp_path / "bat"
    subprocess.run([sys.executable, os.path.join(SCRIPTS, "f35_generate_battery.py"),
                    "--sample", "6", "--seed", "3", "--workers", "1", "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    f, fam, diag = O.compute(str(out))
    assert not [n for n in f if n.endswith("stop_s")]
    for n in f:
        if n.startswith("ramp") and n.endswith("log_slide"):
            assert (f[n] >= -2.0 - 1e-12).all()
    assert np.isfinite(f["dropramp.log_slide"]).all()          # tumblers kept
    tip = diag["covariates"]["dropramp_tumbled"]
    assert tip.shape == (6,) and tip.dtype == bool
    assert diag["dropramp_tumbled"] == int(tip.sum())
    assert fam["dropramp.log_slide"] == "dropramp"
    assert set(diag["covariates"]) == {"kick1.0_tipped", "kick3.0_tipped", "dropramp_tumbled"}
