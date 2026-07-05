"""Tests for the basin-selection training levers (``models/train.py``).

Pure-Python: the warmup ramp and arg parsing import without MLX (all heavy
imports in train.py are lazy inside ``main``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.models.train import behavior_warmup_scale, parse_args  # noqa: E402


def test_warmup_disabled_is_always_full_weight():
    assert behavior_warmup_scale(0, 0) == 1.0
    assert behavior_warmup_scale(0, -3) == 1.0
    assert behavior_warmup_scale(100, 0) == 1.0


def test_warmup_ramps_linearly_then_saturates():
    assert behavior_warmup_scale(0, 10) == 0.0
    assert behavior_warmup_scale(5, 10) == 0.5
    assert behavior_warmup_scale(10, 10) == 1.0
    assert behavior_warmup_scale(50, 10) == 1.0


def test_parse_args_warmup_flag_default_off():
    args = parse_args(["--data", "d"])
    assert args.behavior_warmup_epochs == 0
    args = parse_args(["--data", "d", "--behavior-warmup-epochs", "10"])
    assert args.behavior_warmup_epochs == 10
