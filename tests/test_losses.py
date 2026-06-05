"""Tests for the framework-agnostic loss reference (pure-Python)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.models import losses  # noqa: E402


def test_mse_zero_for_identical():
    a = [[1.0, 2.0], [3.0, 4.0]]
    assert losses.mse(a, a) == 0.0


def test_mse_known_value():
    pred = [[0.0, 0.0]]
    target = [[3.0, 4.0]]  # squared errors 9 + 16 = 25 over 2 elems -> 12.5
    assert losses.mse(pred, target) == 12.5


def test_combined_loss_weights_essence():
    bp, bt = [[0.0]], [[2.0]]   # behavior mse = 4
    ep, et = [[0.0]], [[2.0]]   # essence  mse = 4
    out = losses.combined_loss(bp, bt, ep, et, essence_weight=0.5)
    assert out["behavior"] == 4.0
    assert out["essence"] == 4.0
    assert out["total"] == 4.0 + 0.5 * 4.0


def test_per_field_mse():
    pred = [[0.0, 0.0], [0.0, 0.0]]
    target = [[1.0, 2.0], [3.0, 4.0]]  # field0: (1+9)/2=5 ; field1: (4+16)/2=10
    assert losses.per_field_mse(pred, target) == [5.0, 10.0]


def test_combined_loss_includes_render_when_given():
    bp = bt = [[0.0]]
    ep = et = [[0.0]]
    # one sample, a 1x1 RGB image; mse over the 3 channels = (1+0+0)/3
    rp = [[[[0.0, 0.0, 0.0]]]]
    rt = [[[[1.0, 0.0, 0.0]]]]
    out = losses.combined_loss(bp, bt, ep, et, essence_weight=0.5,
                               render_pred=rp, render_target=rt, render_weight=2.0)
    assert abs(out["render"] - 1.0 / 3.0) < 1e-9
    assert abs(out["total"] - 2.0 * (1.0 / 3.0)) < 1e-9  # behavior+essence are 0


def test_row_length_mismatch_raises():
    try:
        losses.mse([[1.0, 2.0]], [[1.0]])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
