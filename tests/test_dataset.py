"""Tests for the dataset loader and behavior-vector assembly (pure-Python).

We synthesize a tiny dataset on disk (manifest + sample.json, no image files) and
exercise everything that doesn't need pixels: split filtering, target shapes,
stable probe ordering, normalization, and deterministic batching.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble import probes as P  # noqa: E402
from pseudomarble.data import samples  # noqa: E402
from pseudomarble.data.dataset import PseudoMarbleDataset  # noqa: E402
from pseudomarble.materials import MaterialSampler  # noqa: E402


def _outcome(toppled=False, settle=0.5, slid=0.1, bounces=0,
             maxh=0.3, path=0.4, tilt=5.0):
    return {
        "toppled": toppled, "settle_time": settle, "slid_distance": slid,
        "n_bounces": bounces, "max_height": maxh, "path_length": path,
        "final_tilt_deg": tilt,
    }


def _probe(kind, **kw):
    return {"probe": kind, "spec": {"kind": kind}, "outcome": _outcome(**kw)}


def _write_dataset(root, specs):
    sampler = MaterialSampler(seed=0)
    scenes = []
    for i, (split, shape, probe_order) in enumerate(specs):
        ms = sampler.sample(material_id=f"scene_{i:06d}")
        rec = samples.build_sample_record(
            scene_id=f"{split}_{i:06d}", split=split, shape=shape,
            frames=[samples.build_frame(0, "view_000.png", [1, 0, 0], [0, 0, 0])],
            resolution=64, generator="mujoco",
            material=ms.material, material_id=f"scene_{i:06d}",
            behavior=[_probe(k) for k in probe_order],
            material_truth_block=samples.material_truth(ms),
        )
        d = os.path.join(root, rec["scene_id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "sample.json"), "w") as fh:
            json.dump(rec, fh)
        scenes.append(rec)
    manifest = samples.build_manifest("mujoco", {"n": len(scenes)}, [], scenes)
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)


def test_split_filtering_and_lengths():
    with tempfile.TemporaryDirectory() as root:
        _write_dataset(root, [
            ("train", "box", ["drop", "tilt", "push"]),
            ("train", "sphere", ["drop", "tilt", "push"]),
            ("test", "cylinder", ["drop", "tilt", "push"]),
        ])
        assert len(PseudoMarbleDataset(root)) == 3
        assert len(PseudoMarbleDataset(root, split="train")) == 2
        assert len(PseudoMarbleDataset(root, split="test")) == 1


def test_target_shapes():
    with tempfile.TemporaryDirectory() as root:
        _write_dataset(root, [("train", "box", ["drop", "tilt", "push"])])
        ds = PseudoMarbleDataset(root)
        scene = ds[0]
        assert len(scene.behavior_target()) == P.BEHAVIOR_DIM == 21
        assert len(scene.essence_target()) == 3
        assert len(scene.factors_target()) == 4
        assert len(scene.appearance_target()) == 8  # F20 aux target
        assert len(scene.view_files()) == 1


def test_appearance_vector_normalizes_ior_and_orders_channels():
    from pseudomarble.data.dataset import APPEARANCE_FIELDS, appearance_vector

    v = appearance_vector({"base_color": [0.1, 0.2, 0.3, 1.0], "roughness": 0.4,
                           "metallic": 0.5, "transmission": 0.6, "ior": 1.45})
    assert len(v) == len(APPEARANCE_FIELDS) == 8
    assert v[:4] == [0.1, 0.2, 0.3, 1.0]          # base_color RGBA, in order
    assert v[4:7] == [0.4, 0.5, 0.6]              # roughness, metallic, transmission
    assert abs(v[7] - 0.45) < 1e-6                # ior shifted to ~[0,1]


def test_appearance_target_present_in_batch():
    with tempfile.TemporaryDirectory() as root:
        _write_dataset(root, [("train", "box", ["drop", "tilt", "push"])])
        ds = PseudoMarbleDataset(root)
        batch = next(ds.iter_batches(batch_size=1, shuffle=False))
        assert "appearance" in batch
        assert len(batch["appearance"][0]) == 8


def test_probe_order_is_stable_regardless_of_disk_order():
    # Same outcomes, different on-disk probe order -> identical target vector.
    with tempfile.TemporaryDirectory() as root:
        _write_dataset(root, [
            ("train", "box", ["drop", "tilt", "push"]),
            ("train", "box", ["push", "drop", "tilt"]),
        ])
        ds = PseudoMarbleDataset(root)
        a = ds[0].behavior_target()
        b = ds[1].behavior_target()
        assert a == b


def test_missing_probe_is_zero_filled():
    vec = P.behavior_vector([_probe("drop")], normalize=True)
    assert len(vec) == P.BEHAVIOR_DIM
    # tilt + push blocks (the last 14 entries) are zeros.
    assert vec[len(P.OUTCOME_FIELDS):] == [0.0] * (2 * len(P.OUTCOME_FIELDS))


def test_normalization_scales_large_fields():
    raw = P.behavior_vector([_probe("push", tilt=180.0)], normalize=False)
    norm = P.behavior_vector([_probe("push", tilt=180.0)], normalize=True)
    # push.final_tilt_deg is the last field of the push block.
    i = 2 * len(P.OUTCOME_FIELDS) + P.OUTCOME_FIELDS.index("final_tilt_deg")
    assert raw[i] == 180.0
    assert abs(norm[i] - 1.0) < 1e-9


def test_field_names_match_dim():
    assert len(P.behavior_field_names()) == P.BEHAVIOR_DIM
    assert P.behavior_field_names()[0] == "drop.toppled"


def test_batches_are_deterministic_and_cover_all():
    with tempfile.TemporaryDirectory() as root:
        _write_dataset(root, [("train", "box", ["drop", "tilt", "push"])] * 10)
        ds = PseudoMarbleDataset(root, split="train")
        b1 = [b["scene_ids"] for b in ds.iter_batches(4, seed=1)]
        b2 = [b["scene_ids"] for b in ds.iter_batches(4, seed=1)]
        assert b1 == b2
        seen = [sid for batch in b1 for sid in batch]
        assert len(seen) == 10 and len(set(seen)) == 10
        first = next(ds.iter_batches(4, seed=1))
        assert len(first["behavior"]) == 4
        assert len(first["behavior"][0]) == P.BEHAVIOR_DIM


def test_missing_manifest_raises():
    with tempfile.TemporaryDirectory() as root:
        try:
            PseudoMarbleDataset(root)
        except FileNotFoundError as exc:
            assert "manifest.json" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
