"""Dataset loader for the behavior-based task.

Reads a generated dataset (``manifest.json`` + per-scene ``sample.json``) and
serves, for each scene:

  * the **appearance** the model sees   — multi-view RGB images, and
  * the **behavior** target it predicts — the flattened drop/tilt/push outcomes,
  * plus the **essence** (normalized physics) and the hidden **factors**, kept
    for auxiliary supervision and for measuring essence recovery.

Design follows the rest of the package: the manifest/target logic is pure-Python
and dependency-free (so it imports and tests in CI), while image loading needs
``numpy`` + an image reader and MLX batching needs ``mlx`` — both imported lazily
and only when you actually ask for arrays.

    from pseudomarble.data.dataset import PseudoMarbleDataset
    train = PseudoMarbleDataset("data/pseudo_marble", split="train")
    for batch in train.iter_batches(8, with_images=True, as_mlx=True):
        images, targets = batch["images"], batch["behavior"]
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

from pseudomarble import probes as P

ESSENCE_FIELDS = ("density", "friction", "restitution")
FACTOR_FIELDS = ("heaviness", "grip", "hardness", "clarity")
# The 8 clean appearance_params the renderer used (F20 aux target). color RGBA,
# roughness, metallic, transmission are already in [0,1]; ior sits in ~[1,2], so
# it is shifted to ~[0,1] (ior-1) to keep the per-channel MSE balanced.
APPEARANCE_FIELDS = ("color_r", "color_g", "color_b", "color_a",
                     "roughness", "metallic", "transmission", "ior")


def appearance_vector(appearance_params: Dict) -> List[float]:
    """Normalized 8-dim appearance target from material_truth.appearance_params."""
    c = list(appearance_params.get("base_color", [0.0, 0.0, 0.0, 1.0]))
    return c[:4] + [
        float(appearance_params.get("roughness", 0.0)),
        float(appearance_params.get("metallic", 0.0)),
        float(appearance_params.get("transmission", 0.0)),
        float(appearance_params.get("ior", 1.0)) - 1.0,
    ]


def _read_json(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


@dataclass
class Scene:
    """One scene's loaded record plus convenient target accessors."""

    scene_id: str
    split: str
    shape: str
    directory: str
    record: Dict

    # -- targets (pure-Python) ------------------------------------------- #
    def behavior_target(self, normalize: bool = True) -> List[float]:
        probes = self.record.get("behavior", {}).get("probes", [])
        return P.behavior_vector(probes, normalize=normalize)

    def essence_target(self) -> List[float]:
        norm = self.record.get("physics", {}).get("normalized", {})
        return [float(norm.get(k, 0.0)) for k in ESSENCE_FIELDS]

    def appearance_target(self) -> List[float]:
        ap = self.record.get("material_truth", {}).get("appearance_params", {})
        return appearance_vector(ap)

    def factors_target(self) -> Optional[List[float]]:
        factors = self.record.get("material_truth", {}).get("factors")
        if not factors:
            return None
        return [float(factors[k]) for k in FACTOR_FIELDS]

    # -- appearance (paths now, pixels on demand) ------------------------ #
    def view_files(self) -> List[str]:
        frames = self.record.get("appearance", {}).get("frames", [])
        return [os.path.join(self.directory, "renders", f["file"]) for f in frames]


class PseudoMarbleDataset:
    """Indexable dataset over a generated run, optionally filtered by split."""

    def __init__(self, root: str, split: Optional[str] = None,
                 normalize_targets: bool = True) -> None:
        self.root = root
        self.split = split
        self.normalize_targets = normalize_targets
        self.behavior_dim = P.BEHAVIOR_DIM
        self.scenes: List[Scene] = self._load(root, split)

    @staticmethod
    def _load(root: str, split: Optional[str]) -> List[Scene]:
        manifest_path = os.path.join(root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"no manifest.json in {root!r}; generate a dataset first "
                "(python -m pseudomarble.data.generate_mujoco ...)"
            )
        manifest = _read_json(manifest_path)
        scenes: List[Scene] = []
        for entry in manifest.get("scenes", []):
            if split is not None and entry.get("split") != split:
                continue
            scene_id = entry["scene_id"]
            sample_path = os.path.join(root, scene_id, "sample.json")
            if not os.path.exists(sample_path):
                continue  # skip scenes whose generation was incomplete
            record = _read_json(sample_path)
            scenes.append(Scene(
                scene_id=scene_id,
                split=record.get("split", entry.get("split", "")),
                shape=record.get("input", {}).get("shape", entry.get("shape", "")),
                directory=os.path.join(root, scene_id),
                record=record,
            ))
        return scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, i: int) -> Scene:
        return self.scenes[i]

    # -- bulk target access (no images needed) --------------------------- #
    def behavior_targets(self) -> List[List[float]]:
        return [s.behavior_target(self.normalize_targets) for s in self.scenes]

    def essence_targets(self) -> List[List[float]]:
        return [s.essence_target() for s in self.scenes]

    # -- image loading (lazy; needs numpy + an image reader) ------------- #
    def load_views(self, scene: Scene, max_views: Optional[int] = None):
        """Return an array of shape (n_views, H, W, 3), float32 in [0, 1]."""
        import numpy as np  # type: ignore

        files = scene.view_files()
        if max_views is not None:
            files = files[:max_views]
        if not files:
            raise ValueError(f"scene {scene.scene_id} has no rendered views")
        return np.stack([_load_image(f) for f in files]).astype("float32")

    # -- batching -------------------------------------------------------- #
    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        with_images: bool = False,
        max_views: Optional[int] = None,
        as_mlx: bool = False,
        drop_last: bool = False,
    ) -> Iterator[Dict]:
        """Yield batches of ``{scene_ids, behavior, essence[, images]}``.

        Targets are always provided. Images are loaded only if ``with_images``.
        ``as_mlx`` converts arrays to ``mlx.core.array`` (Apple silicon); without
        it, targets are Python lists and images are numpy arrays.
        """
        order = list(range(len(self.scenes)))
        if shuffle:
            random.Random(seed).shuffle(order)

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if drop_last and len(idx) < batch_size:
                break
            scenes = [self.scenes[i] for i in idx]
            batch: Dict = {
                "scene_ids": [s.scene_id for s in scenes],
                "behavior": [s.behavior_target(self.normalize_targets) for s in scenes],
                "essence": [s.essence_target() for s in scenes],
                "appearance": [s.appearance_target() for s in scenes],
            }
            if with_images:
                import numpy as np  # type: ignore
                batch["images"] = np.stack([self.load_views(s, max_views) for s in scenes])
            if as_mlx:
                batch = _to_mlx(batch)
            yield batch


def _load_image(path: str):
    import numpy as np  # type: ignore

    try:
        import imageio.v3 as iio  # type: ignore

        arr = iio.imread(path)
    except Exception:
        from PIL import Image  # type: ignore

        arr = np.asarray(Image.open(path).convert("RGB"))
    arr = np.asarray(arr)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # drop alpha
        arr = arr[..., :3]
    return arr.astype("float32") / 255.0


def _to_mlx(batch: Dict) -> Dict:
    import mlx.core as mx  # type: ignore

    out = dict(batch)
    out["behavior"] = mx.array(batch["behavior"])
    out["essence"] = mx.array(batch["essence"])
    if "appearance" in batch:
        out["appearance"] = mx.array(batch["appearance"])
    if "images" in batch:
        out["images"] = mx.array(batch["images"])
    return out
