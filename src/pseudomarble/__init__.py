"""pseudo-marble: a constrained, MLX/Metal-native instrument for testing the
core idea behind World Labs' Marble — *one latent, two coupled output paths*
(appearance + physics) — on commodity Apple-silicon hardware.

This is **personal research**. It is not Marble, not affiliated with World Labs,
and makes no claim to reproduce it. It is a thinking instrument: a deliberately
small "pseudo-marble" we can run, perturb, and develop intuition on, using data
sources (Blender) that everyday coders can actually generate.

See ``docs/TAXONOMY_NOTES.md`` for the conceptual lineage and
``docs/ARCHITECTURE.md`` for the design.

The top-level package deliberately imports only dependency-light, pure-Python
modules so it stays importable without ``bpy``, ``trimesh``, or ``mlx``
installed. Heavier modules (Blender data generation, MLX models) are imported
explicitly from their submodules by the tools that need them.
"""

from pseudomarble import materials, splits  # noqa: F401

__all__ = ["materials", "splits"]
__version__ = "0.0.1"
